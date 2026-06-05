"""FastAPI tool server.

Endpoints:
    POST /local_search     — query → ranked passages (semantic + BM25 hybrid via FAISS)
    POST /grep             — pattern → exact-match passages (case-insensitive regex scan)
    POST /lookup_by_id     — fetch docs by doc_id
    POST /reload           — reload index from disk
    GET  /healthz          — liveness (200) / not-ready (503)
    GET  /stats            — index metadata
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import OrderedDict
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from tool_server.embedder import E5Embedder
from tool_server.index import DocMeta, FaissIndex
from tool_server.schemas import (
    Doc,
    GrepHit,
    GrepRequest,
    GrepResponse,
    LocalSearchRequest,
    LocalSearchResponse,
    LookupByIdRequest,
    LookupByIdResponse,
    SearchHit,
    StatsResponse,
)

log = logging.getLogger(__name__)


class State:
    embedder: E5Embedder | None = None
    index: FaissIndex | None = None
    cache: "OrderedDict[tuple[str, int], list[SearchHit]]" = OrderedDict()
    cache_size: int = 10000
    idx_cfg: dict | None = None
    # Singleflight + cooldown for /reload. With ~128 rollout workers each
    # running its own circuit breaker, a single brownout produces a thundering
    # herd of reload requests that each load FAISS from disk, starve the
    # threadpool, and prolong the brownout. The lock collapses concurrent
    # callers onto one reload; the cooldown rejects redundant follow-up
    # reloads as "skipped" so callers treat them as success.
    reload_lock: asyncio.Lock | None = None
    reload_last_finished_at: float = 0.0
    reload_cooldown_s: float = 30.0
    # Bound on concurrent /grep scans. Grep is a pure-Python regex scan over
    # ~100K docs; even though individual scans are sub-second, putting them
    # on the default Starlette threadpool (~40 slots) under 128 agent workers
    # caused threadpool starvation and made /local_search queue indefinitely.
    # The semaphore caps the CPU footprint and protects search latency.
    grep_semaphore: asyncio.Semaphore | None = None
    grep_max_concurrency: int = 4
    grep_timeout_s: float = 10.0
    grep_max_text_chars: int = 4096


def _load_index(idx_cfg: dict) -> FaissIndex:
    return FaissIndex.load(
        Path(idx_cfg["dir"]),
        use_gpu=idx_cfg.get("use_gpu", False),
        gpu_id=idx_cfg.get("gpu_id", 3),
    )


def create_app(config_path: Path) -> FastAPI:
    cfg = yaml.safe_load(config_path.read_text())

    app = FastAPI(title="retrieval-agent tool server")

    @app.on_event("startup")
    def _startup() -> None:
        emb_cfg = cfg["embedder"]
        idx_cfg = cfg["index"]
        State.idx_cfg = idx_cfg
        State.cache_size = idx_cfg.get("cache_size", 10000)
        State.reload_cooldown_s = float(idx_cfg.get("reload_cooldown_s", 30.0))
        State.reload_lock = asyncio.Lock()
        State.grep_max_concurrency = int(idx_cfg.get("grep_max_concurrency", 4))
        State.grep_timeout_s = float(idx_cfg.get("grep_timeout_s", 10.0))
        State.grep_max_text_chars = int(idx_cfg.get("grep_max_text_chars", 4096))
        State.grep_semaphore = asyncio.Semaphore(State.grep_max_concurrency)

        State.embedder = E5Embedder(
            name=emb_cfg["name"],
            device=emb_cfg["device"],
            max_length=emb_cfg["max_length"],
            query_prefix=emb_cfg["query_prefix"],
            passage_prefix=emb_cfg["passage_prefix"],
            pooling=emb_cfg.get("pooling", "mean"),
        )
        State.index = _load_index(idx_cfg)
        log.info(
            "tool server ready: %d docs, embedder=%s",
            State.index.num_docs,
            State.embedder.name,
        )

    @app.post("/reload")
    async def reload_index() -> JSONResponse:
        """Re-load the FAISS index from disk.

        Called by the agent loop when its circuit breaker trips (after N
        consecutive tool failures). Fixes in-memory corruption and picks up
        any on-disk index rebuild.

        Concurrency contract:
          * At most one reload runs at a time (``State.reload_lock``).
          * If another caller finished a reload within ``reload_cooldown_s``,
            return ``{"status": "skipped"}`` immediately so the herd dissolves
            after the first successful reload instead of each worker triggering
            its own.
          * The blocking ``_load_index`` runs on a worker thread so the event
            loop stays responsive to ``/local_search`` and ``/grep`` requests
            queued during the reload.
        """
        if State.idx_cfg is None or State.reload_lock is None:
            raise HTTPException(503, "server not initialised")

        # Fast path: someone just reloaded; treat as success without touching state.
        if (
            State.reload_last_finished_at > 0.0
            and (time.monotonic() - State.reload_last_finished_at)
            < State.reload_cooldown_s
        ):
            assert State.index is not None
            return JSONResponse(
                {
                    "status": "skipped",
                    "reason": "recent_reload",
                    "num_docs": State.index.num_docs,
                }
            )

        async with State.reload_lock:
            # Re-check inside the lock: a sibling caller may have just reloaded
            # while we were waiting for the lock.
            if (
                State.reload_last_finished_at > 0.0
                and (time.monotonic() - State.reload_last_finished_at)
                < State.reload_cooldown_s
            ):
                assert State.index is not None
                return JSONResponse(
                    {
                        "status": "skipped",
                        "reason": "recent_reload",
                        "num_docs": State.index.num_docs,
                    }
                )

            idx_cfg = State.idx_cfg
            try:
                new_index = await asyncio.to_thread(_load_index, idx_cfg)
            except Exception as exc:
                log.exception("reload failed")
                raise HTTPException(500, f"reload failed: {exc}") from exc

            State.cache.clear()
            State.index = new_index
            State.reload_last_finished_at = time.monotonic()
            log.info("index reloaded: %d docs", new_index.num_docs)
            return JSONResponse({"status": "ok", "num_docs": new_index.num_docs})

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        # Declared async so Starlette serves it directly on the event loop
        # instead of routing through the anyio threadpool (default ~40
        # threads). With 128 rollout workers all calling /healthz at training
        # startup, threadpool queueing behind in-flight /local_search or
        # /grep calls can push this trivial check past the client timeout.
        if State.embedder is None or State.index is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        # Same async-on-event-loop rationale as /healthz: this is a metadata
        # read, no need to occupy a threadpool slot.
        if State.embedder is None or State.index is None:
            raise HTTPException(503, "not ready")
        return StatsResponse(
            num_docs=State.index.num_docs,
            embedder=State.embedder.name,
            dim=State.index.dim,
            index_type=State.index.index_type,
        )

    @app.post("/local_search", response_model=LocalSearchResponse)
    async def local_search(req: LocalSearchRequest) -> LocalSearchResponse:
        # async + asyncio.to_thread so the CPU/GPU work runs on the default
        # asyncio executor instead of stealing slots from Starlette's anyio
        # threadpool (which also serves /grep). Keeps /local_search latency
        # independent of grep load.
        if State.embedder is None or State.index is None:
            raise HTTPException(503, "not ready")

        t0 = time.perf_counter()
        cache_key = (req.query, req.top_k)
        cached = State.cache.get(cache_key)
        if cached is not None:
            State.cache.move_to_end(cache_key)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LocalSearchResponse(results=cached, latency_ms=latency_ms)

        def _run_search() -> list[SearchHit]:
            assert State.embedder is not None and State.index is not None
            emb = State.embedder.encode_queries([req.query])
            scores, ids = State.index.search(emb, req.top_k)
            out: list[SearchHit] = []
            for score, pos in zip(scores[0].tolist(), ids[0].tolist()):
                if pos < 0:
                    continue
                meta = State.index.lookup_by_position(pos)
                out.append(
                    SearchHit(
                        doc_id=meta.doc_id,
                        title=meta.title,
                        text=meta.text,
                        score=float(score),
                    )
                )
            return out

        hits = await asyncio.to_thread(_run_search)

        State.cache[cache_key] = hits
        if len(State.cache) > State.cache_size:
            State.cache.popitem(last=False)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return LocalSearchResponse(results=hits, latency_ms=latency_ms)

    @app.post("/grep", response_model=GrepResponse)
    async def grep(req: GrepRequest) -> GrepResponse:
        # async + bounded concurrency + wall-clock budget.
        # Why: pure-Python regex over the full corpus is sequential per call,
        # but the previous sync `def` handler made every concurrent call
        # occupy a Starlette threadpool slot for the full scan, starving
        # /local_search. The semaphore caps the CPU footprint of grep; the
        # deadline inside FaissIndex.grep prevents a pathological pattern
        # from holding the slot for a long time.
        if State.index is None or State.grep_semaphore is None:
            raise HTTPException(503, "not ready")

        t0 = time.perf_counter()
        deadline = time.monotonic() + State.grep_timeout_s

        async with State.grep_semaphore:
            def _run_grep() -> tuple[list[DocMeta], int, bool]:
                assert State.index is not None
                return State.index.grep(
                    req.pattern,
                    req.top_k,
                    deadline_monotonic=deadline,
                    max_text_chars=State.grep_max_text_chars,
                )

            hits, total, completed = await asyncio.to_thread(_run_grep)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        if not completed:
            log.warning(
                "grep deadline (%.1fs) hit; pattern=%r partial total=%d hits=%d",
                State.grep_timeout_s,
                req.pattern,
                total,
                len(hits),
            )
        return GrepResponse(
            results=[GrepHit(doc_id=m.doc_id, title=m.title, text=m.text) for m in hits],
            latency_ms=latency_ms,
            total_matches=total,
        )

    @app.post("/lookup_by_id", response_model=LookupByIdResponse)
    def lookup_by_id(req: LookupByIdRequest) -> LookupByIdResponse:
        if State.index is None:
            raise HTTPException(503, "not ready")
        docs: list[Doc] = []
        for doc_id in req.doc_ids:
            meta = State.index.lookup_by_doc_id(doc_id)
            if meta is None:
                continue
            docs.append(Doc(doc_id=meta.doc_id, title=meta.title, text=meta.text))
        return LookupByIdResponse(docs=docs)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    ts_cfg = cfg.get("tool_server", {})
    host = args.host or ts_cfg.get("host", "0.0.0.0")
    port = args.port or ts_cfg.get("port", 8100)

    import uvicorn

    app = create_app(Path(args.config))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
