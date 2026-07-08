"""FastAPI tool server.

Endpoints:
    POST /local_search     — query → ranked passages (semantic + BM25 hybrid via FAISS)
    POST /grep             — pattern → exact-match passages (case-insensitive regex scan)
    POST /get_neighbours   — doc_id → adjacent chunks from the same source document
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
from tool_server.reranker import RerankError, Reranker
from tool_server.schemas import (
    Doc,
    GetNeighboursRequest,
    GetNeighboursResponse,
    GrepHit,
    GrepRequest,
    GrepResponse,
    LocalSearchRequest,
    LocalSearchResponse,
    LookupByIdRequest,
    LookupByIdResponse,
    NeighbourHit,
    SearchHit,
    StatsResponse,
)

log = logging.getLogger(__name__)


class State:
    embedder: E5Embedder | None = None
    # Optional cross-encoder reranker (Qwen3-Reranker served by vLLM). When set,
    # /local_search retrieves ``rerank_candidates`` FAISS hits, reranks them, and
    # returns the top_k. None -> pure FAISS ranking (reranker disabled).
    reranker: "Reranker | None" = None
    # How many FAISS candidates to fetch before reranking. The reranker only
    # reorders within this pool, so it must be >= the requested top_k to add any
    # value; bigger pools trade reranker latency for recall.
    rerank_candidates: int = 50
    # One FAISS index per source (e.g. {"musique": ..., "sbol": ...}). Each may
    # live on a different GPU (see per-source gpu_id in config). The embedder is
    # shared across all sources — the query is encoded once and searched against
    # the selected index.
    indexes: "dict[str, FaissIndex]" = {}
    # Source used when a request omits ``source`` (single-corpus configs and
    # callers that don't route per-row).
    default_source: str | None = None
    # Cache key is (source, query, top_k) so different corpora never collide.
    cache: "OrderedDict[tuple[str, str, int], list[SearchHit]]" = OrderedDict()
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


def _load_indexes(idx_cfg: dict) -> tuple[dict[str, FaissIndex], str]:
    """Load every configured source index, each on its own GPU.

    Two config shapes are supported:

      * Multi-source (preferred)::

            index:
              default_source: musique
              use_gpu: true            # default for sources that omit it
              sources:
                musique: {dir: ./indexes/musique/, gpu_id: 0}
                sbol:    {dir: ./indexes/sbol/,    gpu_id: 1}

        Each source loads its FAISS index onto its own ``gpu_id`` (a *visible*
        device ordinal — see scripts/serve_tool.sh / CUDA_VISIBLE_DEVICES).

      * Single-index (legacy)::

            index:
              dir: ./indexes/unioned/
              use_gpu: false
              gpu_id: 3

        Registered under ``default_source`` (or the literal key ``"default"``)
        so the rest of the server is source-agnostic.

    Returns ``(indexes, default_source)``.
    """
    sources_cfg = idx_cfg.get("sources")
    indexes: dict[str, FaissIndex] = {}

    if sources_cfg:
        default_use_gpu = bool(idx_cfg.get("use_gpu", False))
        default_gpu_id = int(idx_cfg.get("gpu_id", 0))
        for name, sc in sources_cfg.items():
            sc = sc or {}
            indexes[name] = FaissIndex.load(
                Path(sc["dir"]),
                use_gpu=bool(sc.get("use_gpu", default_use_gpu)),
                gpu_id=int(sc.get("gpu_id", default_gpu_id)),
            )
            log.info(
                "loaded source %r: %d docs (gpu=%s id=%s)",
                name,
                indexes[name].num_docs,
                sc.get("use_gpu", default_use_gpu),
                sc.get("gpu_id", default_gpu_id),
            )
        default_source = idx_cfg.get("default_source") or next(iter(sources_cfg))
    else:
        name = idx_cfg.get("default_source", "default")
        indexes[name] = FaissIndex.load(
            Path(idx_cfg["dir"]),
            use_gpu=idx_cfg.get("use_gpu", False),
            gpu_id=idx_cfg.get("gpu_id", 3),
        )
        default_source = name

    if default_source not in indexes:
        raise RuntimeError(
            f"default_source {default_source!r} not among loaded sources "
            f"{list(indexes)}"
        )
    return indexes, default_source


def _resolve_source(source: str | None) -> str:
    """Map a request's ``source`` to a loaded index key.

    Precedence: explicit source (must be loaded) -> default_source. When only
    one index is loaded we tolerate a mismatched/garbage source by falling back
    to that single index, so single-corpus configs never reject a stray arg.
    """
    if not source:
        assert State.default_source is not None
        return State.default_source
    if source in State.indexes:
        return source
    if len(State.indexes) == 1:
        return next(iter(State.indexes))
    raise HTTPException(
        400,
        f"unknown source {source!r}; available={sorted(State.indexes)}",
    )


async def _maybe_rerank(
    query: str, candidates: list[SearchHit], top_k: int
) -> list[SearchHit]:
    """Rerank FAISS candidates with the cross-encoder, returning the top_k.

    Fail-soft: if the reranker is disabled or errors out, return the FAISS order
    truncated to ``top_k``. A reranker brownout must degrade quality gracefully,
    never fail the search (which would starve rewards under a 128-worker rollout).
    The ``SearchHit.score`` of returned hits is replaced with the reranker's
    relevance score so downstream consumers see the ranking signal that ordered
    them.
    """
    if State.reranker is None or not candidates:
        return candidates[:top_k]

    try:
        ranked = await State.reranker.rerank(query, [c.text for c in candidates], top_k)
    except RerankError as exc:
        log.warning("rerank failed (%s); falling back to FAISS order", exc)
        return candidates[:top_k]

    out: list[SearchHit] = []
    for idx, score in ranked:
        hit = candidates[idx]
        out.append(hit.model_copy(update={"score": float(score)}))
    return out


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
            dtype=emb_cfg.get("dtype"),
            add_eos=emb_cfg.get("add_eos", False),
            padding_side=emb_cfg.get("padding_side"),
            device_map=emb_cfg.get("device_map"),
            trust_remote_code=emb_cfg.get("trust_remote_code", False),
            attn_implementation=emb_cfg.get("attn_implementation"),
        )
        State.indexes, State.default_source = _load_indexes(idx_cfg)

        rr_cfg = cfg.get("reranker") or {}
        if rr_cfg.get("enable"):
            State.rerank_candidates = int(rr_cfg.get("candidates", 50))
            State.reranker = Reranker(
                url=rr_cfg["url"],
                model=rr_cfg["model"],
                timeout_s=float(rr_cfg.get("timeout_s", 30.0)),
            )
            log.info(
                "reranker enabled: model=%s url=%s candidates=%d",
                rr_cfg["model"],
                rr_cfg["url"],
                State.rerank_candidates,
            )

        log.info(
            "tool server ready: sources=%s (default=%s), %d docs total, embedder=%s, reranker=%s",
            {name: idx.num_docs for name, idx in State.indexes.items()},
            State.default_source,
            sum(idx.num_docs for idx in State.indexes.values()),
            State.embedder.name,
            State.reranker.model if State.reranker else None,
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if State.reranker is not None:
            await State.reranker.aclose()

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
          * The blocking ``_load_indexes`` runs on a worker thread so the event
            loop stays responsive to ``/local_search`` and ``/grep`` requests
            queued during the reload.
        """
        if State.idx_cfg is None or State.reload_lock is None:
            raise HTTPException(503, "server not initialised")

        def _total_docs() -> int:
            return sum(idx.num_docs for idx in State.indexes.values())

        # Fast path: someone just reloaded; treat as success without touching state.
        if (
            State.reload_last_finished_at > 0.0
            and (time.monotonic() - State.reload_last_finished_at)
            < State.reload_cooldown_s
        ):
            return JSONResponse(
                {
                    "status": "skipped",
                    "reason": "recent_reload",
                    "num_docs": _total_docs(),
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
                return JSONResponse(
                    {
                        "status": "skipped",
                        "reason": "recent_reload",
                        "num_docs": _total_docs(),
                    }
                )

            idx_cfg = State.idx_cfg
            try:
                new_indexes, new_default = await asyncio.to_thread(
                    _load_indexes, idx_cfg
                )
            except Exception as exc:
                log.exception("reload failed")
                raise HTTPException(500, f"reload failed: {exc}") from exc

            State.cache.clear()
            State.indexes = new_indexes
            State.default_source = new_default
            State.reload_last_finished_at = time.monotonic()
            total = sum(idx.num_docs for idx in new_indexes.values())
            log.info(
                "indexes reloaded: %d docs across %d sources", total, len(new_indexes)
            )
            return JSONResponse({"status": "ok", "num_docs": total})

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        # Declared async so Starlette serves it directly on the event loop
        # instead of routing through the anyio threadpool (default ~40
        # threads). With 128 rollout workers all calling /healthz at training
        # startup, threadpool queueing behind in-flight /local_search or
        # /grep calls can push this trivial check past the client timeout.
        if State.embedder is None or not State.indexes:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        # Same async-on-event-loop rationale as /healthz: this is a metadata
        # read, no need to occupy a threadpool slot.
        if State.embedder is None or not State.indexes:
            raise HTTPException(503, "not ready")
        any_index = next(iter(State.indexes.values()))
        return StatsResponse(
            num_docs=sum(idx.num_docs for idx in State.indexes.values()),
            embedder=State.embedder.name,
            dim=any_index.dim,
            index_type=any_index.index_type,
            default_source=State.default_source,
            sources={name: idx.num_docs for name, idx in State.indexes.items()},
        )

    @app.post("/local_search", response_model=LocalSearchResponse)
    async def local_search(req: LocalSearchRequest) -> LocalSearchResponse:
        # async + asyncio.to_thread so the CPU/GPU work runs on the default
        # asyncio executor instead of stealing slots from Starlette's anyio
        # threadpool (which also serves /grep). Keeps /local_search latency
        # independent of grep load.
        if State.embedder is None or not State.indexes:
            raise HTTPException(503, "not ready")

        source = _resolve_source(req.source)
        index = State.indexes[source]

        t0 = time.perf_counter()
        cache_key = (source, req.query, req.top_k)
        cached = State.cache.get(cache_key)
        if cached is not None:
            State.cache.move_to_end(cache_key)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LocalSearchResponse(results=cached, latency_ms=latency_ms)

        # With a reranker, FAISS is only the first stage: pull a larger candidate
        # pool, then let the cross-encoder pick the final top_k from it.
        retrieve_k = req.top_k
        if State.reranker is not None:
            retrieve_k = max(req.top_k, State.rerank_candidates)
        retrieve_k = min(retrieve_k, index.num_docs)

        def _run_search() -> list[SearchHit]:
            assert State.embedder is not None
            emb = State.embedder.encode_queries([req.query])
            scores, ids = index.search(emb, retrieve_k)
            out: list[SearchHit] = []
            for score, pos in zip(scores[0].tolist(), ids[0].tolist()):
                if pos < 0:
                    continue
                meta = index.lookup_by_position(pos)
                out.append(
                    SearchHit(
                        doc_id=meta.doc_id,
                        title=meta.title,
                        text=meta.text,
                        score=float(score),
                        file_name=meta.file_name,
                        index=meta.index,
                    )
                )
            return out

        candidates = await asyncio.to_thread(_run_search)
        hits = await _maybe_rerank(req.query, candidates, req.top_k)

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
        if not State.indexes or State.grep_semaphore is None:
            raise HTTPException(503, "not ready")

        source = _resolve_source(req.source)
        index = State.indexes[source]

        t0 = time.perf_counter()
        deadline = time.monotonic() + State.grep_timeout_s

        async with State.grep_semaphore:

            def _run_grep() -> tuple[list[DocMeta], int, bool]:
                return index.grep(
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
            results=[
                GrepHit(doc_id=m.doc_id, title=m.title, text=m.text) for m in hits
            ],
            latency_ms=latency_ms,
            total_matches=total,
        )

    @app.post("/get_neighbours", response_model=GetNeighboursResponse)
    async def get_neighbours(req: GetNeighboursRequest) -> GetNeighboursResponse:
        # Adjacency lookup is a pure dict/index walk (no embedding, no scan), so
        # it is cheap; still run off the event loop for consistency with search.
        if not State.indexes:
            raise HTTPException(503, "not ready")

        source = _resolve_source(req.source)
        index = State.indexes[source]

        t0 = time.perf_counter()
        neighbours, status = await asyncio.to_thread(
            index.get_neighbours, req.doc_id, req.window
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return GetNeighboursResponse(
            results=[
                NeighbourHit(
                    doc_id=m.doc_id,
                    title=m.title,
                    text=m.text,
                    file_name=m.file_name,
                    index=m.index,
                )
                for m in neighbours
            ],
            status=status,
            anchor_doc_id=req.doc_id,
            window=req.window,
            latency_ms=latency_ms,
        )

    @app.post("/lookup_by_id", response_model=LookupByIdResponse)
    def lookup_by_id(req: LookupByIdRequest) -> LookupByIdResponse:
        if not State.indexes:
            raise HTTPException(503, "not ready")
        # doc_ids are corpus-unique, so scan every loaded source and return the
        # first match. Cheap: it's a dict lookup per index.
        docs: list[Doc] = []
        for doc_id in req.doc_ids:
            for index in State.indexes.values():
                meta = index.lookup_by_doc_id(doc_id)
                if meta is not None:
                    docs.append(
                        Doc(doc_id=meta.doc_id, title=meta.title, text=meta.text)
                    )
                    break
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
