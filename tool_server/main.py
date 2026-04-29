"""FastAPI tool server.

Endpoints:
    POST /local_search     — query → ranked passages
    POST /lookup_by_id     — TODO: stub, not implemented in MVP
    GET  /healthz          — liveness (200) / not-ready (503)
    GET  /stats            — index metadata
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import OrderedDict
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from tool_server.embedder import E5Embedder
from tool_server.index import FaissIndex
from tool_server.schemas import (
    Doc,
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


def create_app(config_path: Path) -> FastAPI:
    cfg = yaml.safe_load(config_path.read_text())

    app = FastAPI(title="retrieval-agent tool server")

    @app.on_event("startup")
    def _startup() -> None:
        emb_cfg = cfg["embedder"]
        idx_cfg = cfg["index"]
        State.cache_size = idx_cfg.get("cache_size", 10000)

        State.embedder = E5Embedder(
            name=emb_cfg["name"],
            device=emb_cfg["device"],
            max_length=emb_cfg["max_length"],
            query_prefix=emb_cfg["query_prefix"],
            passage_prefix=emb_cfg["passage_prefix"],
            pooling=emb_cfg.get("pooling", "mean"),
        )
        State.index = FaissIndex.load(
            Path(idx_cfg["dir"]),
            use_gpu=idx_cfg.get("use_gpu", False),
            gpu_id=idx_cfg.get("gpu_id", 3),
        )
        log.info(
            "tool server ready: %d docs, embedder=%s",
            State.index.num_docs,
            State.embedder.name,
        )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        if State.embedder is None or State.index is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        if State.embedder is None or State.index is None:
            raise HTTPException(503, "not ready")
        return StatsResponse(
            num_docs=State.index.num_docs,
            embedder=State.embedder.name,
            dim=State.index.dim,
            index_type=State.index.index_type,
        )

    @app.post("/local_search", response_model=LocalSearchResponse)
    def local_search(req: LocalSearchRequest) -> LocalSearchResponse:
        if State.embedder is None or State.index is None:
            raise HTTPException(503, "not ready")

        t0 = time.perf_counter()
        cache_key = (req.query, req.top_k)
        cached = State.cache.get(cache_key)
        if cached is not None:
            State.cache.move_to_end(cache_key)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return LocalSearchResponse(results=cached, latency_ms=latency_ms)

        emb = State.embedder.encode_queries([req.query])
        scores, ids = State.index.search(emb, req.top_k)

        hits: list[SearchHit] = []
        for score, pos in zip(scores[0].tolist(), ids[0].tolist()):
            if pos < 0:
                continue
            meta = State.index.lookup_by_position(pos)
            hits.append(
                SearchHit(
                    doc_id=meta.doc_id,
                    title=meta.title,
                    text=meta.text,
                    score=float(score),
                )
            )

        State.cache[cache_key] = hits
        if len(State.cache) > State.cache_size:
            State.cache.popitem(last=False)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return LocalSearchResponse(results=hits, latency_ms=latency_ms)

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
