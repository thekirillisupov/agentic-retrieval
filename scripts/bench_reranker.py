"""Quick latency benchmark for the vLLM reranker /v1/rerank endpoint.

Usage: python scripts/bench_reranker.py [--url ...] [--docs 8 50] [--iters 20]
Reports warm-run mean/p50/p90 latency per doc-count.
"""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

_DOC = (
    "The mitochondria is the powerhouse of the cell, generating most of the "
    "cell's supply of adenosine triphosphate used as a source of chemical "
    "energy. Beyond energy production, mitochondria are involved in signaling, "
    "cellular differentiation, and apoptosis, as well as maintaining control of "
    "the cell cycle and cell growth. " * 2
)
_QUERY = "What organelle produces ATP and regulates apoptosis in the cell?"


def bench(url: str, model: str, n_docs: int, iters: int, warmup: int) -> dict:
    docs = [f"[doc {i}] {_DOC}" for i in range(n_docs)]
    payload = {"model": model, "query": _QUERY, "documents": docs, "top_n": n_docs}
    lats: list[float] = []
    with httpx.Client(timeout=120.0) as c:
        for i in range(warmup + iters):
            t0 = time.perf_counter()
            r = c.post(url, json=payload)
            r.raise_for_status()
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                lats.append(dt)
    lats.sort()
    return {
        "n_docs": n_docs,
        "iters": len(lats),
        "mean_ms": round(statistics.mean(lats), 1),
        "p50_ms": round(lats[len(lats) // 2], 1),
        "p90_ms": round(lats[int(len(lats) * 0.9)], 1),
        "min_ms": round(lats[0], 1),
        "max_ms": round(lats[-1], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8200/v1/rerank")
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--docs", type=int, nargs="+", default=[8, 50])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    for n in args.docs:
        res = bench(args.url, args.model, n, args.iters, args.warmup)
        print(res)


if __name__ == "__main__":
    main()
