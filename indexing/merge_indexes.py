"""Merge existing FAISS indexes (same embedder/dim) into a single on-disk index.

Reads per-dataset dirs under indexes/{sbol,musique}/ and writes:
    indexes/unioned/{faiss.index, metadata.jsonl, config.json, stats.json}

Vectors are reconstructed from the source IndexFlatIP indexes and concatenated;
metadata rows keep the same order as vectors (sbol first, then musique by default).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from tool_server.index import FaissIndex, save_index

log = logging.getLogger(__name__)

DEFAULT_INDEX_DIRS = (
    Path("indexes/sbol"),
    Path("indexes/musique"),
)
DEFAULT_OUT_DIR = Path("indexes/unioned")


def _assert_compatible(indexes: list[FaissIndex]) -> None:
    if len(indexes) < 2:
        raise ValueError("need at least two indexes to merge")
    ref = indexes[0]
    for other in indexes[1:]:
        if other.dim != ref.dim:
            raise ValueError(f"dim mismatch: {ref.dim} vs {other.dim}")
        if other.config.get("embedder") != ref.config.get("embedder"):
            raise ValueError(
                f"embedder mismatch: {ref.config.get('embedder')} vs "
                f"{other.config.get('embedder')}"
            )
        if ref.config.get("normalized") != other.config.get("normalized"):
            raise ValueError("normalization flag mismatch between indexes")


def merge_indexes(
    index_dirs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    """Load, concatenate, and persist a merged CPU FAISS index."""
    loaded = [FaissIndex.load(d, use_gpu=False) for d in index_dirs]
    _assert_compatible(loaded)

    dim = loaded[0].dim
    embedder_name = str(loaded[0].config.get("embedder", ""))

    chunks: list[np.ndarray] = []
    merged_meta = []
    per_source: dict[str, int] = {}

    for src_dir, idx in zip(index_dirs, loaded, strict=True):
        n = idx.index.ntotal
        if n != len(idx.metadata):
            raise RuntimeError(f"{src_dir}: {n} vectors vs {len(idx.metadata)} metadata rows")
        vecs = idx.index.reconstruct_n(0, n)
        chunks.append(vecs.astype(np.float32))
        merged_meta.extend(idx.metadata)
        per_source[str(src_dir)] = n
        log.info("loaded %s: %d docs", src_dir, n)

    merged_vecs = np.vstack(chunks)
    merged_index = faiss.IndexFlatIP(dim)
    merged_index.add(merged_vecs)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_index(
        merged_index,
        merged_meta,
        out_dir=out_dir,
        embedder_name=embedder_name,
        dim=dim,
    )

    stats: dict[str, Any] = {
        "out_dir": str(out_dir),
        "embedder": embedder_name,
        "dim": dim,
        "num_docs": len(merged_meta),
        "per_source": per_source,
        "sources": [str(p) for p in index_dirs],
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info(
        "merged %d docs -> %s (%s)",
        stats["num_docs"],
        out_dir,
        ", ".join(f"{k}={v}" for k, v in per_source.items()),
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        action="append",
        dest="index_dirs",
        type=Path,
        help="Source index directory (repeatable; default: sbol then musique).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    index_dirs = args.index_dirs or list(DEFAULT_INDEX_DIRS)
    for d in index_dirs:
        if not (d / "faiss.index").exists():
            raise FileNotFoundError(f"missing faiss.index under {d}")

    merge_indexes(index_dirs, args.out_dir)


if __name__ == "__main__":
    main()
