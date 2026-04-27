"""Faiss wrapper.

Layout on disk:
    {dir}/faiss.index        # binary
    {dir}/metadata.jsonl     # i-th line ↔ i-th vector
    {dir}/config.json        # embedder name / dim / normalization

The CPU/GPU split is one line: see `build_index`. On MuSiQue scale (~50k passages
after dedup) CPU IndexFlatIP latency is ~5–20ms, well under generation latency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

log = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.jsonl"
CONFIG_FILENAME = "config.json"


@dataclass
class DocMeta:
    doc_id: str
    title: str
    text: str


def build_index(dim: int, *, use_gpu: bool = False, gpu_id: int = 3) -> faiss.Index:
    cpu_index = faiss.IndexFlatIP(dim)
    if not use_gpu:
        return cpu_index
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)


def save_index(
    index: faiss.Index,
    metadata: list[DocMeta],
    *,
    out_dir: Path,
    embedder_name: str,
    dim: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always serialize the CPU index — faiss-gpu indexes need to be moved back first.
    cpu_index = index
    if hasattr(faiss, "index_gpu_to_cpu"):
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index
    faiss.write_index(cpu_index, str(out_dir / INDEX_FILENAME))

    with (out_dir / METADATA_FILENAME).open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(
                json.dumps(
                    {"doc_id": m.doc_id, "title": m.title, "text": m.text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    config = {
        "embedder": embedder_name,
        "dim": dim,
        "normalized": True,
        "index_type": type(cpu_index).__name__,
        "num_docs": len(metadata),
    }
    (out_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2))
    log.info("wrote index (%d docs) to %s", len(metadata), out_dir)


class FaissIndex:
    """Loaded, query-ready faiss index + metadata."""

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[DocMeta],
        config: dict[str, Any],
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.config = config
        self.doc_id_to_pos: dict[str, int] = {
            m.doc_id: i for i, m in enumerate(metadata)
        }

    @property
    def num_docs(self) -> int:
        return len(self.metadata)

    @property
    def dim(self) -> int:
        return int(self.config["dim"])

    @property
    def index_type(self) -> str:
        return str(self.config.get("index_type", type(self.index).__name__))

    @classmethod
    def load(
        cls, dir_path: Path, *, use_gpu: bool = False, gpu_id: int = 3
    ) -> "FaissIndex":
        config = json.loads((dir_path / CONFIG_FILENAME).read_text())
        cpu_index = faiss.read_index(str(dir_path / INDEX_FILENAME))

        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        else:
            index = cpu_index

        metadata: list[DocMeta] = []
        with (dir_path / METADATA_FILENAME).open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                metadata.append(DocMeta(d["doc_id"], d["title"], d["text"]))

        if len(metadata) != index.ntotal:
            raise RuntimeError(
                f"metadata/index mismatch: {len(metadata)} rows vs {index.ntotal} vectors"
            )
        return cls(index, metadata, config)

    def search(
        self, queries: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        scores, ids = self.index.search(queries, top_k)
        return scores, ids

    def lookup_by_position(self, pos: int) -> DocMeta:
        return self.metadata[pos]

    def lookup_by_doc_id(self, doc_id: str) -> DocMeta | None:
        pos = self.doc_id_to_pos.get(doc_id)
        if pos is None:
            return None
        return self.metadata[pos]
