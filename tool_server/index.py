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
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

log = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.jsonl"
CONFIG_FILENAME = "config.json"

# Characters that, if present in a /grep pattern, mean we must treat the
# pattern as a regex. If none are present we can use the literal-substring
# fast path (``str.find``-based), which is 5-20x faster on the same corpus
# because the C-level substring search does no backtracking and does not
# go through the Python regex VM.
_REGEX_METACHARS = frozenset(r".^$*+?{}[]|()\\")


def _is_literal_pattern(pattern: str) -> bool:
    return not any(c in _REGEX_METACHARS for c in pattern)


@dataclass
class DocMeta:
    doc_id: str
    title: str
    text: str
    # Adjacency metadata for sliced chunks. Both are None for "independent"
    # chunks and for legacy indexes built before chunk slicing existed. When
    # set, ``file_name`` is the source document a chunk was cut from and
    # ``index`` is its 0-based position within that document.
    file_name: str | None = None
    index: int | None = None


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
            rec: dict[str, Any] = {"doc_id": m.doc_id, "title": m.title, "text": m.text}
            # Only emit adjacency fields when present, so legacy corpora keep the
            # exact same metadata.jsonl shape as before.
            if m.file_name is not None:
                rec["file_name"] = m.file_name
            if m.index is not None:
                rec["index"] = m.index
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
        # Adjacency map for get_neighbours: file_name -> [(index, pos), ...]
        # sorted by index. Only built from chunks that carry both file_name and
        # index, so legacy corpora produce an empty map (-> supports_neighbours
        # is False) and get_neighbours degrades to a no-op.
        self._file_to_chunks: dict[str, list[tuple[int, int]]] = {}
        for pos, m in enumerate(metadata):
            if m.file_name is not None and m.index is not None:
                self._file_to_chunks.setdefault(m.file_name, []).append((m.index, pos))
        for chunks in self._file_to_chunks.values():
            chunks.sort()
        self.supports_neighbours: bool = bool(self._file_to_chunks)
        # Precomputed casefolded title/text used by the literal-pattern fast
        # path in ``grep``. Casefolding once at load time (a few hundred ms on
        # the unioned corpus) avoids re-folding ~50 MB of text on every grep
        # call, which would otherwise dominate the literal-path latency.
        # Memory cost is ~+50 MB; we accept that for the much better grep
        # latency under concurrent load.
        t0 = time.monotonic()
        self._titles_cf: list[str] = [m.title.casefold() for m in metadata]
        self._texts_cf: list[str] = [m.text.casefold() for m in metadata]
        log.info(
            "casefolded haystack ready in %.2fs (%d docs)",
            time.monotonic() - t0,
            len(metadata),
        )

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
                metadata.append(
                    DocMeta(
                        d["doc_id"],
                        d["title"],
                        d["text"],
                        d.get("file_name"),
                        d.get("index"),
                    )
                )

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

    def get_neighbours(
        self, doc_id: str, window: int = 1
    ) -> tuple[list[DocMeta], str]:
        """Return chunks adjacent to ``doc_id`` within +/-``window`` of its index.

        The status string tells the caller *why* there are no neighbours so the
        rendered tool message can be honest with the model:

          * ``"unknown_doc"``       -- doc_id not in this corpus.
          * ``"no_metadata"``       -- corpus carries no adjacency info at all
                                       (e.g. a legacy index built before chunk
                                       slicing). get_neighbours is a no-op here.
          * ``"independent_chunk"`` -- corpus supports adjacency but this doc is
                                       an independent chunk (no file_name/index).
          * ``"ok"``                -- results hold the neighbours (possibly
                                       empty if none fall inside the window).

        The anchor chunk itself is excluded; results are ordered by index.
        """
        pos = self.doc_id_to_pos.get(doc_id)
        if pos is None:
            return [], "unknown_doc"
        if not self.supports_neighbours:
            return [], "no_metadata"
        anchor = self.metadata[pos]
        if anchor.file_name is None or anchor.index is None:
            return [], "independent_chunk"
        window = max(0, window)
        lo, hi = anchor.index - window, anchor.index + window
        neighbours = [
            self.metadata[p]
            for idx, p in self._file_to_chunks.get(anchor.file_name, [])
            if lo <= idx <= hi and idx != anchor.index
        ]
        return neighbours, "ok"

    def grep(
        self,
        pattern: str,
        top_k: int,
        *,
        deadline_monotonic: float | None = None,
        max_text_chars: int = 4096,
        deadline_check_every: int = 1024,
    ) -> tuple[list[DocMeta], int, bool]:
        """Case-insensitive search across all document titles and text.

        Two execution paths:

          * **Literal fast path** (``_is_literal_pattern(pattern)`` is True):
            uses precomputed casefolded title/text and Python's C-level
            ``str.__contains__`` (Crochemore-Perrin two-way string search),
            which is 5-20x faster than the regex VM on the same corpus and
            cannot backtrack. Covers the common "agent searched for a
            literal word or phrase" case (~70% of observed traffic).
          * **Regex slow path**: falls back to ``re.compile(...).search``
            with ``re.IGNORECASE``. If the pattern fails to compile it is
            re-escaped and retried as a literal.

        Safety knobs (added because pure-Python regex scans over the full
        corpus can be hijacked by pathological patterns or simply pile up
        under load and wedge the FastAPI threadpool):

          * ``max_text_chars``: only the leading ``max_text_chars`` of each
            document's text are scanned. Title is always scanned in full.
            Titles are ~18 chars on average and texts ~444 chars (p95 ≈ 1 KB,
            max ≈ 6 KB), so 4096 is effectively no truncation in the common
            case while bounding the worst case.
          * ``deadline_monotonic``: ``time.monotonic()`` wall-clock cutoff.
            When the deadline is exceeded the scan stops and partial results
            are returned with ``completed=False``. The check is amortised to
            once per ``deadline_check_every`` documents so it adds negligible
            overhead in the happy path.

        Returns ``(hits[:top_k], total_match_count, completed)``. When
        ``completed`` is False, ``total_match_count`` reflects matches found
        only up to the point the deadline fired.
        """
        use_literal = _is_literal_pattern(pattern)
        needle_cf = pattern.casefold() if use_literal else ""
        compiled: re.Pattern[str] | None = None
        if not use_literal:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error:
                compiled = re.compile(re.escape(pattern), re.IGNORECASE)

        hits: list[DocMeta] = []
        total = 0
        completed = True
        check_every = max(1, int(deadline_check_every))
        titles_cf = self._titles_cf
        texts_cf = self._texts_cf
        for i, meta in enumerate(self.metadata):
            if (
                deadline_monotonic is not None
                and (i % check_every) == 0
                and time.monotonic() > deadline_monotonic
            ):
                completed = False
                break
            if use_literal:
                t_cf = titles_cf[i]
                x_cf = texts_cf[i]
                if len(x_cf) > max_text_chars:
                    x_cf = x_cf[:max_text_chars]
                matched = needle_cf in t_cf or needle_cf in x_cf
            else:
                text = meta.text if len(meta.text) <= max_text_chars else meta.text[:max_text_chars]
                assert compiled is not None
                matched = bool(compiled.search(meta.title) or compiled.search(text))
            if matched:
                total += 1
                if len(hits) < top_k:
                    hits.append(meta)
        return hits, total, completed
