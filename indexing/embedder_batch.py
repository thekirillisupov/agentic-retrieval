"""Batched-encoding helper for the indexing pipeline.

Thin wrapper that streams a list of texts through E5Embedder.encode_passages
with a tqdm progress bar. Kept separate from the embedder itself so the tool
server doesn't need to depend on tqdm.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from tool_server.embedder import E5Embedder

log = logging.getLogger(__name__)


def encode_passages_streaming(
    embedder: E5Embedder,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    out = np.empty((len(texts), embedder.dim), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        chunk = texts[i : i + batch_size]
        out[i : i + len(chunk)] = embedder.encode_passages(chunk, batch_size=batch_size)
    return out
