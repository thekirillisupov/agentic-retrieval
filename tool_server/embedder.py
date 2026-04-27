"""E5 embedder wrapper.

Encapsulates the `query: ` / `passage: ` prefix discipline. Callers pass plain
text — the embedder is the only place that knows about prefixes. This means the
indexing pipeline and the tool server can never accidentally mix conventions.

L2-normalization is enforced here too, so the index can use IndexFlatIP and get
cosine similarity for free.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


class E5Embedder:
    def __init__(
        self,
        name: str = "intfloat/e5-large-v2",
        *,
        device: str = "cuda:0",
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self.name = name
        self.device = device
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        log.info("loading embedder %s on %s", name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()

        with torch.no_grad():
            test = self._encode_texts([self.passage_prefix + "test"])
        self.dim = int(test.shape[1])
        log.info("embedder dim=%d", self.dim)

    @torch.no_grad()
    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [self.query_prefix + q for q in queries]
        return self._encode_batched(prefixed, batch_size)

    @torch.no_grad()
    def encode_passages(self, passages: list[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [self.passage_prefix + p for p in passages]
        return self._encode_batched(prefixed, batch_size)

    def _encode_batched(self, texts: list[str], batch_size: int) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            out.append(self._encode_texts(chunk))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**batch)
        emb = _average_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def count_truncated(self, texts: Iterable[str]) -> int:
        """Count how many passages exceed max_length tokens (without prefix offset).

        Used during indexing to monitor whether E5's 512-token limit is biting.
        """
        n = 0
        for t in texts:
            ids = self.tokenizer.encode(self.passage_prefix + t, add_special_tokens=True)
            if len(ids) > self.max_length:
                n += 1
        return n


def _average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask[..., None].bool()
    last = last_hidden_states.masked_fill(~mask, 0.0)
    return last.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp(min=1)
