"""Text embedder wrapper.

Encapsulates prefix discipline and pooling strategy. Callers pass plain text —
the embedder is the only place that knows about prefixes and pooling. This means
the indexing pipeline and the tool server can never accidentally mix conventions.

Supported pooling modes:
  "mean"             — weighted average pool over non-padding tokens (E5-family)
  "cls"              — first [CLS] token (BGE-M3 and most BGE models)
  "last"             — last non-padding token (decoder embedders, e.g. Qwen3-Embedding)
  "latent_attention" — model does its own pooling internally (e.g.
                        ai-sage/Giga-Embeddings-instruct's latent-attention head);
                        we call forward(..., return_embeddings=True) and use the
                        returned tensor as-is instead of touching hidden states.
                        Requires trust_remote_code=True.

L2-normalization is enforced here too, so the index can use IndexFlatIP and get
cosine similarity for free.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])

PoolingMode = Literal["mean", "cls", "last", "latent_attention"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class E5Embedder:
    def __init__(
        self,
        name: str = "intfloat/e5-large-v2",
        *,
        device: str = "cuda:0",
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        pooling: PoolingMode = "mean",
        dtype: str | None = None,
        add_eos: bool = False,
        padding_side: str | None = None,
        device_map: str | dict | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        self.name = name
        self.device = device
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling = pooling
        # Decoder-style embedders (Qwen3-Embedding) read the representation off
        # the last token, so each sequence must end with <|endoftext|> and be
        # left-padded; otherwise the "last" position would be a pad token.
        self.add_eos = add_eos
        # When set (e.g. "auto"), shard the model across every *visible* GPU via
        # accelerate so no single GPU holds the full weights. This is layer/
        # model sharding (the model is split by layer across devices), which is
        # the practical in-process way to fit a large embedder alongside a
        # GPU-hungry training job; it is not torch.distributed tensor parallel
        # (that needs torchrun). Restrict which GPUs are used with
        # CUDA_VISIBLE_DEVICES (see scripts/serve_tool.sh GPUS=...).
        self.device_map = device_map

        log.info(
            "loading embedder %s (device=%s device_map=%s pooling=%s dtype=%s add_eos=%s)",
            name,
            device,
            device_map,
            pooling,
            dtype,
            add_eos,
        )
        tok_kwargs = {}
        if padding_side is None and pooling == "last":
            padding_side = "left"
        if padding_side is not None:
            tok_kwargs["padding_side"] = padding_side
        if trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(name, **tok_kwargs)

        model_kwargs: dict = {"use_safetensors": True}
        if dtype is not None:
            if dtype not in _DTYPES:
                raise ValueError(
                    f"unknown dtype {dtype!r}; choose from {list(_DTYPES)}"
                )
            # transformers>=5 renamed the from_pretrained kwarg torch_dtype ->
            # dtype; older transformers (e.g. the pinned env some
            # trust_remote_code models require) only accepts torch_dtype.
            dtype_kwarg = "dtype" if _TRANSFORMERS_MAJOR >= 5 else "torch_dtype"
            model_kwargs[dtype_kwarg] = _DTYPES[dtype]
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            # accelerate dispatches modules across GPUs; do NOT call .to(device)
            # afterwards (it would undo the dispatch). Inputs are placed on the
            # input-embedding device; hidden states emerge on the last shard.
            self.model = AutoModel.from_pretrained(
                name, device_map=device_map, **model_kwargs
            ).eval()
            self._input_device = self.model.get_input_embeddings().weight.device
            log.info(
                "sharded embedder device_map=%s",
                getattr(self.model, "hf_device_map", None),
            )
        else:
            self.model = (
                AutoModel.from_pretrained(name, **model_kwargs).to(device).eval()
            )
            self._input_device = torch.device(device)

        self._eos_id: int | None = None
        if self.add_eos:
            self._eos_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if self._eos_id is None or self._eos_id < 0:
                self._eos_id = self.tokenizer.eos_token_id

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
        if self.add_eos and self._eos_id is not None:
            # Append <|endoftext|> to every sequence (reserve one slot for it),
            # then pad. Matches the official Qwen3-Embedding tokenization.
            batch = self.tokenizer(
                texts,
                max_length=self.max_length - 1,
                padding=False,
                truncation=True,
            )
            for ids, att in zip(batch["input_ids"], batch["attention_mask"]):
                ids.append(self._eos_id)
                att.append(1)
            batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(
                self._input_device
            )
        else:
            batch = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self._input_device)
        if self.pooling == "latent_attention":
            # The model's own forward pass does pooling (e.g. a learned
            # latent-attention head) and hands back embeddings directly —
            # there is no last_hidden_state to pool ourselves.
            emb = self.model(**batch, return_embeddings=True)
        else:
            outputs = self.model(**batch)
            # With device_map sharding, hidden states emerge on the last
            # shard's device; align the mask to it so pooling never crosses
            # devices.
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].to(hidden.device)
            if self.pooling == "cls":
                emb = hidden[:, 0, :]
            elif self.pooling == "last":
                emb = _last_token_pool(hidden, mask)
            else:
                emb = _average_pool(hidden, mask)
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def count_truncated(self, texts: Iterable[str]) -> int:
        """Count how many passages exceed max_length tokens (including prefix)."""
        n = 0
        for t in texts:
            ids = self.tokenizer.encode(
                self.passage_prefix + t, add_special_tokens=True
            )
            if len(ids) > self.max_length:
                n += 1
        return n


def _average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask[..., None].bool()
    last = last_hidden_states.masked_fill(~mask, 0.0)
    return last.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp(min=1)


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Take the hidden state of the last non-padding token.

    Handles both left- and right-padded batches (official Qwen3-Embedding impl).
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]
