"""Shared runtime helpers for the expert-pruning pipeline (needs torch/transformers).

Kept out of plan.py/remap.py so the pure logic stays importable without a GPU env.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from quantize.calibration import (
    build_calibration_from_messages,
    load_messages_from_trajectory_parquet,
)

_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "moe_top_k", "num_selected_experts")


def _text_config(config: Any) -> Any:
    """The config scope that actually carries the MoE keys.

    The quantized checkpoint is wrapped for vLLM (quantize/vllm_config.py):
    a ``qwen3_5_moe`` parent with the real model keys under ``text_config``.
    The bf16 actor config is flat.
    """
    inner = getattr(config, "text_config", None)
    if inner is not None and any(hasattr(inner, k) for k in _NUM_EXPERTS_KEYS):
        return inner
    return config


def moe_params(config: Any) -> tuple[int, int, bool]:
    """(num_experts, top_k, norm_topk_prob) from a loaded HF config."""
    scope = _text_config(config)
    num_experts = next(
        (getattr(scope, k) for k in _NUM_EXPERTS_KEYS if getattr(scope, k, None)),
        None,
    )
    top_k = next(
        (getattr(scope, k) for k in _TOP_K_KEYS if getattr(scope, k, None)), None
    )
    if not num_experts or not top_k:
        raise ValueError(
            f"could not find MoE keys in config (looked for {_NUM_EXPERTS_KEYS} "
            f"and {_TOP_K_KEYS})"
        )
    return int(num_experts), int(top_k), bool(getattr(scope, "norm_topk_prob", False))


def load_model(model_dir: str, *, dtype: str = "bfloat16", device_map: str = "auto"):
    """Load either the flat CausalLM actor or the vLLM-wrapped multimodal shell.

    compressed-tensors checkpoints (the W8A8 candidate) are handled by
    transformers' quantizer integration — requires ``compressed-tensors`` in the
    env (pulled in by llmcompressor, same env as quantize/).
    """
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    kwargs = dict(torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except (ValueError, KeyError):
        # Wrapped checkpoint: resolve the architecture class directly
        # (e.g. Qwen3_5MoeForConditionalGeneration is not in the CausalLM mapping).
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        arch = (getattr(config, "architectures", None) or [None])[0]
        if arch is None or not hasattr(transformers, arch):
            raise
        model = getattr(transformers, arch).from_pretrained(model_dir, **kwargs)
    model.eval()
    return model


def free_model(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_split(
    parquet: str | Path,
    tokenizer: Any,
    *,
    split: str,
    num_stats: int,
    num_val: int,
    max_seq_len: int,
    seed: int = 0,
    prompt_field: str = "prompt",
    default_prompt_version: str = "v2_search_only",
):
    """Deterministic stats/val split of the train parquet, tokenized.

    The train parquet doubles as the pipeline's validation domain, so the two
    consumers must not overlap: after one shuffle(seed) the stats pass reads
    rows [:num_stats] and validation reads rows [-num_val:]. Overlap is an
    error — shrink the sample sizes rather than validating on rows the
    selection already saw.

    Returns a datasets.Dataset with input_ids/attention_mask (same rendering
    path as quantization calibration: chat template + tool schemas +
    add_generation_prompt, see quantize/calibration.py).
    """
    import random

    if split not in ("stats", "val"):
        raise ValueError(f"split must be 'stats' or 'val', got {split!r}")

    msgs = load_messages_from_trajectory_parquet(
        parquet,
        prompt_field=prompt_field,
        default_prompt_version=default_prompt_version,
    )
    if num_stats + num_val > len(msgs):
        raise ValueError(
            f"stats ({num_stats}) + val ({num_val}) samples exceed parquet rows "
            f"({len(msgs)}) — the val slice would overlap the stats slice"
        )
    random.Random(seed).shuffle(msgs)
    pool = msgs[:num_stats] if split == "stats" else msgs[len(msgs) - num_val :]

    return build_calibration_from_messages(
        pool,
        tokenizer,
        num_samples=len(pool),
        max_seq_len=max_seq_len,
        # Parquet prompts are single-turn [system, user]: match inference.
        add_generation_prompt=True,
        seed=seed,
    )
