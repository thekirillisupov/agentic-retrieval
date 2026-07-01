"""Fix compressed-tensors checkpoints for vLLM serving.

llmcompressor saves a flat ``qwen3_5_moe_text`` config, but vLLM 0.18 resolves
``Qwen3_5MoeForCausalLM`` to the multimodal ``Qwen3_5MoeForConditionalGeneration``
implementation and expects a nested ``qwen3_5_moe`` parent config with
``text_config`` + ``vision_config``. Weight tensors already use the
``model.language_model.*`` prefix; only config.json needs restructuring.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoConfig

# Parent-level keys copied from the source HF checkpoint (vision shell).
_PARENT_KEYS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "vision_config",
    "tie_word_embeddings",
    "transformers_version",
)

# Flat keys that belong only on the parent, not inside text_config.
_TOP_LEVEL_ONLY = frozenset(
    {
        "architectures",
        "model_type",
        "quantization_config",
        *_PARENT_KEYS,
    }
)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def _rewrite_ignore_paths(ignore: list[str]) -> list[str]:
    out: list[str] = []
    for entry in ignore:
        if entry.startswith("model.layers."):
            out.append(entry.replace("model.layers.", "model.language_model.layers.", 1))
        else:
            out.append(entry)
    return out


def wrap_saved_config_for_vllm(out_dir: str | Path, source_model: str) -> None:
    """Restructure a flat text-only config.json for vLLM multimodal loading."""
    out = Path(out_dir)
    cfg_path = out / "config.json"
    saved = json.loads(cfg_path.read_text())

    if saved.get("model_type") == "qwen3_5_moe" and "text_config" in saved:
        # Idempotent: fix ignore prefixes if an older wrap missed them.
        qcfg = saved.get("quantization_config")
        if isinstance(qcfg, dict) and "ignore" in qcfg:
            fixed = _rewrite_ignore_paths(qcfg["ignore"])
            if fixed != qcfg["ignore"]:
                qcfg["ignore"] = fixed
                cfg_path.write_text(json.dumps(saved, indent=2) + "\n")
        return

    source = _config_to_dict(
        AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    )
    if source.get("model_type") != "qwen3_5_moe":
        raise ValueError(
            f"Expected source model_type=qwen3_5_moe, got {source.get('model_type')!r} "
            f"from {source_model}"
        )

    text_config = source.get("text_config") or {}
    text_config = deepcopy(text_config)
    for key, value in saved.items():
        if key not in _TOP_LEVEL_ONLY:
            text_config[key] = value
    text_config["model_type"] = "qwen3_5_moe_text"

    parent: dict[str, Any] = {k: deepcopy(source[k]) for k in _PARENT_KEYS if k in source}
    parent["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    parent["model_type"] = "qwen3_5_moe"
    parent["text_config"] = text_config

    if "quantization_config" in saved:
        qcfg = deepcopy(saved["quantization_config"])
        if "ignore" in qcfg:
            qcfg["ignore"] = _rewrite_ignore_paths(qcfg["ignore"])
        parent["quantization_config"] = qcfg

    cfg_path.write_text(json.dumps(parent, indent=2) + "\n")
