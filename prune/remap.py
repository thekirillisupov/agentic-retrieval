"""Tensor-name remapping for expert pruning. Pure python, no torch.

Works at the safetensors-name level so the SAME code prunes both the bf16
actor (``model.layers.*``) and the vLLM-wrapped compressed-tensors W8A8
checkpoint (``model.language_model.layers.*``, where every expert Linear
carries extra tensors such as ``weight_scale``). Quantization suffixes are
renamed together with their module, so no scheme-specific handling is needed.

Touched tensors:

  *.layers.{L}.mlp.experts.{E}.<rest>   dropped, or renamed to the expert's
                                        new contiguous index.
  *.layers.{L}.mlp.gate.<rest>          router Linear — rows sliced down to
                                        the kept experts (dim 0 == num_experts
                                        for both weight [E, H] and bias [E]).

Everything else (shared_expert, shared_expert_gate, linear_attn, lm_head, ...)
passes through unchanged. ``mlp.gate.`` does not match ``mlp.shared_expert_gate.``.
"""

from __future__ import annotations

import re
from typing import Any

# Both the flat actor ("model.layers.N...") and the multimodal wrapper
# ("model.language_model.layers.N...") end with the same suffix.
EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts\.)(?P<expert>\d+)\.(?P<rest>.+)$"
)
GATE_RE = re.compile(r"^.*\.layers\.(?P<layer>\d+)\.mlp\.gate\.(?P<rest>.+)$")

# Actions returned by plan_tensor:
#   ("keep", name)        — copy through unchanged
#   ("rename", new_name)  — kept expert, index compacted
#   ("drop", None)        — pruned expert
#   ("slice0", kept_ids)  — router gate tensor: take rows kept_ids along dim 0
Action = tuple[str, Any]


class PlanIndex:
    """O(1) lookups over a prune plan (plan.py format)."""

    def __init__(self, plan: dict[str, Any]):
        self.num_experts_orig = int(plan["num_experts_orig"])
        self.keep_per_layer = int(plan["keep_per_layer"])
        self.kept: dict[int, list[int]] = {
            int(lidx): list(ids) for lidx, ids in plan["layers"].items()
        }
        self.old2new: dict[int, dict[int, int]] = {
            lidx: {old: new for new, old in enumerate(ids)}
            for lidx, ids in self.kept.items()
        }


def plan_tensor(name: str, index: PlanIndex) -> Action:
    """Decide what happens to one tensor under the prune plan."""
    m = EXPERT_RE.match(name)
    if m:
        layer = int(m.group("layer"))
        mapping = index.old2new.get(layer)
        if mapping is None:
            # Layer absent from the plan (e.g. mlp_only / dense layer): untouched.
            return ("keep", name)
        expert = int(m.group("expert"))
        new = mapping.get(expert)
        if new is None:
            return ("drop", None)
        return ("rename", f"{m.group('prefix')}{new}.{m.group('rest')}")

    m = GATE_RE.match(name)
    if m:
        layer = int(m.group("layer"))
        kept = index.kept.get(layer)
        if kept is None:
            return ("keep", name)
        return ("slice0", kept)

    return ("keep", name)


_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def update_config_num_experts(cfg: dict[str, Any], keep: int) -> list[str]:
    """Set the expert count in a checkpoint config dict (in place).

    Handles both the flat actor config and the vLLM-wrapped one where the real
    keys live under ``text_config``. Every location that carries an expert-count
    key is updated. Returns the dotted paths that were changed; raises if none
    were found (silently pruning a config we don't understand is worse).
    """
    changed: list[str] = []
    scopes: list[tuple[str, dict[str, Any]]] = [("", cfg)]
    if isinstance(cfg.get("text_config"), dict):
        scopes.append(("text_config.", cfg["text_config"]))
    for prefix, scope in scopes:
        for key in _NUM_EXPERTS_KEYS:
            if key in scope:
                scope[key] = keep
                changed.append(prefix + key)
    if not changed:
        raise ValueError(
            f"no expert-count key ({', '.join(_NUM_EXPERTS_KEYS)}) found in config "
            f"— refusing to write a pruned checkpoint with a stale config"
        )
    return changed
