"""Tests for the torch-free core of the expert-pruning pipeline.

prune/plan.py (expert selection) and prune/remap.py (tensor-name remapping +
config update) carry all the decision logic; the runtime stages
(collect_stats/apply/validate) only stream tensors through these functions.
These tests pin the invariants the checkpoint format relies on: uniform keep
count, contiguous renumbering, gate slicing, and both weight-name layouts
(bf16 actor vs vLLM-wrapped W8A8).
"""

from __future__ import annotations

import pytest

from prune.plan import select_experts, summarize_plan
from prune.remap import PlanIndex, plan_tensor, update_config_num_experts


def make_stats(
    masses: dict[int, list[float]], top_k: int = 2, counts: dict[int, list[int]] | None = None
) -> dict:
    num_experts = len(next(iter(masses.values())))
    layers = {}
    for lidx, mass in masses.items():
        count = (counts or {}).get(lidx) or [1 if m > 0 else 0 for m in mass]
        layers[str(lidx)] = {"count": count, "mass": mass, "tokens": 100}
    return {"num_experts": num_experts, "top_k": top_k, "layers": layers}


# ------------------------------------------------------------------ selection


def test_keep_selects_top_experts_by_mass():
    stats = make_stats({0: [0.1, 0.5, 0.0, 0.4], 1: [0.9, 0.0, 0.05, 0.05]})
    plan = select_experts(stats, keep=2)
    assert plan["keep_per_layer"] == 2
    assert plan["layers"]["0"] == [1, 3]
    assert plan["layers"]["1"] == [0, 2]  # tie 2 vs 3 broken by lower id


def test_kept_ids_are_sorted_and_uniform_count():
    stats = make_stats({0: [0.0, 0.1, 0.9, 0.2], 5: [0.4, 0.3, 0.2, 0.1]})
    plan = select_experts(stats, keep=3)
    for ids in plan["layers"].values():
        assert ids == sorted(ids)
        assert len(ids) == 3


def test_coverage_takes_max_over_layers():
    # layer 0 needs 1 expert for 90% coverage, layer 1 needs 3.
    stats = make_stats({0: [0.95, 0.05, 0.0, 0.0], 1: [0.4, 0.3, 0.25, 0.05]})
    plan = select_experts(stats, coverage=0.9)
    assert plan["keep_per_layer"] == 3


def test_coverage_full_keeps_everything():
    stats = make_stats({0: [0.4, 0.3, 0.2, 0.1]})
    plan = select_experts(stats, coverage=1.0)
    assert plan["keep_per_layer"] == 4


def test_min_keep_defaults_to_top_k():
    stats = make_stats({0: [1.0, 0.0, 0.0, 0.0]}, top_k=2)
    plan = select_experts(stats, coverage=0.5)  # coverage alone would keep 1
    assert plan["keep_per_layer"] == 2


def test_explicit_min_keep_floor():
    stats = make_stats({0: [1.0, 0.0, 0.0, 0.0]}, top_k=1)
    plan = select_experts(stats, keep=1, min_keep=3)
    assert plan["keep_per_layer"] == 3


def test_keep_greater_than_num_experts_raises():
    stats = make_stats({0: [0.5, 0.5, 0.0, 0.0]})
    with pytest.raises(ValueError):
        select_experts(stats, keep=5)


def test_exactly_one_of_keep_and_coverage():
    stats = make_stats({0: [0.5, 0.5, 0.0, 0.0]})
    with pytest.raises(ValueError):
        select_experts(stats)
    with pytest.raises(ValueError):
        select_experts(stats, keep=2, coverage=0.9)


def test_zero_mass_layer_keeps_everything_under_coverage():
    stats = make_stats({0: [0.0, 0.0, 0.0, 0.0], 1: [0.9, 0.1, 0.0, 0.0]})
    plan = select_experts(stats, coverage=0.9)
    assert plan["keep_per_layer"] == 4  # dead layer forces a full keep


def test_report_dropped_mass():
    stats = make_stats({0: [0.5, 0.3, 0.15, 0.05]})
    plan = select_experts(stats, keep=2)
    assert plan["report"]["0"]["dropped_mass_frac"] == pytest.approx(0.2)
    assert "->" in summarize_plan(plan)


def test_count_metric_ranking():
    stats = make_stats(
        {0: [0.4, 0.4, 0.1, 0.1]}, counts={0: [1, 100, 50, 2]}
    )
    plan = select_experts(stats, keep=2, metric="count")
    assert plan["layers"]["0"] == [1, 2]


# ------------------------------------------------------------------- remap


def make_plan(layers: dict[int, list[int]], num_experts: int = 4) -> PlanIndex:
    keep = len(next(iter(layers.values())))
    return PlanIndex(
        {
            "num_experts_orig": num_experts,
            "keep_per_layer": keep,
            "layers": {str(k): v for k, v in layers.items()},
        }
    )


@pytest.mark.parametrize(
    "prefix",
    ["model.layers", "model.language_model.layers"],  # bf16 actor / W8A8 wrapper
)
def test_expert_tensor_drop_and_rename(prefix):
    index = make_plan({3: [1, 3]})
    assert plan_tensor(f"{prefix}.3.mlp.experts.0.gate_proj.weight", index) == (
        "drop",
        None,
    )
    kind, new = plan_tensor(f"{prefix}.3.mlp.experts.1.gate_proj.weight", index)
    assert (kind, new) == ("rename", f"{prefix}.3.mlp.experts.0.gate_proj.weight")
    kind, new = plan_tensor(f"{prefix}.3.mlp.experts.3.down_proj.weight_scale", index)
    assert (kind, new) == ("rename", f"{prefix}.3.mlp.experts.1.down_proj.weight_scale")


def test_gate_tensor_sliced():
    index = make_plan({0: [0, 2]})
    kind, kept = plan_tensor("model.layers.0.mlp.gate.weight", index)
    assert kind == "slice0"
    assert kept == [0, 2]


def test_shared_expert_and_other_tensors_untouched():
    index = make_plan({0: [0, 1]})
    for name in (
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
        "model.layers.0.linear_attn.in_proj.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ):
        assert plan_tensor(name, index) == ("keep", name)


def test_layer_not_in_plan_untouched():
    index = make_plan({0: [0, 1]})
    name = "model.layers.7.mlp.experts.3.up_proj.weight"
    assert plan_tensor(name, index) == ("keep", name)
    gate = "model.layers.7.mlp.gate.weight"
    assert plan_tensor(gate, index) == ("keep", gate)


# ------------------------------------------------------------------- config


def test_update_flat_config():
    cfg = {"num_experts": 128, "num_experts_per_tok": 8}
    assert update_config_num_experts(cfg, 64) == ["num_experts"]
    assert cfg["num_experts"] == 64
    assert cfg["num_experts_per_tok"] == 8


def test_update_wrapped_config():
    cfg = {
        "model_type": "qwen3_5_moe",
        "text_config": {"num_experts": 128, "num_experts_per_tok": 8},
        "quantization_config": {"ignore": ["lm_head"]},
    }
    assert update_config_num_experts(cfg, 96) == ["text_config.num_experts"]
    assert cfg["text_config"]["num_experts"] == 96


def test_update_config_without_expert_key_raises():
    with pytest.raises(ValueError):
        update_config_num_experts({"hidden_size": 4096}, 64)
