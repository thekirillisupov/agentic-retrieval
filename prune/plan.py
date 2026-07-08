"""Expert selection: router stats -> prune plan. Pure python, no torch.

The plan keeps a UNIFORM number of experts per layer. vLLM and the HF config
carry a single ``num_experts`` for the whole model, so per-layer expert counts
are not representable in the checkpoint — we pick WHICH experts each layer
keeps, but every layer keeps the same COUNT.

Stats format (produced by prune/collect_stats.py):

    {
      "model": "...", "train_parquet": "...",
      "num_experts": E, "top_k": K, "norm_topk_prob": bool,
      "num_samples": N, "max_seq_len": L,
      "layers": {
        "<layer_idx>": {"count": [E ints], "mass": [E floats], "tokens": T},
        ...
      }
    }

``count`` = how many times each expert was in the router's top-k;
``mass``  = accumulated (top-k, optionally renormalised) router probability.

Plan format (consumed by prune/apply.py):

    {
      "num_experts_orig": E, "keep_per_layer": Knew, "top_k": K,
      "metric": "mass",
      "layers": {"<layer_idx>": [sorted original expert ids kept]},
      "report": {"<layer_idx>": {"dropped_mass_frac": f, "dead_kept": n}, ...}
    }
"""

from __future__ import annotations

from typing import Any

VALID_METRICS = ("mass", "count")


def _layer_ranking(layer_stats: dict[str, Any], metric: str) -> list[int]:
    """Expert ids sorted by importance, most important first. Deterministic:
    ties broken by the other metric, then by lower expert id."""
    primary = layer_stats[metric]
    secondary = layer_stats["count" if metric == "mass" else "mass"]
    ids = list(range(len(primary)))
    ids.sort(key=lambda e: (-primary[e], -secondary[e], e))
    return ids


def _min_keep_for_coverage(
    layer_stats: dict[str, Any], ranking: list[int], coverage: float
) -> int:
    """Smallest k such that the top-k experts carry >= coverage of layer mass."""
    total = float(sum(layer_stats["mass"]))
    if total <= 0.0:
        # Router never fired on this layer in the stats pass — keep everything
        # rather than pruning blind.
        return len(ranking)
    acc = 0.0
    for k, e in enumerate(ranking, start=1):
        acc += float(layer_stats["mass"][e])
        if acc / total >= coverage:
            return k
    return len(ranking)


def select_experts(
    stats: dict[str, Any],
    *,
    keep: int | None = None,
    coverage: float | None = None,
    metric: str = "mass",
    min_keep: int | None = None,
) -> dict[str, Any]:
    """Build a prune plan from router stats.

    Exactly one of ``keep`` (uniform expert count to retain) or ``coverage``
    (probability-mass fraction that must survive in EVERY layer; the uniform
    keep count is the max over layers of the per-layer minimum) must be given.

    ``min_keep`` defaults to ``top_k`` — the router cannot select k experts out
    of fewer than k.
    """
    if (keep is None) == (coverage is None):
        raise ValueError("pass exactly one of keep= or coverage=")
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")

    num_experts = int(stats["num_experts"])
    top_k = int(stats["top_k"])
    floor = max(int(min_keep) if min_keep is not None else top_k, top_k)

    layers = stats["layers"]
    if not layers:
        raise ValueError("stats has no layers")
    for lidx, ls in layers.items():
        for key in ("count", "mass"):
            if len(ls[key]) != num_experts:
                raise ValueError(
                    f"layer {lidx}: len({key})={len(ls[key])} != num_experts={num_experts}"
                )

    rankings = {lidx: _layer_ranking(ls, metric) for lidx, ls in layers.items()}

    if coverage is not None:
        if not (0.0 < coverage <= 1.0):
            raise ValueError(f"coverage must be in (0, 1], got {coverage}")
        keep = max(
            _min_keep_for_coverage(layers[lidx], rankings[lidx], coverage)
            for lidx in layers
        )

    assert keep is not None
    keep = max(int(keep), floor)
    if keep > num_experts:
        raise ValueError(f"keep={keep} > num_experts={num_experts}")

    plan_layers: dict[str, list[int]] = {}
    report: dict[str, dict[str, Any]] = {}
    for lidx, ls in layers.items():
        kept = sorted(rankings[lidx][:keep])
        total_mass = float(sum(ls["mass"]))
        kept_mass = float(sum(ls["mass"][e] for e in kept))
        dropped_frac = 0.0 if total_mass <= 0 else 1.0 - kept_mass / total_mass
        dead_kept = sum(1 for e in kept if ls["count"][e] == 0)
        plan_layers[str(lidx)] = kept
        report[str(lidx)] = {
            "dropped_mass_frac": dropped_frac,
            "dead_kept": dead_kept,
        }

    return {
        "num_experts_orig": num_experts,
        "keep_per_layer": keep,
        "top_k": top_k,
        "metric": metric,
        "coverage": coverage,
        "layers": plan_layers,
        "report": report,
    }


def summarize_plan(plan: dict[str, Any]) -> str:
    """Human-readable one-screen summary for logs."""
    rep = plan["report"]
    worst = max(rep.items(), key=lambda kv: kv[1]["dropped_mass_frac"])
    dead = sum(v["dead_kept"] for v in rep.values())
    lines = [
        f"experts: {plan['num_experts_orig']} -> {plan['keep_per_layer']} per layer "
        f"({plan['keep_per_layer'] / plan['num_experts_orig']:.1%} kept, "
        f"metric={plan['metric']}, top_k={plan['top_k']})",
        f"layers: {len(plan['layers'])}",
        f"worst layer by dropped router mass: {worst[0]} "
        f"({worst[1]['dropped_mass_frac']:.3%} dropped)",
    ]
    if dead:
        lines.append(
            f"WARNING: {dead} kept expert slots never fired during the stats pass "
            f"— raise --num-samples or lower --keep"
        )
    return "\n".join(lines)
