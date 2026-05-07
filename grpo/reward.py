"""NDCG@k reward for GRPO.

Three things to be careful about, all driven by the user's spec:

1. **NDCG over gold ids.** We reuse `eval_.metrics.ndcg_at_k` so the reward
   number printed during training is exactly comparable to the eval-time
   number printed by `eval_.run_eval`. No accidental drift between the two.

2. **Length normalisation.** GRPO already does token-level loss aggregation
   (`actor.loss_agg_mode=token-mean`), which removes most length bias inside
   the optimiser. *In addition*, we apply a small explicit penalty per
   response token at the reward level:

       r = ndcg - alpha * (response_len - target_len).clip(min=0) / target_len

   This is a one-sided penalty: short rollouts get full credit, long ones get
   docked. ``alpha=0`` recovers pure NDCG. The default (``alpha=0.05``,
   ``target_len=2048``) is gentle enough to leave the gradient direction
   dominated by NDCG while discouraging runaway tool-call loops.

3. **Format / shape penalties.** A run that ends in `parse_error` or
   `max_turns` with no answer gets reward = 0 even if the seen set happens to
   contain a gold id by accident — we only credit the model for ranked output
   it actually produced. Stop reasons are surfaced as auxiliary metrics so
   filter_groups can see them when computing zero-variance flags.

Filtering zero-variance groups is veRL's responsibility (see
``algorithm.filter_groups`` in the trainer config). All this function has to
do is return a stable scalar per rollout; veRL groups by ``uid`` and drops
groups whose reward variance is zero before the policy update.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from eval_.metrics import ndcg_at_k

log = logging.getLogger(__name__)


# Default knobs; can be overridden via reward_kwargs in the trainer config.
DEFAULTS = {
    "ndcg_k": 10,
    "length_alpha": 0.05,
    "length_target_tokens": 2048,
    "parse_error_penalty": 0.0,  # set to e.g. -0.1 to punish malformed outputs
}


def _gold_from_ground_truth(ground_truth: Any, extra_info: dict[str, Any] | None) -> list[str]:
    """`reward_model.ground_truth` is JSON-encoded in the parquet (str list)."""
    if extra_info and isinstance(extra_info.get("gold_doc_ids"), list):
        return list(extra_info["gold_doc_ids"])
    if isinstance(ground_truth, list):
        return list(ground_truth)
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return []


def _ranked_from_extra(extra_info: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
    """The AgentLoop puts ``ranked_doc_ids`` etc. under extra_fields; the trainer
    folds extra_fields into ``extra_info`` before calling this fn."""
    if not extra_info:
        return [], {}
    ranked = extra_info.get("ranked_doc_ids") or []
    return list(ranked), extra_info


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Reward callback signature expected by veRL's ``custom_reward_function``.

    Returns a dict with ``score`` (the scalar driving GRPO) plus auxiliary
    metrics for logging / filter_groups.
    """
    cfg = {**DEFAULTS, **{k: v for k, v in (kwargs or {}).items() if k in DEFAULTS}}
    k = int(cfg["ndcg_k"])
    alpha = float(cfg["length_alpha"])
    target = max(1, int(cfg["length_target_tokens"]))
    parse_pen = float(cfg["parse_error_penalty"])

    gold = _gold_from_ground_truth(ground_truth, extra_info)
    ranked, info = _ranked_from_extra(extra_info)
    stopped = str(info.get("stopped_reason", "unknown"))
    response_len = int(info.get("response_len", info.get("active_response_len", 0)))

    if stopped == "parse_error":
        ndcg = 0.0
        format_penalty = parse_pen
    else:
        ndcg = ndcg_at_k(ranked, set(gold), k=k) if gold else 0.0
        format_penalty = 0.0

    # One-sided length penalty in [0, 1+] applied to the reward, so longer
    # rollouts get docked. We cap at 1.0 of penalty to avoid driving rewards
    # arbitrarily negative.
    overrun = max(0, response_len - target)
    len_pen = min(1.0, alpha * (overrun / target)) if alpha > 0 else 0.0

    score = ndcg - len_pen + format_penalty

    return {
        "score": float(score),
        "ndcg": float(ndcg),
        "length_penalty": float(len_pen),
        "format_penalty": float(format_penalty),
        "response_len": float(response_len),
        "stopped_reason": stopped,  # string passes through veRL metric agg as-is
        "num_tool_calls": float(info.get("num_tool_calls", 0)),
        "answered": float(stopped == "answer"),
    }
