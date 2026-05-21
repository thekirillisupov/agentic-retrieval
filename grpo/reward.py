"""Composite NDCG@k + F_beta reward for GRPO/GSPO.

Reward
------

    r = alpha * NDCG@k + (1 - alpha) * F_beta(output)
        - length_penalty
        + format_penalty
        + answer_bonus * [stopped_reason == "answer"]

* ``alpha`` mixes a graded ranking signal (NDCG) with a recall-oriented
  set-quality signal (F_beta over the answered list). Defaults follow the
  GSPO-MoE recipe the user requested: ``alpha=0.7``, ``beta=2.0``, ``k=10``.
* ``length_penalty`` is one-sided and OFF by default (``length_alpha=0.0``).
  GRPO/GSPO already aggregate the loss per token (``actor.loss_agg_mode``);
  the explicit penalty is kept here as an *option* to discourage runaway
  tool-call loops if you observe length blowups during training.
* ``format_penalty`` (``parse_error_penalty``) is applied only when the
  rollout ends in ``parse_error``; both ranking metrics are zeroed in that
  case so the model only ever gets credit for ranked output it produced.
* ``answer_bonus`` (epsilon) is a small additive bonus applied iff the
  rollout terminates by emitting an ``answer`` (rather than running out of
  turns / tool calls / parse errors). It nudges the policy toward decisive
  termination without dominating the ranking signal; recommended range
  0.02-0.05.

Per-rollout metrics returned (consumed by veRL aggregation / the trajectory
log via ``extra_fields``):

    score             - the scalar driving the policy update
    ndcg              - NDCG@k over the answered list vs. gold
    dcg               - DCG@k over the answered list vs. gold (raw, un-normalised)
    f_beta            - F_beta over the answered list vs. gold (the term used in r)
    f_beta_at_output  - alias for f_beta, exposed as a separately-named metric
    precision         - precision over the answered list
    recall            - recall over the answered list
    length_penalty    - the value subtracted from r (>=0)
    format_penalty    - the value added to r (typically <=0)
    response_len      - tokens the policy produced this rollout
    stopped_reason    - terminal state of the rollout
    num_tool_calls    - number of tool calls executed
    answered          - 1.0 iff the rollout ended in ``answer``

Filtering zero-variance groups stays veRL's responsibility (see
``algorithm.filter_groups`` in the trainer config); this function just has to
return a stable scalar per rollout.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from eval_.metrics import (
    dcg_at_k,
    f_beta,
    ndcg_at_k,
    simple_precision,
    simple_recall,
)

log = logging.getLogger(__name__)


# Default knobs; can be overridden via reward_kwargs in the trainer config.
DEFAULTS = {
    "ndcg_k": 10,
    "alpha": 0.7,                   # NDCG weight; (1 - alpha) goes to F_beta
    "f_beta": 2.0,                  # beta for F_beta (beta>1 => recall-heavy)
    "length_alpha": 0.0,            # length penalty OFF by default; opt-in
    "length_target_tokens": 4096,
    "parse_error_penalty": 0.0,     # set to e.g. -0.1 to punish malformed outputs
    "answer_bonus": 0.0,            # epsilon added when stopped_reason == "answer"
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

    Returns a dict with ``score`` (the scalar driving GRPO/GSPO) plus auxiliary
    metrics for logging / filter_groups.
    """
    cfg = {**DEFAULTS, **{k: v for k, v in (kwargs or {}).items() if k in DEFAULTS}}
    k = int(cfg["ndcg_k"])
    alpha = float(cfg["alpha"])
    beta = float(cfg["f_beta"])
    length_alpha = float(cfg["length_alpha"])
    target = max(1, int(cfg["length_target_tokens"]))
    parse_pen = float(cfg["parse_error_penalty"])
    answer_bonus = float(cfg["answer_bonus"])

    gold_list = _gold_from_ground_truth(ground_truth, extra_info)
    gold = set(gold_list)
    ranked, info = _ranked_from_extra(extra_info)
    stopped = str(info.get("stopped_reason", "unknown"))
    response_len = int(info.get("response_len", info.get("active_response_len", 0)))

    if stopped == "parse_error" or not gold:
        ndcg = 0.0
        dcg = 0.0
        precision = 0.0
        recall = 0.0
        fbeta_val = 0.0
        format_penalty = parse_pen if stopped == "parse_error" else 0.0
    else:
        ndcg = ndcg_at_k(ranked, gold, k=k)
        dcg = dcg_at_k(ranked, gold, k=k)
        precision = simple_precision(ranked, gold)
        recall = simple_recall(ranked, gold)
        fbeta_val = f_beta(precision, recall, beta=beta)
        format_penalty = 0.0

    # One-sided length penalty in [0, 1] applied to the reward; longer rollouts
    # get docked. Capped at 1.0 to avoid driving rewards arbitrarily negative.
    # length_alpha=0.0 disables the penalty entirely (default).
    overrun = max(0, response_len - target)
    len_pen = min(1.0, length_alpha * (overrun / target)) if length_alpha > 0 else 0.0

    answered = stopped == "answer"
    answer_bonus_val = answer_bonus if answered else 0.0

    composite = alpha * ndcg + (1.0 - alpha) * fbeta_val
    score = composite - len_pen + format_penalty + answer_bonus_val

    return {
        "score": float(score),
        "ndcg": float(ndcg),
        "dcg": float(dcg),
        "f_beta": float(fbeta_val),
        "f_beta_at_output": float(fbeta_val),
        "precision": float(precision),
        "recall": float(recall),
        "length_penalty": float(len_pen),
        "format_penalty": float(format_penalty),
        "answer_bonus": float(answer_bonus_val),
        "response_len": float(response_len),
        "stopped_reason": stopped,
        "num_tool_calls": float(info.get("num_tool_calls", 0)),
        "answered": float(answered),
    }
