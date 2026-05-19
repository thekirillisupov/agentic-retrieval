"""WandB diagnostic metrics for ``algorithm.filter_groups``.

The ``filter_groups`` block in our configs (see e.g.
``configs/gspo_qwen3_moe.yaml:52``) tells veRL to drop zero-variance
prompt groups before the policy update. In the upstream ``verl==0.7.1``
``RayPPOTrainer.fit`` this block is a no-op — only the DAPO recipe
consumes it. :mod:`grpo.filter_groups_trainer` ports the DAPO filter
loop into a ``RayPPOTrainer`` subclass so it actually applies, and
``scripts/main_ppo.py`` installs both that subclass and this
diagnostic wrapper.

When the filter is **on** (the default path under
``scripts/main_ppo.py``), ``compute_data_metrics`` runs on the
*post-filter* batch — i.e. the batch actually fed to the actor
update. So we expect:

  * ``filter_groups/n_filtered`` ≈ 0 every step.
  * ``filter_groups/effective_batch_size`` ≈
    ``filter_groups/nominal_batch_size``.

The richer "how much rollout compute did the filter absorb?" view is
emitted by :class:`grpo.filter_groups_trainer.FilterGroupsRayPPOTrainer`
itself, as ``train/num_gen_batches`` and
``filter_groups/total_filtered_groups``.

When the filter is **off**
(``algorithm.filter_groups.enable=false``), the trainer subclass's
``fit()`` short-circuits and behaves identically to upstream
``RayPPOTrainer.fit``. In that case ``compute_data_metrics`` runs on
the *pre-filter* batch, so ``n_filtered > 0`` is the "what-if" view —
the count of groups the filter *would* have dropped if enabled.

Metrics emitted (always, regardless of whether filtering is on):

* ``filter_groups/n_groups``              unique groups (uids) in the batch
* ``filter_groups/n_filtered``            groups with zero-variance metric
* ``filter_groups/n_filtered_all_zero``   filtered groups where every rollout scored 0 (mean == 0)
* ``filter_groups/n_filtered_all_one``    filtered groups where every rollout scored 1 (mean == 1)
* ``filter_groups/n_kept``                ``n_groups - n_filtered``
* ``filter_groups/n_not_filtered_zero``   total traces in kept groups whose mean reward is 0 (sum of rollout counts)
* ``filter_groups/filter_ratio``          ``n_filtered / n_groups``
* ``filter_groups/effective_batch_size``  ``n_kept * rollout.n``
* ``filter_groups/nominal_batch_size``    ``n_groups * rollout.n``

Expected batch layout when ``compute_data_metrics`` is called:

* ``batch.non_tensor_batch["uid"]``    one prompt-id per row,
                                       repeated ``rollout.n`` times
                                       per prompt.
* ``batch.non_tensor_batch[metric]``   one scalar per row, surfaced
                                       from ``reward_extra_infos_dict``
                                       (our ``grpo/reward.py`` returns
                                       ``score``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


def _per_row_metric(batch: Any, metric: str) -> np.ndarray | None:
    """Resolve the per-rollout metric values used by `filter_groups`.

    Mirrors what the DAPO recipe accepts: a key in ``non_tensor_batch``
    (the common case — our reward fn returns ``score``) or one of the
    aggregate fallbacks ``seq_reward`` / ``seq_final_reward``.
    """
    nt = batch.non_tensor_batch
    if metric in nt:
        arr = np.asarray(nt[metric])
        if arr.dtype.kind in ("U", "O"):
            try:
                arr = arr.astype(np.float32)
            except (TypeError, ValueError):
                return None
        return arr.astype(np.float32, copy=False)

    if metric == "seq_reward" and "token_level_rewards" in batch.batch:
        return batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
    if metric == "seq_final_reward" and "token_level_scores" in batch.batch:
        return batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
    return None


def compute_filter_groups_metrics(
    batch: Any,
    *,
    metric: str,
    rollout_n: int,
) -> dict[str, float]:
    """Compute the diagnostic metrics described in the module docstring.

    Returns an empty dict if the batch is missing ``uid`` or the metric
    column — i.e. we never crash the training step over telemetry.
    """
    uids = batch.non_tensor_batch.get("uid")
    if uids is None:
        log.debug("filter_groups_metrics: batch has no 'uid' column; skipping.")
        return {}

    vals = _per_row_metric(batch, metric)
    if vals is None:
        log.debug(
            "filter_groups_metrics: metric %r not found in batch.non_tensor_batch; "
            "skipping. Available keys: %s",
            metric,
            list(batch.non_tensor_batch.keys()),
        )
        return {}

    uid_arr = np.asarray(uids)
    if uid_arr.shape[0] != vals.shape[0]:
        log.warning(
            "filter_groups_metrics: uid length %d != metric length %d; skipping.",
            uid_arr.shape[0],
            vals.shape[0],
        )
        return {}

    unique_uids, inverse = np.unique(uid_arr, return_inverse=True)
    n_groups = int(unique_uids.size)
    if n_groups == 0:
        return {}

    # Per-group std on `vals`. A group is "zero-variance" iff every
    # rollout produced the exact same metric value — that's the
    # condition the DAPO recipe filters on (`std == 0`).
    sums = np.zeros(n_groups, dtype=np.float64)
    sqsums = np.zeros(n_groups, dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, inverse, vals)
    np.add.at(sqsums, inverse, vals.astype(np.float64) ** 2)
    np.add.at(counts, inverse, 1)
    means = sums / np.maximum(counts, 1)
    variances = np.maximum(sqsums / np.maximum(counts, 1) - means**2, 0.0)
    is_filtered = variances <= 0.0

    n_filtered = int(is_filtered.sum())
    n_kept = n_groups - n_filtered

    nominal_batch_size = int(uid_arr.shape[0])
    effective_batch_size = n_kept * int(rollout_n)

    # Groups whose mean reward is exactly zero (all rollouts produced 0).
    is_zero_mean = means == 0.0

    is_one_mean = means == 1.0

    # Filtered groups split by whether every rollout scored zero or one.
    n_filtered_all_zero = int((is_filtered & is_zero_mean).sum())
    n_filtered_all_one = int((is_filtered & is_one_mean).sum())

    # Number of traces (individual rollout rows) that belong to a kept group
    # whose mean reward is zero — summed over all such groups.
    not_filtered_zero_mask = ~is_filtered & is_zero_mean
    n_not_filtered_zero = int(counts[not_filtered_zero_mask].sum())

    return {
        "filter_groups/n_groups": float(n_groups),
        "filter_groups/n_filtered": float(n_filtered),
        "filter_groups/n_filtered_all_zero": float(n_filtered_all_zero),
        "filter_groups/n_filtered_all_one": float(n_filtered_all_one),
        "filter_groups/n_kept": float(n_kept),
        "filter_groups/n_not_filtered_zero": float(n_not_filtered_zero),
        "filter_groups/filter_ratio": float(n_filtered) / float(n_groups),
        "filter_groups/effective_batch_size": float(effective_batch_size),
        "filter_groups/nominal_batch_size": float(nominal_batch_size),
    }


def wrap_compute_data_metrics(
    orig_fn: Callable[..., dict],
    *,
    algorithm_cfg: Any,
    rollout_n: int,
) -> Callable[..., dict]:
    """Return a drop-in replacement for ``compute_data_metrics`` that also
    emits :func:`compute_filter_groups_metrics`.

    Reads ``algorithm_cfg.filter_groups.metric`` (default ``score``);
    if the ``filter_groups`` sub-config is missing the wrapper is a no-op.
    """
    fg = getattr(algorithm_cfg, "filter_groups", None)
    if fg is None:
        try:
            fg = algorithm_cfg.get("filter_groups", None)  # OmegaConf path
        except Exception:
            fg = None
    if fg is None:
        log.info(
            "filter_groups_metrics: algorithm.filter_groups not set; "
            "wrapper is a no-op (no diagnostic metrics emitted)."
        )
        return orig_fn

    metric = getattr(fg, "metric", None)
    if metric is None:
        try:
            metric = fg.get("metric", None)
        except Exception:
            metric = None
    metric = metric or "score"

    # ``filter_groups.enable`` decides whether the trainer subclass
    # actually drops zero-variance groups; the diagnostic wrapper is
    # installed either way. When enable=True, the batch reaching
    # ``compute_data_metrics`` has already been filtered, so we expect
    # ``n_filtered == 0`` on every step.
    fg_enable = bool(getattr(fg, "enable", False))
    try:
        fg_enable = bool(fg_enable or fg.get("enable", False))
    except Exception:
        pass

    log.info(
        "filter_groups_metrics: logging metric=%r, rollout_n=%d "
        "(filter_groups.enable=%s — when True the trainer subclass in "
        "grpo.filter_groups_trainer has already dropped zero-variance "
        "groups, so n_filtered should be 0 here; the trainer also "
        "emits train/num_gen_batches and "
        "filter_groups/total_filtered_groups to capture the absorbed "
        "rollout cost).",
        metric,
        rollout_n,
        fg_enable,
    )

    def wrapper(*, batch, use_critic=True):
        metrics = orig_fn(batch=batch, use_critic=use_critic)
        try:
            metrics.update(
                compute_filter_groups_metrics(
                    batch,
                    metric=metric,
                    rollout_n=rollout_n,
                )
            )
        except Exception as e:
            log.warning("filter_groups_metrics: failed to compute (%s)", e)
        return metrics

    return wrapper
