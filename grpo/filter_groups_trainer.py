"""DAPO-style ``filter_groups`` for verl 0.7.1's ``RayPPOTrainer``.

verl 0.7.1 declares ``algorithm.filter_groups`` in ``FilterGroupsConfig``
but its ``RayPPOTrainer.fit`` does not consume it — only the DAPO recipe
(``recipe/dapo/dapo_ray_trainer.py``) does. This subclass ports the DAPO
filter loop into the v0.7.1 fit() body so we can actually drop
zero-variance prompt groups *before* the policy update without forking
verl.

Algorithm (matches ``recipe/dapo/dapo_ray_trainer.py`` upstream):

1. Pull ``data.train_batch_size`` prompts from the dataloader, repeat
   each ``rollout.n`` times, generate, compute reward.
2. For each unique ``uid`` in the batch, compute the per-rollout std of
   ``algorithm.filter_groups.metric`` (default ``score``).
3. Keep groups iff ``std > 0`` (or ``len(metric_vals) == 1``).
4. Accumulate kept rows into a buffer; concat across gen batches.
5. If ``num_prompt_in_batch < data.train_batch_size``, pull *another*
   dataloader batch (fresh prompts, not retries) and repeat. The number
   of dataloader batches consumed per training step is capped by
   ``algorithm.filter_groups.max_num_gen_batches``. ``0`` = unlimited.
6. Once we have enough kept groups, trim the buffer to exactly
   ``train_batch_size * rollout.n`` rows and run old_log_prob, ref,
   values, KL/advantage, and the critic/actor update on the trimmed
   batch.
7. If the budget is exhausted before we have enough, raise
   ``ValueError`` (same behaviour as the DAPO recipe).

Implementation notes:

* The body of :meth:`fit` is a near-line-for-line copy of v0.7.1's
  ``RayPPOTrainer.fit`` so that all v0.7.1-specific machinery
  (``checkpoint_manager.update_weights``, ``async_rollout_manager``,
  ``rollout_correction`` bypass/decoupled modes, the new
  ``global_profiler``, GDPO per-component metrics, ``RolloutSkip``,
  ``AbstractCurriculumSampler``, ``on_batch_end``, ``balance_batch``,
  multimodal ``images_seqlens``, ``compute_variance_proxy_metrics``,
  memory-snapshot dumps) is preserved unchanged. The only structural
  change is the filter/accumulate/continue block injected immediately
  after ``extract_reward`` and before the operating-mode-selection /
  ``old_log_prob`` computation.

* ``token_level_scores`` and the ``reward_extra_infos_dict`` keys are
  pushed into the batch *before* the filter (so ``DataProto.concat``
  carries them across gen batches). The corresponding assignments in
  the original ``adv`` marked_timer block are skipped (they'd otherwise
  re-overwrite the accumulated batch with only the *last* gen batch's
  reward tensor).

* Extra wandb keys emitted by this trainer (in addition to the
  ``filter_groups/*`` diagnostics from
  :mod:`grpo.filter_groups_metrics`):

      * ``train/num_gen_batches``
            How many fresh dataloader batches we consumed to assemble
            this training step. ``1`` means no refill was needed.
      * ``filter_groups/total_filtered_groups``
            Total prompt groups dropped across all gen batches this
            step (i.e. how much rollout compute was "wasted" on
            zero-variance groups).
      * ``filter_groups/total_generated_groups``
            Total unique prompt groups generated across all gen batches
            this step (=
            ``num_gen_batches * data.train_batch_size``).
      * ``filter_groups/n_filtered_all_zero``
            Among the dropped (zero-variance) groups across all gen
            batches this step, how many had every rollout score exactly
            0 (model fully failed). Computed pre-filter — overrides the
            post-filter 0 emitted by
            :mod:`grpo.filter_groups_metrics`.
      * ``filter_groups/n_filtered_all_one``
            Same as above but for groups where every rollout scored
            exactly 1 (model fully saturated).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.utils import Role

from grpo.tool_penalty import compute_anchored_tool_penalty

log = logging.getLogger(__name__)


def _compute_kept_uids(
    uids: np.ndarray, metric_vals: np.ndarray
) -> tuple[list, int, int, int]:
    """Return ``(kept_uid_list, n_filtered, n_all_zero, n_all_one)`` for the
    DAPO filter rule.

    A group is kept iff its per-rollout metric values have ``std > 0``,
    matching the DAPO recipe (``recipe/dapo/dapo_ray_trainer.py``).
    Singleton groups (``len(vals) == 1``) are always kept — DAPO does
    the same.

    ``n_all_zero`` / ``n_all_one`` count, among the dropped (zero-variance)
    groups, how many had every rollout score exactly 0 or 1 respectively.
    Computed here — i.e. *before* the kept rows are extracted from
    ``new_batch`` — so downstream wandb keys reflect the pre-filter view.
    """
    prompt_uid2metric_vals: dict = defaultdict(list)
    for uid, metric_val in zip(uids, metric_vals, strict=True):
        prompt_uid2metric_vals[uid].append(metric_val)
    kept: list = []
    n_all_zero = 0
    n_all_one = 0
    for uid, vals in prompt_uid2metric_vals.items():
        if np.std(vals) > 0 or len(vals) == 1:
            kept.append(uid)
            continue
        v = float(vals[0])
        if v == 0.0:
            n_all_zero += 1
        elif v == 1.0:
            n_all_one += 1
    n_filtered = len(prompt_uid2metric_vals) - len(kept)
    return kept, n_filtered, n_all_zero, n_all_one


class FilterGroupsRayPPOTrainer(RayPPOTrainer):
    """``RayPPOTrainer`` subclass that implements ``filter_groups``.

    When ``algorithm.filter_groups.enable=False`` (or the block is
    absent) the fit loop behaves identically to upstream v0.7.1
    ``RayPPOTrainer.fit``.
    """

    def _apply_count_penalty(
        self,
        batch: DataProto,
        *,
        count_key: str,
        penalty_key: str,
        label: str,
        alpha_win: float,
        alpha_lose: float,
        beta_lose: float,
        winner_threshold: float,
        std_threshold: float,
    ) -> None:
        """Subtract a group-relative anchored *count* penalty in place.

        Generic over the per-rollout count being shaped (``count_key`` in
        ``non_tensor_batch``): ``num_tool_calls`` for the tool-call penalty,
        ``num_answer_docs`` for the answer-document penalty, etc.

        Operates on a single gen batch *before* ``filter_groups`` so the
        penalized reward drives both group filtering and the GRPO/GSPO
        advantage. The original (pre-penalty) reward is stashed in
        ``non_tensor_batch['base_score']`` so winners/zero-reward rollouts can
        still be identified for monitoring; this also lets multiple penalties
        stack without one demoting winners (winner status + the per-group std
        gate are always evaluated against the *base* reward, never the already
        partially-penalized ``score``).

        Requires ``uid``, ``score`` and ``count_key`` in ``non_tensor_batch``
        (``score`` and the counts are emitted by
        ``grpo/reward.py::compute_score``). If anything is missing we skip
        silently rather than crash the step.
        """
        nt = batch.non_tensor_batch
        if not ("uid" in nt and "score" in nt and count_key in nt):
            print(
                f"[{label}] missing one of uid/score/{count_key} in "
                f"non_tensor_batch; skipping {label} for this batch.",
                flush=True,
            )
            return

        uids = nt["uid"]
        # Winners + std gate are defined by the *base* (pre-penalty) reward so
        # stacking multiple penalties never demotes a winner below threshold.
        base_scores = np.asarray(
            nt["base_score"] if "base_score" in nt else nt["score"],
            dtype=np.float64,
        )
        counts = np.asarray(nt[count_key], dtype=np.float64)

        penalties, _is_winner = compute_anchored_tool_penalty(
            uids,
            base_scores,
            counts,
            alpha_win=alpha_win,
            alpha_lose=alpha_lose,
            beta_lose=beta_lose,
            winner_threshold=winner_threshold,
            std_threshold=std_threshold,
        )

        # Stash the pre-penalty reward once (first penalty to run owns it), then
        # subtract this penalty from the running ``score`` (drives filter_groups).
        if "base_score" not in nt:
            nt["base_score"] = base_scores.copy()
        nt[penalty_key] = penalties.astype(np.float32)
        nt["score"] = np.asarray(nt["score"], dtype=np.float64) - penalties

        # Subtract the penalty from the sequence reward. GRPO/GSPO advantages
        # depend only on the *sum* of ``token_level_scores`` over the response,
        # but we still place the subtraction on a real response token (the last
        # assistant token, mask==1) so it is never on padding.
        tls = batch.batch["token_level_scores"]
        resp_mask = batch.batch["response_mask"].bool()
        bsz, seq_len = tls.shape

        has_any = resp_mask.any(dim=1)
        rev = torch.flip(resp_mask.int(), dims=[1])
        last_idx = (seq_len - 1) - torch.argmax(rev, dim=1)
        last_idx = torch.where(
            has_any, last_idx, torch.full_like(last_idx, seq_len - 1)
        )

        pen_t = torch.as_tensor(penalties, dtype=tls.dtype, device=tls.device)
        rows = torch.arange(bsz, device=tls.device)
        tls[rows, last_idx] = tls[rows, last_idx] - pen_t

    # noqa: C901 — fit() is unavoidably long because it tracks v0.7.1's fit() body.
    def fit(self):
        """The training loop of PPO with DAPO-style filter_groups."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # --- filter_groups state -------------------------------------------------
        fg_cfg = self.config.algorithm.get("filter_groups", None)
        fg_enable = bool(fg_cfg is not None and fg_cfg.get("enable", False))
        fg_metric = (fg_cfg.get("metric", "score") if fg_cfg is not None else "score") or "score"
        fg_max_num_gen_batches = (
            fg_cfg.get("max_num_gen_batches", 0) if fg_cfg is not None else 0
        ) or 0

        # --- anchored tool-call penalty state ------------------------------------
        tp_cfg = self.config.algorithm.get("tool_penalty", None)
        tp_enable = bool(tp_cfg is not None and tp_cfg.get("enable", True))
        tp_alpha_win = float(tp_cfg.get("alpha_win", 0.1)) if tp_cfg is not None else 0.1
        tp_alpha_lose = float(tp_cfg.get("alpha_lose", 0.0)) if tp_cfg is not None else 0.0
        tp_beta_lose = float(tp_cfg.get("beta_lose", 0.0)) if tp_cfg is not None else 0.0
        tp_winner_threshold = (
            float(tp_cfg.get("winner_threshold", 1.0)) if tp_cfg is not None else 1.0
        )
        tp_std_threshold = (
            float(tp_cfg.get("std_threshold", 0.0)) if tp_cfg is not None else 0.0
        )
        if tp_enable:
            print(
                f"[tool_penalty] ENABLED. alpha_win={tp_alpha_win}, "
                f"alpha_lose={tp_alpha_lose}, beta_lose={tp_beta_lose}, "
                f"winner_threshold={tp_winner_threshold}, "
                f"std_threshold={tp_std_threshold}",
                flush=True,
            )
        # -------------------------------------------------------------------------

        # --- anchored answer-document penalty state ------------------------------
        # Same anchored-count mechanism as the tool-call penalty, but shapes the
        # number of doc_ids a rollout emits in its ``<answer>`` (num_answer_docs):
        # winners that answer with many docs (vs. the per-group winner mean) get
        # docked, discouraging the policy from padding the answer to game recall.
        adp_cfg = self.config.algorithm.get("answer_doc_penalty", None)
        adp_enable = bool(adp_cfg is not None and adp_cfg.get("enable", False))
        adp_alpha_win = float(adp_cfg.get("alpha_win", 0.05)) if adp_cfg is not None else 0.05
        adp_alpha_lose = float(adp_cfg.get("alpha_lose", 0.0)) if adp_cfg is not None else 0.0
        adp_beta_lose = float(adp_cfg.get("beta_lose", 0.0)) if adp_cfg is not None else 0.0
        adp_winner_threshold = (
            float(adp_cfg.get("winner_threshold", 1.0)) if adp_cfg is not None else 1.0
        )
        adp_std_threshold = (
            float(adp_cfg.get("std_threshold", 0.0)) if adp_cfg is not None else 0.0
        )
        if adp_enable:
            print(
                f"[answer_doc_penalty] ENABLED. alpha_win={adp_alpha_win}, "
                f"alpha_lose={adp_alpha_lose}, beta_lose={adp_beta_lose}, "
                f"winner_threshold={adp_winner_threshold}, "
                f"std_threshold={adp_std_threshold}",
                flush=True,
            )
        # -------------------------------------------------------------------------

        rollout_n = self.config.actor_rollout_ref.rollout.n
        train_bsz = self.config.data.train_batch_size

        # ``flush=True`` is important: Ray actors' stdout is pipe-buffered
        # and we want this banner to appear in nohup.out *before* the first
        # rollout begins, so an operator can verify at a glance that the
        # subclass's fit() is in fact running.
        if fg_enable:
            print(
                f"[filter_groups] ENABLED. metric={fg_metric!r}, "
                f"max_num_gen_batches={fg_max_num_gen_batches} (0 = unlimited), "
                f"train_batch_size={train_bsz}, rollout.n={rollout_n}",
                flush=True,
            )
        else:
            print(
                "[filter_groups] disabled — falling through to vanilla fit().",
                flush=True,
            )

        # These three accumulate across multiple dataloader iterations *within* a
        # single training step when filter_groups is enabled.
        accumulated_batch: DataProto | None = None
        accumulated_metrics: dict[str, Any] = {}
        accumulated_timing_raw: dict[str, Any] = {}
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_filtered_groups = 0
        total_filtered_all_zero = 0
        total_filtered_all_one = 0
        # -------------------------------------------------------------------------

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                # When refilling, reuse the previous step's metrics/timing buffers
                # so we don't lose timing info from earlier gen batches.
                if accumulated_batch is None:
                    metrics: dict[str, Any] = {}
                    timing_raw: dict[str, Any] = {}
                else:
                    metrics = accumulated_metrics
                    timing_raw = accumulated_timing_raw

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=rollout_n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        # NOTE: do NOT call self.checkpoint_manager.sleep_replicas() here.
                        # Upstream RayPPOTrainer.fit() sleeps right after generate because
                        # it only does one rollout per step. With filter_groups, when
                        # ``num_prompt_in_batch < train_bsz`` we ``continue`` and immediately
                        # call generate_sequences() again for the refill batch — vLLM would
                        # be asleep (KV cache deallocated, weights swapped out) and crash
                        # with ``_StaticCudaLauncher`` invalid-argument / illegal-memory
                        # access. The wake_up that pairs with sleep_replicas() is performed
                        # implicitly by ``checkpoint_manager.update_weights`` at line ~557,
                        # which only runs once per training step (after _update_actor), so
                        # nothing wakes vLLM back up between filter-loop iterations.
                        #
                        # Instead we sleep ONCE per step, after the filter loop has assembled
                        # a full ``batch`` — see the matching ``sleep_replicas()`` call right
                        # after the ``# === end filter_groups loop ===`` marker below.
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                gen_baseline_batch
                            )
                            # NOTE: same reason as above — sleep is deferred until the
                            # filter loop exits with a complete batch.
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            new_batch = new_batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(batch_reward)

                            reward_baseline_tensor = new_batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch = new_batch.repeat(repeat_times=rollout_n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                    # Persist reward into the batch *before* the filter so
                    # ``DataProto.concat`` carries it across gen batches.
                    # (The original v0.7.1 fit() does these two assignments
                    # inside the ``adv`` marked_timer block at lines 1452
                    # / 1455; we move them earlier and skip them below.)
                    new_batch.batch["token_level_scores"] = reward_tensor
                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update(
                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        )

                    # Group-relative anchored count penalties. Applied *before*
                    # the filter so the penalized reward feeds both filter_groups
                    # and the GRPO/GSPO advantage. No-op when disabled. Both stack
                    # additively on the same base reward (winner status is judged
                    # against ``base_score``, set by whichever runs first).
                    if tp_enable:
                        self._apply_count_penalty(
                            new_batch,
                            count_key="num_tool_calls",
                            penalty_key="tool_call_penalty",
                            label="tool_penalty",
                            alpha_win=tp_alpha_win,
                            alpha_lose=tp_alpha_lose,
                            beta_lose=tp_beta_lose,
                            winner_threshold=tp_winner_threshold,
                            std_threshold=tp_std_threshold,
                        )
                    if adp_enable:
                        self._apply_count_penalty(
                            new_batch,
                            count_key="num_answer_docs",
                            penalty_key="answer_doc_penalty",
                            label="answer_doc_penalty",
                            alpha_win=adp_alpha_win,
                            alpha_lose=adp_alpha_lose,
                            beta_lose=adp_beta_lose,
                            winner_threshold=adp_winner_threshold,
                            std_threshold=adp_std_threshold,
                        )

                    # ================== filter_groups loop ==================
                    if fg_enable:
                        # Resolve the per-rollout metric values.
                        if fg_metric not in new_batch.non_tensor_batch:
                            if fg_metric == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                                )
                            elif fg_metric == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                                )
                            else:
                                raise KeyError(
                                    f"filter_groups.metric={fg_metric!r} not present in "
                                    f"batch.non_tensor_batch and is not a known aggregate "
                                    f"(seq_reward / seq_final_reward). Available keys: "
                                    f"{list(new_batch.non_tensor_batch.keys())}"
                                )

                        metric_vals = np.asarray(new_batch.non_tensor_batch[fg_metric])
                        kept_uids, n_filtered, n_all_zero, n_all_one = _compute_kept_uids(
                            new_batch.non_tensor_batch["uid"], metric_vals
                        )

                        total_filtered_groups += n_filtered
                        total_filtered_all_zero += n_all_zero
                        total_filtered_all_one += n_all_one
                        num_prompt_in_batch += len(kept_uids)
                        num_gen_batches += 1

                        kept_set = set(kept_uids)
                        kept_traj_idxs = [
                            idx
                            for idx, uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if uid in kept_set
                        ]
                        new_batch = new_batch[kept_traj_idxs]

                        accumulated_batch = (
                            new_batch
                            if accumulated_batch is None
                            else DataProto.concat([accumulated_batch, new_batch])
                        )

                        if num_prompt_in_batch < train_bsz:
                            if fg_max_num_gen_batches <= 0 or num_gen_batches < fg_max_num_gen_batches:
                                print(
                                    f"[filter_groups] step={self.global_steps} "
                                    f"num_prompt_in_batch={num_prompt_in_batch} < "
                                    f"train_bsz={train_bsz}; num_gen_batches="
                                    f"{num_gen_batches}/{fg_max_num_gen_batches or 'inf'}. "
                                    f"Pulling another dataloader batch...",
                                    flush=True,
                                )
                                with marked_timer("stop_profile", timing_raw):
                                    next_step_profile = (
                                        self.global_steps + 1 in self.config.global_profiler.steps
                                        if self.config.global_profiler.steps is not None
                                        else False
                                    )
                                    self._stop_profiling(
                                        curr_step_profile and not next_step_profile
                                        if self.config.global_profiler.profile_continuous_steps
                                        else curr_step_profile
                                    )
                                    prev_step_profile = curr_step_profile
                                    curr_step_profile = next_step_profile
                                accumulated_metrics = metrics
                                accumulated_timing_raw = timing_raw
                                continue
                            else:
                                raise ValueError(
                                    f"num_gen_batches={num_gen_batches} >= "
                                    f"max_num_gen_batches={fg_max_num_gen_batches}. "
                                    f"Generated too many. Please check if your data are "
                                    f"too difficult. You could also try set "
                                    f"max_num_gen_batches=0 to enable endless trials."
                                )

                        # We have enough kept groups — trim to exact size.
                        traj_bsz = train_bsz * rollout_n
                        batch = accumulated_batch[:traj_bsz]
                    else:
                        batch = new_batch
                        num_gen_batches = 1
                    # ================ end filter_groups loop ================

                    # Now that the rollout phase for this step is fully done (we won't
                    # call generate_sequences() again until the next training step's
                    # update_weights wakes vLLM back up), it is safe — and necessary —
                    # to release the rollout engine's GPU memory back to the actor for
                    # the upcoming _update_actor call. This matches upstream's
                    # post-generate sleep, just deferred to after the filter loop.
                    self.checkpoint_manager.sleep_replicas()

                    # ``compute_response_mask`` / ``_balance_batch`` were
                    # already applied per gen batch above, but after the
                    # potential concat we redo balance on the assembled
                    # batch so DP ranks stay balanced wrt the *final*
                    # batch composition.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch.get(
                        "multi_modal_inputs", []
                    ):
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # ``token_level_scores`` and ``reward_extra_infos_dict``
                    # were already merged into ``batch`` above (before the
                    # filter). We deliberately don't repeat those
                    # assignments here so we don't clobber the accumulated
                    # batch with only the *last* gen batch's reward.

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                        "bypass_mode", False
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if (
                                "routed_experts" in batch.batch
                                and "routed_experts" in old_log_prob.batch
                            ):
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, (
                        f'"old_log_prob" not in {batch.batch.keys()=}'
                    )

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # ``batch.batch["token_level_scores"]`` and the
                        # reward_extra_infos_dict merge were done above
                        # (pre-filter) — see notes there.
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=rollout_n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in (
                    "gdpo",
                    AdvantageEstimator.GDPO,
                ):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                # filter_groups bookkeeping metrics
                metrics["train/num_gen_batches"] = float(num_gen_batches)
                if fg_enable:
                    metrics["filter_groups/total_filtered_groups"] = float(total_filtered_groups)
                    metrics["filter_groups/total_generated_groups"] = float(
                        num_gen_batches * train_bsz
                    )
                    # Pre-filter breakdown of dropped (zero-variance) groups by
                    # their common metric value. Overrides the post-filter 0s
                    # emitted by ``compute_filter_groups_metrics`` (which sees
                    # the trimmed batch, where n_filtered is always 0).
                    metrics["filter_groups/n_filtered_all_zero"] = float(total_filtered_all_zero)
                    metrics["filter_groups/n_filtered_all_one"] = float(total_filtered_all_one)

                if "response_len" in batch.non_tensor_batch:
                    _rlen = np.asarray(batch.non_tensor_batch["response_len"], dtype=np.float32)
                    metrics["response_length/q_99"] = float(np.percentile(_rlen, 99))

                # Anchored tool-call penalty diagnostics: mean tool calls among
                # winners (base reward == 1) vs. zero-reward rollouts, plus the
                # mean penalty actually applied. ``base_score`` is the pre-penalty
                # reward stashed by ``_apply_count_penalty``.
                if (
                    tp_enable
                    and "base_score" in batch.non_tensor_batch
                    and "num_tool_calls" in batch.non_tensor_batch
                ):
                    _base = np.asarray(batch.non_tensor_batch["base_score"], dtype=np.float64)
                    _ntc = np.asarray(batch.non_tensor_batch["num_tool_calls"], dtype=np.float64)
                    _win_mask = _base >= (tp_winner_threshold - 1e-6)
                    _zero_mask = _base <= 1e-6
                    metrics["tool_penalty/frac_winners"] = float(np.mean(_win_mask))

                    # Mean per-group std of the (pre-penalty) reward — the
                    # quantity gating the penalty via ``std_threshold``.
                    if "uid" in batch.non_tensor_batch:
                        _uids = np.asarray(batch.non_tensor_batch["uid"])
                        _grp_stds = [
                            float(np.std(_base[_uids == u])) for u in np.unique(_uids)
                        ]
                        if _grp_stds:
                            metrics["tool_penalty/reward_std_mean"] = float(
                                np.mean(_grp_stds)
                            )
                    if _win_mask.any():
                        metrics["tool_penalty/mean_tool_calls_winners"] = float(
                            _ntc[_win_mask].mean()
                        )
                    if _zero_mask.any():
                        metrics["tool_penalty/mean_tool_calls_zero_reward"] = float(
                            _ntc[_zero_mask].mean()
                        )
                    if "tool_call_penalty" in batch.non_tensor_batch:
                        _pen = np.asarray(
                            batch.non_tensor_batch["tool_call_penalty"], dtype=np.float64
                        )
                        metrics["tool_penalty/mean"] = float(_pen.mean())

                # Anchored answer-document penalty diagnostics: mean answer-doc
                # count among winners vs. zero-reward rollouts, plus the mean
                # penalty actually applied.
                if (
                    adp_enable
                    and "base_score" in batch.non_tensor_batch
                    and "num_answer_docs" in batch.non_tensor_batch
                ):
                    _base = np.asarray(batch.non_tensor_batch["base_score"], dtype=np.float64)
                    _nad = np.asarray(batch.non_tensor_batch["num_answer_docs"], dtype=np.float64)
                    _win_mask = _base >= (adp_winner_threshold - 1e-6)
                    _zero_mask = _base <= 1e-6
                    metrics["answer_doc_penalty/frac_winners"] = float(np.mean(_win_mask))
                    if _win_mask.any():
                        metrics["answer_doc_penalty/mean_answer_docs_winners"] = float(
                            _nad[_win_mask].mean()
                        )
                    if _zero_mask.any():
                        metrics["answer_doc_penalty/mean_answer_docs_zero_reward"] = float(
                            _nad[_zero_mask].mean()
                        )
                    if "answer_doc_penalty" in batch.non_tensor_batch:
                        _pen = np.asarray(
                            batch.non_tensor_batch["answer_doc_penalty"], dtype=np.float64
                        )
                        metrics["answer_doc_penalty/mean"] = float(_pen.mean())

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                # Advance by the number of dataloader batches this optimizer step
                # consumed (filter_groups pulls extra batches to refill the train
                # batch). ``total_training_steps`` is computed upstream as
                # ``len(train_dataloader) * total_epochs`` (= dataloader batches),
                # so stepping by ``num_gen_batches`` lets the bar reach 100%.
                # Stepping by 1 would cap the bar at ~50% with ~2 batches/step.
                progress_bar.update(num_gen_batches)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

                # Reset filter_groups accumulator for the next training step.
                accumulated_batch = None
                accumulated_metrics = {}
                accumulated_timing_raw = {}
                num_prompt_in_batch = 0
                num_gen_batches = 0
                total_filtered_groups = 0
                total_filtered_all_zero = 0
                total_filtered_all_one = 0

    # ------------------------------------------------------------------
    # Validation with trajectory saving
    # ------------------------------------------------------------------

    def _validate(self, merged: bool = False):
        """Run validation and optionally save full rollout trajectories.

        When ``trainer.val_trajectory_dir`` is set in the config, saves a
        JSONL file per validation call to ``<val_trajectory_dir>/<step>.jsonl``.
        Each line is one rollout with: step, uid, score, messages_full and
        every key returned by the reward function (ndcg, f_beta, stopped_reason,
        num_tool_calls, etc.).

        All other behaviour is identical to the parent ``_validate``.
        """
        val_traj_dir: str | None = self.config.trainer.get("val_trajectory_dir", None)

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        # Extra accumulators for trajectory saving.
        traj_messages_full: list = []
        traj_uids: list = []
        traj_gold_ids: list = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                self.checkpoint_manager.update_weights(self.global_steps)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # Collect messages_full for trajectory saving. Always extend by the
            # batch size (filling None when absent) so traj_* stay aligned 1:1
            # with sample_scores / reward_extra_infos_dict.
            if val_traj_dir:
                batch_uids = test_batch.non_tensor_batch["uid"].tolist()
                if "messages_full" in test_output_gen_batch.non_tensor_batch:
                    batch_msgs = test_output_gen_batch.non_tensor_batch["messages_full"].tolist()
                else:
                    batch_msgs = [None] * len(batch_uids)
                if "gold_doc_ids" in test_output_gen_batch.non_tensor_batch:
                    batch_gold = test_output_gen_batch.non_tensor_batch["gold_doc_ids"].tolist()
                else:
                    batch_gold = [None] * len(batch_uids)
                traj_uids.extend(batch_uids)
                traj_messages_full.extend(batch_msgs)
                traj_gold_ids.extend(batch_gold)

            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        if val_traj_dir and traj_messages_full:
            self._dump_val_trajectories(
                uids=traj_uids,
                messages_full=traj_messages_full,
                gold_ids=traj_gold_ids,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_traj_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
            )

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _dump_val_trajectories(
        self,
        uids: list,
        messages_full: list,
        gold_ids: list,
        scores: list[float],
        reward_extra_infos_dict: dict[str, list],
        dump_path: str,
    ) -> None:
        """Save validation rollout trajectories to a JSONL file.

        Each line: {step, uid, score, gold_ids, messages_full, <reward metrics>}.
        File is named ``<dump_path>/<global_steps>.jsonl``.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(uids)
        lines = []
        for i in range(n):
            entry: dict[str, Any] = {
                "step": self.global_steps,
                "uid": uids[i] if i < len(uids) else None,
                "score": scores[i] if i < len(scores) else None,
                "gold_ids": gold_ids[i] if i < len(gold_ids) else None,
                "messages_full": messages_full[i] if i < len(messages_full) else None,
            }
            for key, vals in reward_extra_infos_dict.items():
                if len(vals) == n:
                    entry[key] = vals[i]
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Saved {n} val trajectories → {filename}")
