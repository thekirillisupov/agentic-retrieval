"""Wrapper around ``verl.trainer.main_ppo`` that turns on DAPO-style
``algorithm.filter_groups`` and logs ``filter_groups/*`` diagnostics.

What this wrapper does
----------------------

``verl==0.7.1`` exposes ``algorithm.filter_groups`` in
``FilterGroupsConfig`` but its ``RayPPOTrainer.fit`` does not actually
consume it — the only consumer in the upstream tree is the DAPO recipe
(``recipe/dapo/dapo_ray_trainer.py``). We don't want to fork verl just
for this, so this wrapper installs two patches:

1. Module-level (driver + actor, applied at import time of this script):

   * ``verl.trainer.main_ppo.TaskRunner`` →
     :class:`_TaskRunnerWithFilterGroups`. Used so we can intercept
     ``run()`` for Patch 2.
   * ``verl.trainer.main_ppo.RayPPOTrainer`` →
     :class:`grpo.filter_groups_trainer.FilterGroupsRayPPOTrainer`.
     A subclass that overrides ``fit()`` with a DAPO-style filter loop:
     zero-variance prompt groups (groups whose ``score`` is identical
     across all ``rollout.n`` rollouts) are dropped before the policy
     update, and we pull additional fresh dataloader batches up to
     ``algorithm.filter_groups.max_num_gen_batches`` times to refill the
     training batch with non-zero-variance groups.

     IMPORTANT: this rebind has to happen at *module* level (not
     inside ``TaskRunner.run``). Upstream ``TaskRunner.run`` resolves
     ``RayPPOTrainer`` via its own ``__globals__`` (= ``vm.__dict__``)
     at line 343 of ``verl/trainer/main_ppo.py``. We previously did
     the rebind inside our ``run()`` *before* delegating to
     ``super().run``, but in the live training that mutation didn't
     land in time — upstream's ``trainer = RayPPOTrainer(...)`` still
     resolved to the original. Doing it at module load (= as soon as
     the actor imports ``main_ppo``) is the robust fix.

2. Actor-side, inside ``TaskRunner.run`` (needs ``config``):

   * ``verl.trainer.ppo.ray_trainer.compute_data_metrics`` and
     ``grpo.filter_groups_trainer.compute_data_metrics`` →
     :func:`grpo.filter_groups_metrics.wrap_compute_data_metrics`.

     We patch *both* namespaces because Python resolves the bare
     ``compute_data_metrics`` reference via the calling function's
     ``__globals__``: upstream ``RayPPOTrainer.fit`` lives in
     ``ray_trainer.__dict__``, but our override
     ``FilterGroupsRayPPOTrainer.fit`` lives in
     ``grpo.filter_groups_trainer.__dict__`` and has its own bound
     copy of the symbol (from the ``from verl.trainer.ppo.metric_utils
     import compute_data_metrics`` at the top of that module). Patch
     one namespace and you silently miss the other.

     The wrapper adds ``filter_groups/n_groups``,
     ``filter_groups/n_filtered``, ``filter_groups/n_kept``,
     ``filter_groups/filter_ratio``,
     ``filter_groups/effective_batch_size``, and
     ``filter_groups/nominal_batch_size`` to the per-step metrics.
     With filtering ON these are computed on the *post-filter* batch
     so ``n_filtered`` ≈ 0 every step. The richer
     ``train/num_gen_batches``,
     ``filter_groups/total_filtered_groups``, and
     ``filter_groups/total_generated_groups`` metrics emitted by the
     trainer itself show how much rollout compute the filter
     absorbed.

No verl source file is modified. To roll back, set
``algorithm.filter_groups.enable=false`` in the config: the subclass's
``fit()`` short-circuits and behaves identically to upstream
``RayPPOTrainer.fit``.

Launch exactly like ``verl.trainer.main_ppo``::

    PYTHONPATH=".:$PYTHONPATH" python scripts/main_ppo.py \
        --config-path "$(realpath configs)" \
        --config-name gspo_qwen3_moe \
        trainer.n_gpus_per_node=8 trainer.nnodes=1
"""

from __future__ import annotations

import verl.trainer.main_ppo as _verl_main_ppo

# Patch 1 (trainer subclass) — applied at MODULE level so it's in place
# before upstream ``TaskRunner.run`` does its ``trainer = RayPPOTrainer(...)``
# lookup. This module is re-executed inside the Ray actor process when
# Ray imports ``main_ppo`` to unpickle the actor class, so the rebind
# lands on the actor too (the ``if __name__ == "__main__"`` guard below
# stops ``main()`` from being re-entered there).
from grpo.filter_groups_trainer import FilterGroupsRayPPOTrainer

_verl_main_ppo.RayPPOTrainer = FilterGroupsRayPPOTrainer


class _TaskRunnerWithFilterGroups(_verl_main_ppo.TaskRunner):
    """``TaskRunner`` subclass that installs Patch 2 (the
    ``compute_data_metrics`` diagnostic wrapper) on the actor before
    delegating to the upstream ``run`` method.

    Patch 1 (the ``RayPPOTrainer`` swap) is applied at module load —
    see the top of this file — because doing it from inside ``run()``
    races upstream's symbol resolution.
    """

    def run(self, config):
        from grpo.filter_groups_metrics import wrap_compute_data_metrics
        import grpo.filter_groups_trainer as fgt
        import verl.trainer.ppo.ray_trainer as ray_trainer

        # Patch 2: diagnostic metrics (filter_groups/* keys in wandb).
        # We must rebind on BOTH namespaces:
        #   * ``ray_trainer.compute_data_metrics`` is what upstream
        #     ``RayPPOTrainer.fit`` (running if/when fg.enable=False
        #     short-circuits via super) resolves the bare symbol to.
        #   * ``fgt.compute_data_metrics`` is what our overridden
        #     ``FilterGroupsRayPPOTrainer.fit`` resolves the bare
        #     symbol to (because it imports
        #     ``from verl.trainer.ppo.metric_utils import
        #     compute_data_metrics`` at module top, which creates a
        #     fresh binding in ``fgt.__dict__``).
        # Patch the wrapper exactly once and assign the same wrapped
        # function to both namespaces so the diagnostic metrics appear
        # whichever ``fit()`` ends up running.
        wrapped = wrap_compute_data_metrics(
            ray_trainer.compute_data_metrics,
            algorithm_cfg=config.algorithm,
            rollout_n=config.actor_rollout_ref.rollout.n,
        )
        ray_trainer.compute_data_metrics = wrapped
        fgt.compute_data_metrics = wrapped
        return super().run(config)


_verl_main_ppo.TaskRunner = _TaskRunnerWithFilterGroups


if __name__ == "__main__":
    _verl_main_ppo.main()
