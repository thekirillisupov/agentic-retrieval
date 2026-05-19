#!/usr/bin/env bash
# Launch GSPO training on the Qwen3.5-35B-A3B MoE actor.
#
# GSPO = Group Sequence Policy Optimization (Qwen team). Same group-relative
# advantages as GRPO, but a sequence-level importance ratio in the policy loss
# (more stable for MoE — top-k routing makes per-token ratios noisy).
#
# Backend: Megatron with expert parallelism (EP=8), vLLM rollout with TP=4.
# This is the official verl recipe for Qwen3-30B-A3B GSPO, adapted to our
# AgentLoop / NDCG reward.
#
# Prerequisites:
#   * verl installed with Megatron-LM support (see verl docs; the FSDP-only
#     install used by scripts/train_grpo.sh is NOT enough).
#   * scripts/build_grpo_data.sh has been run -> grpo_{train,val}.parquet.
#   * Tool server is up:  bash scripts/serve_tool.sh
#
# Override individual fields with KEY=VALUE extra args, e.g.:
#   bash scripts/train_gspo.sh actor_rollout_ref.rollout.n=16
set -euo pipefail

CONFIG="${CONFIG:-configs/gspo_qwen3_moe.yaml}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"

EXTRA_OVERRIDES=("$@")

# Required by Megatron + NCCL for deterministic comm scheduling.
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
# NOTE: expandable_segments:True is intentionally NOT set here.
# vLLM's CuMemAllocator (used by free_cache_engine / --enable_sleep_mode)
# hard-asserts that expandable segments are disabled — the two allocator
# strategies are mutually exclusive. OOM on the Adam step is mitigated by
# the actor's param_offload + grad_offload + optimizer_offload settings.

# Use scripts/main_ppo.py (a thin wrapper around verl.trainer.main_ppo) so that
# the `algorithm.filter_groups` block in the config produces would-be-filtered
# / effective-batch-size diagnostics in WandB. See grpo/filter_groups_metrics.py.
CUDA_VISIBLE_DEVICES="${GPUS}" \
PYTHONPATH=".:${PYTHONPATH:-}" \
python scripts/main_ppo.py \
    --config-path "$(realpath "$(dirname "${CONFIG}")")" \
    --config-name "$(basename "${CONFIG}" .yaml)" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${NNODES}" \
    "${EXTRA_OVERRIDES[@]}"
