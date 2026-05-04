#!/usr/bin/env bash
# Launch GRPO training with veRL.
#
# Prerequisites:
#   * `pip install -e .[verl]` (pulls in verl, ray, hydra-core, ...)
#   * scripts/build_grpo_data.sh has been run -> grpo_{train,val}.parquet exist
#   * tool server is up:  bash scripts/serve_tool.sh
#
# This script does NOT start its own vLLM server — veRL spins up an internal
# rollout server for the actor (the model under training). The external vLLM
# server in scripts/serve_vllm.sh is for inference/eval only and is not used
# during training.
set -euo pipefail

CONFIG="${CONFIG:-configs/grpo_qwen3_14b.yaml}"
GPUS="${GPUS:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
NNODES="${NNODES:-1}"

# We pass the YAML config directly. veRL accepts both Hydra-style overrides and
# a single `+config_path=...` style; here we use the config-file form for
# readability and let users override individual fields with `KEY=VALUE` extra
# args, e.g.:
#   bash scripts/train_grpo.sh actor_rollout_ref.rollout.n=16
EXTRA_OVERRIDES=("$@")

CUDA_VISIBLE_DEVICES="${GPUS}" \
PYTHONPATH=".:${PYTHONPATH:-}" \
python -m verl.trainer.main_ppo \
    --config-path "$(realpath "$(dirname "${CONFIG}")")" \
    --config-name "$(basename "${CONFIG}" .yaml)" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${NNODES}" \
    "${EXTRA_OVERRIDES[@]}"
