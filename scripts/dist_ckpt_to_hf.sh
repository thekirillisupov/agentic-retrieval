#!/usr/bin/env bash
#bash scripts/dist_ckpt_to_hf.sh \
#  --actor-dir checkpoints/gspo_qwen3_moe/global_step_N/actor \
#  --hf-index ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Must match the Megatron parallel layout used during training (see configs/gspo_qwen3_moe.yaml).
# TP=2, EP=8 → world_size 8 (one DP rank). Override NPROC_PER_NODE if you change TP/EP.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/dist_ckpt_to_hf.py" "$@"
