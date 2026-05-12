#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TP=2, EP=8, PP=1, CP=1, ETP=1 → world_size must be a multiple of 8
# By default also writes Megatron dist_checkpointing under ${HF_OUT:-checkpoints/qwen3_5_35b_TP2_EP8_PP1}/dist_ckpt
# (override with MEGATRON_DIST_DIR or --megatron-dist-dir). Extra args: scripts/model_merger.py --help
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/model_merger.py" "$@"
