#!/usr/bin/env bash
# Serve Qwen3.5-35B-A3B with tensor-parallel=2 on GPUs 0,1.
# max-model-len=32k is more than enough for 8-turn rollouts; keeps KV cache sane.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-32768}"
GPUS="${GPUS:-0,1}"

CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --port "${PORT}"
