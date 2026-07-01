#!/usr/bin/env bash
# Serve the W8A16-quantized actor on A100. vLLM auto-detects the quantization
# from the compressed-tensors checkpoint (config.json -> quantization_config),
# so no --quantization flag is needed.
#
# 35B int8 weights (~35 GB) fit a single 80 GB A100, so TP=1 by default; the
# GatedDeltaNet/router/lm_head layers stay bf16 (see quantize/quantize.py:IGNORE).
# Everything else mirrors scripts/serve_vllm.sh (same tool-call / reasoning parsers,
# thinking off) so the served behaviour matches training.
#
# Usage: MODEL=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 bash scripts/serve_vllm_int8.sh
set -euo pipefail

MODEL="${MODEL:-checkpoints/quantized/qwen3_5_35b_a3b_w8a16}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-32768}"
GPUS="${GPUS:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --default-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
  --port "${PORT}"
