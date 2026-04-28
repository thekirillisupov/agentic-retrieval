#!/usr/bin/env bash
# Serve Qwen3.5-35B-A3B with tensor-parallel=2 on GPUs 0,1.
# max-model-len=32k is more than enough for 8-turn rollouts; keeps KV cache sane.
#
# Tool-call format: Qwen3.5 emits XML inside <tool_call>, NOT Hermes JSON.
# Use qwen3_xml (the modern streaming parser); qwen3_coder is the older fallback.
# Thinking is off by default: faster rollouts, and avoids vLLM 0.19's reasoning-parser
# bug where <tool_call> emitted inside <think> blocks never reaches the tool parser
# (https://github.com/vllm-project/vllm/issues/39056). Set ENABLE_THINKING=true to opt in.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-32768}"
GPUS="${GPUS:-0,1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_THINKING="${ENABLE_THINKING:-true}"

CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --default-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
  --port "${PORT}"
