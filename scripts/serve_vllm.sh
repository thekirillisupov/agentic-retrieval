#!/usr/bin/env bash
# Serve a tool-calling actor over vLLM's OpenAI-compatible API.
# Defaults target Qwen3.5-35B-A3B (tensor-parallel=2 on GPUs 0,1); every
# architecture-specific knob is env-overridable so the same script serves
# other families (see scripts/serve_vllm_gemma.sh for Gemma-4).
# max-model-len=32k+ is plenty for 8-turn rollouts and keeps the KV cache sane.
#
# Tool-call format: Qwen3.5 emits XML inside <tool_call>, NOT Hermes JSON.
# Use qwen3_xml (the modern streaming parser); qwen3_coder is the older fallback.
# Thinking is off by default: faster rollouts, and avoids vLLM 0.19's reasoning-parser
# bug where <tool_call> emitted inside <think> blocks never reaches the tool parser
# (https://github.com/vllm-project/vllm/issues/39056). Set ENABLE_THINKING=true to opt in.
#
# Architecture-specific knobs (all optional):
#   TOOL_CALL_PARSER      --tool-call-parser      (default qwen3_xml)
#   REASONING_PARSER      --reasoning-parser      (default qwen3; empty = omit)
#   CHAT_TEMPLATE         --chat-template         (default: model's own; empty = omit)
#   CHAT_TEMPLATE_KWARGS  --default-chat-template-kwargs
#                         (default {"enable_thinking": $ENABLE_THINKING}; empty = omit)
#
# Usage: MODEL=Qwen/Qwen3-14B bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-65536}"
GPUS="${GPUS:-0,1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
# Qwen toggles thinking via a chat-template kwarg; other families (e.g. Gemma-4)
# use the same knob. Default preserves the historical behaviour; set empty to omit.
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": ${ENABLE_THINKING}}}"

ARGS=(
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --port "${PORT}"
)
# Reasoning parser must ride together with the tool-call parser for families
# that interleave thought channels and tool calls (Gemma-4); omit when empty.
[[ -n "${REASONING_PARSER}" ]] && ARGS+=(--reasoning-parser "${REASONING_PARSER}")
# Some tool-call parsers require a matching chat template that differs from the
# HF-repo default (e.g. Gemma-4's tool_chat_template_gemma4.jinja).
[[ -n "${CHAT_TEMPLATE}" ]] && ARGS+=(--chat-template "${CHAT_TEMPLATE}")
[[ -n "${CHAT_TEMPLATE_KWARGS}" ]] && ARGS+=(--default-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")

CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" "${ARGS[@]}"
