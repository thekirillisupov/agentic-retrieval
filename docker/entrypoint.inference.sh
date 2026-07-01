#!/usr/bin/env bash
# Inference entrypoint: bring up vLLM (serving the quantized actor) and the
# agent harness HTTP service in one container.
#
#   [vLLM :8000 OpenAI API]  <--local-->  [harness /retrieve :8080]  -->  external SEARCH_URL
#
# Weights are mounted at runtime (default /models/qwen3_5_35b_a3b_w8a8). vLLM
# auto-detects the W8A8 compressed-tensors scheme from config.json, so no
# --quantization flag is needed. Flags mirror scripts/serve_vllm_int8.sh so the
# served behaviour matches training (qwen3_xml tool parser, thinking off).
set -euo pipefail

# --- model / vLLM ---
MODEL="${MODEL:-/models/qwen3_5_35b_a3b_w8a8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-agentic-retrieval}"
VLLM_PORT="${VLLM_PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# --- harness service ---
AGENT_HOST="${AGENT_HOST:-0.0.0.0}"
AGENT_PORT="${AGENT_PORT:-8080}"
export AGENT_CONFIG="${AGENT_CONFIG:-configs/inference.yaml}"
# The external retrieval service (embedder + reranker + index). REQUIRED.
export SEARCH_URL="${SEARCH_URL:-http://localhost:8100}"
# Wire the harness to the co-located vLLM and the served model name.
export VLLM_URL="${VLLM_URL:-http://localhost:${VLLM_PORT}/v1}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export HOST="${AGENT_HOST}"
export PORT="${AGENT_PORT}"

if [[ ! -e "${MODEL}" ]]; then
  echo "[entrypoint] ERROR: weights not found at ${MODEL}." >&2
  echo "[entrypoint] Mount them, e.g. -v /path/to/qwen3_5_35b_a3b_w8a8:${MODEL}:ro" >&2
  exit 1
fi

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[entrypoint] stopping vLLM (pid ${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[entrypoint] launching vLLM: model=${MODEL} served-name=${SERVED_MODEL_NAME} tp=${TP}"
vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --default-chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
  --port "${VLLM_PORT}" &
VLLM_PID=$!

echo "[entrypoint] waiting for vLLM on :${VLLM_PORT} ..."
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[entrypoint] ERROR: vLLM exited before becoming ready" >&2
    wait "${VLLM_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
echo "[entrypoint] vLLM ready."

echo "[entrypoint] launching harness service on :${AGENT_PORT} (SEARCH_URL=${SEARCH_URL})"
exec uvicorn agent.service:app --host "${AGENT_HOST}" --port "${AGENT_PORT}"
