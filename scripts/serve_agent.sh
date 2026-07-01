#!/usr/bin/env bash
# Run the agent harness HTTP service standalone (no vLLM launch — point it at an
# already-running vLLM via VLLM_URL). Inside the inference image the entrypoint
# starts both; use this for local dev or a harness-only deployment.
#
# Usage:
#   SEARCH_URL=http://localhost:8100 VLLM_URL=http://localhost:8000/v1 \
#     bash scripts/serve_agent.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export AGENT_CONFIG="${AGENT_CONFIG:-configs/inference.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

exec uvicorn agent.service:app --host "${HOST}" --port "${PORT}"
