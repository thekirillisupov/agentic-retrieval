#!/usr/bin/env bash
# Tool server: embedder on GPU 2; faiss on CPU by default.
# When index.use_gpu: true in the config, also expose GPU 3:
#   GPUS=2,3 bash scripts/serve_tool.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
GPUS="${GPUS:-2}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m tool_server.main --config "${CONFIG}"
