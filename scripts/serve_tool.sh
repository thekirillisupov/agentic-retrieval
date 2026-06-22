#!/usr/bin/env bash
# Tool server. GPUS sets CUDA_VISIBLE_DEVICES; per-source index.gpu_id values in
# the config index into that visible list (0 = first GPU in GPUS, etc.).
#
# Single CPU index (legacy default.yaml):       GPUS=2 bash scripts/serve_tool.sh
# Single GPU index (index.use_gpu: true):       GPUS=2,3 bash scripts/serve_tool.sh
# Multi-source split (configs/unioned.yaml, all indexes CPU,
#   sources: musique, sbol, rnd, ckr):            GPUS=0 CONFIG=configs/unioned.yaml bash scripts/serve_tool.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-configs/default.yaml}"
GPUS="${GPUS:-2}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m tool_server.main --config "${CONFIG}"
