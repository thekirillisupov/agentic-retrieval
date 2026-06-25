#!/usr/bin/env bash
# W8A8 INT8 quantization (int8 weights + int8 per-token-dynamic activations) of
# the merged actor, for testing the A100 int8-tensor-core compute path against the
# W8A16 build (scripts/quantize_w8a16.sh).
#
# SmoothQuant is OFF by default so the GatedDeltaNet norm-corruption bug
# (llm-compressor#2059) cannot trigger. Pass SMOOTHQUANT=1 to A/B test it.
#
# Usage:
#   MODEL=checkpoints/gspo_qwen3_moe/global_step_40/actor/huggingface \
#   QUESTIONS=data/calib/questions.jsonl \
#   OUTPUT=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
#   bash scripts/quantize_w8a8.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:?set MODEL to the merged HF checkpoint dir}"
QUESTIONS="${QUESTIONS:?set QUESTIONS to your calibration questions file}"
OUTPUT="${OUTPUT:-checkpoints/quantized/qwen3_5_35b_a3b_w8a8}"
PROMPT_VERSION="${PROMPT_VERSION:-v2_search_only}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"

SMOOTHQUANT_FLAG=()
if [[ "${SMOOTHQUANT:-0}" == "1" ]]; then
  SMOOTHQUANT_FLAG=(--smoothquant)
fi

cd "${REPO_ROOT}"
python -m quantize.quantize \
  --model "${MODEL}" \
  --questions "${QUESTIONS}" \
  --question-field "${QUESTION_FIELD}" \
  --output "${OUTPUT}" \
  --scheme W8A8 \
  --prompt-version "${PROMPT_VERSION}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  "${SMOOTHQUANT_FLAG[@]}" \
  "$@"
