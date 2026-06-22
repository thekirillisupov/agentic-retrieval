#!/usr/bin/env bash
# Weight-only INT8 (W8A16) quantization of the merged actor for A100 inference.
#
# Prereqs:
#   * A merged HF checkpoint (run scripts/dist_ckpt_to_hf.sh on the Megatron actor first).
#   * Generated calibration questions (.jsonl/.json/.txt/.parquet); see quantize/README.md.
#   * An inference/quant env with llmcompressor>=0.9.0 (NOT the Megatron training env).
#
# Usage:
#   MODEL=checkpoints/gspo_qwen3_moe/global_step_40/actor/huggingface \
#   QUESTIONS=data/calib/questions.jsonl \
#   OUTPUT=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 \
#   bash scripts/quantize_w8a16.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:?set MODEL to the merged HF checkpoint dir}"
QUESTIONS="${QUESTIONS:?set QUESTIONS to your calibration questions file}"
OUTPUT="${OUTPUT:-checkpoints/quantized/qwen3_5_35b_a3b_w8a16}"
PROMPT_VERSION="${PROMPT_VERSION:-v2_search_only}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"

cd "${REPO_ROOT}"
python -m quantize.quantize \
  --model "${MODEL}" \
  --questions "${QUESTIONS}" \
  --question-field "${QUESTION_FIELD}" \
  --output "${OUTPUT}" \
  --prompt-version "${PROMPT_VERSION}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  "$@"
