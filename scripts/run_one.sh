#!/usr/bin/env bash
# Run the agent on a single question with full thinking + tool trace.
# Assumes vLLM and tool server are already up (same as run_eval.sh).
# Use --retrieve-only to query the index only (tool server required, no vLLM).
#
# Examples:
#   bash scripts/run_one.sh --index 0
#   bash scripts/run_one.sh --question-id 2hop__460946_294723
#   bash scripts/run_one.sh --question "Who founded Apple?"
#   bash scripts/run_one.sh --index 0 --wrap-client
#   CONFIG=configs/sbol.yaml bash scripts/run_one.sh --index 5
#   MODEL_NAME=checkpoints/.../hf_model bash scripts/run_one.sh --index 0
#
# Retrieve-only (no model):
#   bash scripts/run_one.sh --retrieve-only --index 0
#   bash scripts/run_one.sh --retrieve-only --index 0 --top-k 20
#   bash scripts/run_one.sh --retrieve-only --question "Who founded Apple?" --top-k 5
#   bash scripts/run_one.sh --retrieve-only --index 0 --wrap-client
#
# --wrap-client wraps the question as <client>...</client> (v2_search_only format).
set -euo pipefail

CONFIG="${CONFIG:-configs/sbol.yaml}"
OUT="${OUT:-data/processed/one_question_debug.json}"
MODEL_NAME="${MODEL_NAME:-}"

MODEL_ARGS=()
if [[ -n "${MODEL_NAME}" ]]; then
    MODEL_ARGS=(--model-name "${MODEL_NAME}")
fi

python -m eval_.run_one \
    --config "${CONFIG}" \
    --out "${OUT}" \
    "${MODEL_ARGS[@]}" \
    "$@"

echo "saved: ${OUT}"
