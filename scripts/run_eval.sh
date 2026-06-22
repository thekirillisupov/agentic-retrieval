#!/usr/bin/env bash
# Run baseline + agent eval. Assumes vLLM and tool server are already up.
# Usage: CONFIG=configs/musique.yaml MODEL_NAME=checkpoints/gspo_qwen3_moe/global_step_60/hf_model bash scripts/run_eval.sh
#   bash scripts/run_eval.sh --wrap-client
#   BUDGET_SWEEP=1 bash scripts/run_eval.sh --wrap-client
#
# --wrap-client wraps each question as <client>...</client> (v2_search_only format).
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
OUT_DIR="${OUT_DIR:-data/processed/musique}"
MODEL_NAME="${MODEL_NAME:-}"

MODEL_ARGS=()
if [[ -n "${MODEL_NAME}" ]]; then
    MODEL_ARGS=(--model-name "${MODEL_NAME}")
fi

echo "[1/2] single-shot baseline"
python -m eval_.baseline --config "${CONFIG}" --out "${OUT_DIR}/baseline_results.json"

echo "[2/2] agent eval"
if [[ "${BUDGET_SWEEP:-0}" == "1" ]]; then
    python -m eval_.run_eval --config "${CONFIG}" --budget-sweep \
        "${MODEL_ARGS[@]}" \
        --out "${OUT_DIR}/agent_sweep.json" \
        "$@"
else
    python -m eval_.run_eval --config "${CONFIG}" \
        "${MODEL_ARGS[@]}" \
        --out "${OUT_DIR}/agent_results.json" \
        "$@"
fi

echo "done."
