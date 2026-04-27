#!/usr/bin/env bash
# Run baseline + agent eval. Assumes vLLM and tool server are already up.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
OUT_DIR="${OUT_DIR:-data/processed}"

echo "[1/2] single-shot baseline"
python -m eval_.baseline --config "${CONFIG}" --out "${OUT_DIR}/baseline_results.json"

echo "[2/2] agent eval"
if [[ "${BUDGET_SWEEP:-0}" == "1" ]]; then
    python -m eval_.run_eval --config "${CONFIG}" --budget-sweep \
        --out "${OUT_DIR}/agent_sweep.json"
else
    python -m eval_.run_eval --config "${CONFIG}" \
        --out "${OUT_DIR}/agent_results.json"
fi

echo "done."
