#!/usr/bin/env bash
# Classify SBOL questions (simple vs complex) and build dialog train.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
EVAL="${EVAL:-data/processed/sbol/eval.jsonl}"
INDEX_DIR="${INDEX_DIR:-indexes/sbol}"
RAW="${RAW:-data/raw/sbol/faq_index_28_apr.json}"
OUT_DIR="${OUT_DIR:-data/processed/sbol}"
TOP_K="${TOP_K:-3}"
PROMPT_VERSION="${PROMPT_VERSION:-v2_search_only}"
VAL_FRAC="${VAL_FRAC:-0.05}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON}" -m indexing.compose_sbol_train \
    --eval "${EVAL}" \
    --index-dir "${INDEX_DIR}" \
    --raw "${RAW}" \
    --out-dir "${OUT_DIR}" \
    --top-k "${TOP_K}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --write-grpo \
    --prompt-version "${PROMPT_VERSION}" \
    --val-frac "${VAL_FRAC}"

echo "done. See ${OUT_DIR}/compose_sbol_stats.json"
