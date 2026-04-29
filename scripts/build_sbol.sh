#!/usr/bin/env bash
# Parse SBOL FAQ → build FAISS index.
# Idempotent: re-running is safe but rebuilds from scratch.
set -euo pipefail

CONFIG="${CONFIG:-configs/sbol.yaml}"
RAW_FAQ="${RAW_FAQ:-data/raw/sbol/faq_index_28_apr.json}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/sbol}"
INDEX_DIR="${INDEX_DIR:-indexes/sbol}"

echo "[1/2] parse SBOL FAQ"
python -m indexing.parse_sbol \
    --raw "${RAW_FAQ}" \
    --out-dir "${PROCESSED_DIR}"

echo "[2/2] build FAISS index"
python -m indexing.build_index \
    --config "${CONFIG}" \
    --corpus "${PROCESSED_DIR}/corpus.jsonl" \
    --out-dir "${INDEX_DIR}"

echo "done."
