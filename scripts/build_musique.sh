#!/usr/bin/env bash
# Full data pipeline: parse raw MuSiQue → build faiss index → build eval dataset.
# Idempotent: re-running is safe but rebuilds from scratch.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RAW_DIR="${RAW_DIR:-data/raw/musique}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/musique}"

echo "[1/3] parse MuSiQue"
python -m indexing.parse_musique --raw-dir "${RAW_DIR}" --out-dir "${PROCESSED_DIR}"

echo "[2/3] build faiss index"
python -m indexing.build_index --config "${CONFIG}" \
    --corpus "${PROCESSED_DIR}/corpus.jsonl"

echo "[3/3] build eval dataset"
python -m eval_.build_eval \
    --raw-dir "${RAW_DIR}" \
    --processed-dir "${PROCESSED_DIR}" \
    --out "${PROCESSED_DIR}/musique_dev_eval.jsonl"

echo "done."
