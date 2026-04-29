#!/usr/bin/env bash
# Build the GRPO parquet from MuSiQue train.
#   1. parse_musique          -> data/processed/musique/{corpus.jsonl, source_to_doc_id.json}
#   2. eval_.build_train      -> data/processed/musique/musique_train.jsonl  (with gold ids)
#   3. grpo.data_prep         -> data/processed/musique/grpo_{train,val}.parquet
#
# Step (1) only needs to run once and is shared with the eval pipeline. If you
# ran scripts/build_all.sh before, you can skip it here — set SKIP_PARSE=1.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RAW_DIR="${RAW_DIR:-data/raw/musique}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/musique}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
VAL_SIZE="${VAL_SIZE:-256}"
LIMIT="${LIMIT:-}"
SKIP_PARSE="${SKIP_PARSE:-0}"

if [[ "${SKIP_PARSE}" != "1" ]]; then
    echo "[1/3] parse MuSiQue (train + dev)"
    python -m indexing.parse_musique --raw-dir "${RAW_DIR}" --out-dir "${PROCESSED_DIR}"
fi

echo "[2/3] build train jsonl (with gold doc_ids)"
python -m eval_.build_train \
    --raw-dir "${RAW_DIR}" \
    --processed-dir "${PROCESSED_DIR}" \
    --out "${PROCESSED_DIR}/musique_train.jsonl"

echo "[3/3] convert to veRL parquet"
PY_ARGS=(
    --in "${PROCESSED_DIR}/musique_train.jsonl"
    --out-train "${PROCESSED_DIR}/grpo_train.parquet"
    --out-val "${PROCESSED_DIR}/grpo_val.parquet"
    --prompt-version "${PROMPT_VERSION}"
    --val-size "${VAL_SIZE}"
)
if [[ -n "${LIMIT}" ]]; then
    PY_ARGS+=(--limit "${LIMIT}")
fi
python -m grpo.data_prep "${PY_ARGS[@]}"

echo "done."
