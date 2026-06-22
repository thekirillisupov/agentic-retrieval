#!/usr/bin/env bash
# Union the per-source GRPO parquets (data/processed/<src>/grpo_{train,val}.parquet)
# into data/processed/unioned/grpo_{train,val}.parquet.
set -euo pipefail

OUT_DIR="${OUT_DIR:-data/processed/unioned}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
SOURCES="${SOURCES:-musique sbol rnd}"
PROMPT_VERSION="${PROMPT_VERSION:-v2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
"${PYTHON}" -m grpo.build_union_data \
    --sources ${SOURCES} \
    --processed-dir "${PROCESSED_DIR}" \
    --out-dir "${OUT_DIR}" \
    --prompt-version "${PROMPT_VERSION}"

echo "done. See ${OUT_DIR}/stats.json"
