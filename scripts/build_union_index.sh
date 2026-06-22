#!/usr/bin/env bash
# Merge indexes/sbol + indexes/musique + indexes/rnd_index -> indexes/unioned/
# Idempotent: overwrites indexes/unioned/ if it already exists.
set -euo pipefail

OUT_DIR="${OUT_DIR:-indexes/unioned}"
INDEX_SBOL="${INDEX_SBOL:-indexes/sbol}"
INDEX_MUSIQUE="${INDEX_MUSIQUE:-indexes/musique}"
INDEX_RND="${INDEX_RND:-indexes/rnd_index}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

"${PYTHON}" -m indexing.merge_indexes \
    --index-dir "${INDEX_SBOL}" \
    --index-dir "${INDEX_MUSIQUE}" \
    --index-dir "${INDEX_RND}" \
    --out-dir "${OUT_DIR}"

echo "done. See ${OUT_DIR}/stats.json"
