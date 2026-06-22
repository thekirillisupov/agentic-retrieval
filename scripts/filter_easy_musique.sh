#!/usr/bin/env bash
# Remove "easy" musique questions — those where every gold_doc_id already
# surfaces in the top-k semantic search results.  Requires the tool server
# to be running (scripts/serve_tool.sh).
#
# After this script completes, re-run scripts/build_union_grpo.sh to rebuild
# the unioned train/val parquets with the trimmed musique data.
#
# Options (env vars):
#   TOP_K           top-k threshold for "easy" (default 5)
#   SUBSAMPLE_FRAC  keep this fraction per hop group after easy filter, e.g. 0.5
#   SEED            random seed for subsampling (default 0)
#   TRAIN           path to musique train parquet
#   VAL             path to musique val parquet
#   OUT_DIR         output directory (default: same as TRAIN, i.e. in-place)
#   SPLIT           train | val | both  (default both)
#   TOOL_SERVER     base URL of the tool server (default http://localhost:8100)
#   SOURCE          index source name (default musique)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

TOP_K="${TOP_K:-5}"
SUBSAMPLE_FRAC="${SUBSAMPLE_FRAC:-}"
SEED="${SEED:-0}"
TRAIN="${TRAIN:-data/processed/musique/grpo_train.parquet}"
VAL="${VAL:-data/processed/musique/grpo_val.parquet}"
SPLIT="${SPLIT:-both}"
TOOL_SERVER="${TOOL_SERVER:-http://localhost:8100}"
SOURCE="${SOURCE:-musique}"

CMD=(
    "${PYTHON}" -m grpo.filter_easy_musique
    --train "${TRAIN}"
    --val   "${VAL}"
    --top-k "${TOP_K}"
    --tool-server "${TOOL_SERVER}"
    --source "${SOURCE}"
    --split "${SPLIT}"
    --seed "${SEED}"
)

if [[ -n "${SUBSAMPLE_FRAC}" ]]; then
    CMD+=(--subsample-frac "${SUBSAMPLE_FRAC}")
fi

if [[ -n "${OUT_DIR:-}" ]]; then
    CMD+=(--out-dir "${OUT_DIR}")
fi

cd "${ROOT}"
echo "filtering musique (top_k=${TOP_K}, subsample_frac=${SUBSAMPLE_FRAC:-none}, split=${SPLIT}) ..."
"${CMD[@]}"
echo "done."
