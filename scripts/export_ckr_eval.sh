#!/usr/bin/env bash
# Export CKR val parquets to eval JSONL for run_eval.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

cd "${ROOT}"

for name in ckr ckr_per_doc_filtered; do
  echo "=== ${name} ==="
  "${PYTHON}" -m eval_.export_grpo_val \
    --parquet "data/processed/${name}/grpo_val.parquet" \
    --out "data/processed/${name}/eval.jsonl"
done

echo "done."
