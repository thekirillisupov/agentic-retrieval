#!/usr/bin/env bash
# Evaluate both CKR eval sets against the /retrieve HTTP endpoint.
#
# Prereq: agent service listening (e.g. port 8090), same as:
#   curl -s -X POST http://127.0.0.1:8090/retrieve \
#     -H 'content-type: application/json' \
#     -d '{"messages":[{"role":"user","content":"..."}],
#          "search_params":{"configuration":...},"prompt_version":"v2_search_only",
#          "include_trajectory":true}'
#
# Usage:
#   bash scripts/run_ckr_eval.sh
#   URL=http://127.0.0.1:8090 PROMPT_VERSION=v2_search_only bash scripts/run_ckr_eval.sh
#   SUBSET_SIZE=5 bash scripts/run_ckr_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

URL="${URL:-http://127.0.0.1:8090}"
PROMPT_VERSION="${PROMPT_VERSION:-v2_search_only}"
OUT_DIR="${OUT_DIR:-data/processed/ckr_eval}"
SEARCH_PARAMS="${SEARCH_PARAMS:-}"
if [[ -z "${SEARCH_PARAMS}" ]]; then
  SEARCH_PARAMS="$("${PYTHON}" -c "import json; from pathlib import Path; print(json.dumps({'configuration': json.loads(Path('configs/search_config.v.0.2.4.json').read_text())}))")"
fi

ARGS=(
  --all
  --url "${URL}"
  --prompt-version "${PROMPT_VERSION}"
  --search-params "${SEARCH_PARAMS}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${SUBSET_SIZE:-}" ]]; then
  ARGS+=(--subset-size "${SUBSET_SIZE}")
fi

cd "${ROOT}"
"${PYTHON}" -m eval_.run_ckr_eval "${ARGS[@]}" "$@"
