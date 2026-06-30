#!/usr/bin/env bash
# Re-encode all four source indexes with the embedder defined in CONFIG
# (configs/unioned.yaml -> Qwen/Qwen3-Embedding-4B) and write them under
# indexes/qwen3/. Each source is re-encoded from its EXISTING metadata.jsonl,
# which carries the same {doc_id,title,text,file_name?,index?} schema as
# corpus.jsonl, so no raw source data is needed (handy for rnd).
#
#   GPU=0 CONFIG=configs/unioned.yaml bash scripts/build_qwen_indexes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-configs/unioned.yaml}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-indexes/qwen3}"

# source_key:source_metadata_dir:out_dir
SOURCES=(
  "musique:indexes/musique:${OUT_ROOT}/musique"
  "sbol:indexes/sbol:${OUT_ROOT}/sbol"
  "rnd:indexes/rnd_index:${OUT_ROOT}/rnd_index"
  "ckr:indexes/ckr_index:${OUT_ROOT}/ckr_index"
)

cd "${ROOT}"
for entry in "${SOURCES[@]}"; do
  IFS=":" read -r key src out <<<"${entry}"
  echo "=== rebuilding ${key}: ${src}/metadata.jsonl -> ${out} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m indexing.build_index \
    --config "${CONFIG}" \
    --corpus "${src}/metadata.jsonl" \
    --out-dir "${out}"
done

echo "done. New indexes under ${OUT_ROOT}/"
