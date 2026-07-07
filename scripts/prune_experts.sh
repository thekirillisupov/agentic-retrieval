#!/usr/bin/env bash
# End-to-end expert pruning of the MoE candidate, validated on the train
# parquet (stats on the head of the shuffled rows, NLL scoring on the reserved
# tail — see prune/README.md).
#
# Usage (defaults target the W8A8 candidate):
#   bash scripts/prune_experts.sh                       # coverage-driven keep
#   KEEP=64 bash scripts/prune_experts.sh               # fixed keep per layer
#   COVERAGE=0.995 bash scripts/prune_experts.sh
#
#   MODEL=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
#   TRAIN_PARQUET=data/processed/unioned/grpo_train.parquet \
#   WORKDIR=checkpoints/pruned/qwen3_5_35b_a3b_w8a8 \
#   bash scripts/prune_experts.sh
#
# Preferred data source: recorded multi-turn rollouts (tool calls + tool
# results — activates the experts single-turn prompts never touch). Setting
# TRAJECTORIES switches BOTH the stats pass and validation to the JSONL:
#   TRAJECTORIES=trajectories_data/gspo_qwen3_moe/65.jsonl \
#   MAX_SEQ_LEN=8192 bash scripts/prune_experts.sh
#
# Run in the inference/quant venv (llmcompressor pulls in compressed-tensors,
# needed to load the W8A8 candidate) — NOT the Megatron training env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-checkpoints/quantized/qwen3_5_35b_a3b_w8a8}"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/processed/unioned/grpo_train.parquet}"
TRAJECTORIES="${TRAJECTORIES:-}"    # trajectory JSONL; overrides TRAIN_PARQUET
WORKDIR="${WORKDIR:-checkpoints/pruned/$(basename "${MODEL}")}"

KEEP="${KEEP:-}"                 # fixed experts kept per layer; overrides COVERAGE
COVERAGE="${COVERAGE:-0.99}"     # min router-mass every layer must retain
METRIC="${METRIC:-mass}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"   # stats slice (head of shuffled parquet)
VAL_SAMPLES="${VAL_SAMPLES:-128}"   # held-out slice (tail), reserved from stats
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
SEED="${SEED:-0}"
PROMPT_VERSION="${PROMPT_VERSION:-v2_search_only}"

STATS="${WORKDIR}/router_stats.json"
PLAN="${WORKDIR}/prune_plan.json"
PRUNED="${WORKDIR}/model"
REPORT="${WORKDIR}/validation.json"

if [[ -n "${KEEP}" ]]; then
  SELECT_ARGS=(--keep "${KEEP}")
else
  SELECT_ARGS=(--coverage "${COVERAGE}")
fi

# The stats pass and validation MUST read the same source (and the same
# NUM_SAMPLES/VAL_SAMPLES/SEED) to keep their slices disjoint.
if [[ -n "${TRAJECTORIES}" ]]; then
  DATA_ARGS=(--trajectories "${TRAJECTORIES}")
  DATA_DESC="${TRAJECTORIES} (full rollouts incl. tool results)"
else
  DATA_ARGS=(--train-parquet "${TRAIN_PARQUET}")
  DATA_DESC="${TRAIN_PARQUET} (single-turn prompts)"
fi

cd "${REPO_ROOT}"
mkdir -p "${WORKDIR}"

echo "== [1/4] router stats on ${DATA_DESC} =="
python -m prune.collect_stats \
  --model "${MODEL}" \
  "${DATA_ARGS[@]}" \
  --output "${STATS}" \
  --num-samples "${NUM_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --seed "${SEED}" \
  --prompt-version "${PROMPT_VERSION}"

echo "== [2/4] expert selection =="
python -m prune.select \
  --stats "${STATS}" \
  --output "${PLAN}" \
  --metric "${METRIC}" \
  "${SELECT_ARGS[@]}"

echo "== [3/4] prune checkpoint =="
python -m prune.apply \
  --model "${MODEL}" \
  --plan "${PLAN}" \
  --output "${PRUNED}"

echo "== [4/4] validate on held-out slice of ${DATA_DESC} =="
python -m prune.validate \
  "${DATA_ARGS[@]}" \
  --models "base=${MODEL}" "pruned=${PRUNED}" \
  --output "${REPORT}" \
  --num-samples "${NUM_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --seed "${SEED}" \
  --prompt-version "${PROMPT_VERSION}"

echo
echo "Pruned checkpoint: ${PRUNED}"
echo "Accept gate: MODEL=${PRUNED} bash scripts/serve_vllm_int8.sh && bash scripts/run_eval.sh"
