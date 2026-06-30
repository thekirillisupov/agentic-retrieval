#!/usr/bin/env bash
# Serve Qwen3-Reranker-4B as a cross-encoder scoring server via vLLM.
#
# The FAISS retriever (scripts/serve_tool.sh) returns a candidate pool; the tool
# server then POSTs (query, [docs]) to this server's /v1/rerank endpoint and
# keeps the top_k by reranker score. See tool_server/reranker.py.
#
# GPU layout (8-GPU box): the tool-server embedder shards across GPUs 0-3
# (GPUS=0,1,2,3 bash scripts/serve_tool.sh), so the reranker takes the LAST 4
# GPUs with tensor-parallel=4:
#   GPUS=4,5,6,7 bash scripts/serve_reranker.sh
#
# Qwen3-Reranker is a generative model that scores relevance off the "yes"/"no"
# logits. vLLM cannot infer this from the HF config, so hf_overrides routes it
# to Qwen3ForSequenceClassification (efficient, /score + /v1/rerank compatible).
# `--runner pooling` selects the scoring runner (this vLLM build uses --runner/
# --convert, not the older --task). See
# https://docs.vllm.ai/en/latest/examples/offline_inference/qwen3_reranker.html
#
# Memory: this server shares GPUs 4-7 with the trainer, so the footprint must be
# tiny. The 4B weights sharded over TP=4 are ~2 GB/GPU — that's the hard floor.
# Reranking only scores short (query, doc) pairs, so it needs almost no KV cache;
# --kv-cache-memory-bytes caps it exactly (overriding --gpu-memory-utilization,
# which otherwise pre-allocates a huge KV cache: at 0.6 it grabbed ~48 GB/GPU).
# CUDA graphs are kept on (no --enforce-eager) for higher rerank throughput
# under the 64-worker rollout load; they cost ~0.3 GB capture + graph memory.
# Net per-GPU footprint: ~2 GB weights + KV_CACHE_BYTES + CUDA graphs + context.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Reranker-4B}"
PORT="${PORT:-8200}"
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"
# (query, doc) pairs are short; 2048 is ample and keeps the profiling pass small.
MAX_LEN="${MAX_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# Exact KV-cache size per GPU. 512 MiB is plenty for short scoring sequences;
# raise it if you batch many long passages. This is the main footprint knob.
KV_CACHE_BYTES="${KV_CACHE_BYTES:-512M}"
# Ceiling only (not the allocation target) — kv-cache-memory-bytes drives the
# real KV size; this just bounds the startup profiling pass. Keep it low so the
# colocated trainer keeps the rest of the card.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.15}"

CUDA_VISIBLE_DEVICES="${GPUS}" vllm serve "${MODEL}" \
  --runner pooling \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --port "${PORT}"
