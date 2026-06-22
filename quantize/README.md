# Quantizing the actor for A100 inference (W8A16 INT8)

Post-training, weight-only INT8 quantization of the merged Qwen3.5-35B-A3B actor,
for serving on a single A100 with vLLM.

## Why this scheme

| | A100 fit |
|---|---|
| **W8A16 (weight-only INT8)** ✅ chosen | int8 weights + bf16 activations. ~2x weight-memory savings (~35 GB, fits one 80 GB A100), decode speedup from halved weight bandwidth, activations untouched so the RL-tuned policy is preserved. |
| W8A8 (INT8 weights + activations) | Would use A100's int8 tensor cores for a bigger compute win, **but** needs SmoothQuant, which corrupts GatedDeltaNet norm params on this hybrid arch ([llm-compressor#2059](https://github.com/vllm-project/llm-compressor/issues/2059)). Higher throughput, real accuracy risk. |
| FP8-dynamic | The validated Qwen3-Next recipe, but A100 (Ampere) has **no FP8 tensor cores** — memory savings only, no compute speedup. |

The model is a **Qwen3-Next-family hybrid**: GatedDeltaNet linear-attention (SSM)
layers interleaved with gated attention, plus a 3B-active MoE. The fragile,
non-Linear pieces are left in bf16; the MoE experts (the bulk of the weights) are
quantized. Ignore list (`quantize/quantize.py:IGNORE`):

- `lm_head`
- `re:.*mlp.gate$` — MoE router
- `re:.*mlp.shared_expert_gate$` — shared-expert gate
- `re:.*linear_attn.*` — GatedDeltaNet/SSM block (conv1d, in_proj, out_proj, A_log, dt_bias)

## Calibration data

GPTQ minimises weight error on the calibration distribution, so calibration
prompts must look like real agentic-retrieval inputs, not bare questions.
`quantize/calibration.py` renders each generated question through the **same**
path the harness uses (`agent/prompts.py`): versioned system prompt +
`<client>`-wrapped user turn + the tool schemas for that prompt version,
tokenized with the model chat template (`add_generation_prompt=True`,
`enable_thinking=False`).

Provide your generated questions in any of:
- `.jsonl` — one object per line with a `question` field (or a bare string per line)
- `.json` — a list of strings or objects with `question`
- `.txt` — one question per line
- `.parquet` — a `question` column

Pick the prompt version that matches the corpus you'll serve: `v2_search_only`
(search-only: musique/sbol/rnd) or `v2` (ckr, adds `get_neighbours`). To cover
both, quantize twice or concatenate questions and run once per version on a split.

> The chosen questions are single-turn (system + query). That covers the
> system-prompt + tool-schema + query distribution, which is what weight-only
> quant needs. For full fidelity including tool-result turns, build calibration
> from recorded `trajectories_data/` rollouts instead (not wired here — ask if
> you want it).

## Run

In an inference/quant env (NOT the Megatron training env) with
`llmcompressor>=0.9.0`, `transformers`, `torch`:

```bash
# 1. Merge the Megatron actor to HF first (if not already done)
bash scripts/dist_ckpt_to_hf.sh --actor-dir checkpoints/gspo_qwen3_moe/global_step_40/actor ...

# 2. Quantize (weight-only INT8 + GPTQ, calibrated)
MODEL=checkpoints/gspo_qwen3_moe/global_step_40/actor/huggingface \
QUESTIONS=data/calib/questions.jsonl \
OUTPUT=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 \
PROMPT_VERSION=v2_search_only NUM_SAMPLES=512 \
bash scripts/quantize_w8a16.sh

# 2b. Or zero-risk, no-calibration round-to-nearest:
#   bash scripts/quantize_w8a16.sh --rtn   (QUESTIONS still required by the wrapper; ignored)

# 3. Serve on A100 (TP=1, vLLM auto-detects the quant)
MODEL=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 bash scripts/serve_vllm_int8.sh
```

## Validate after quantizing

The quant is only good if retrieval quality holds. Re-run the agent eval against
the quantized endpoint and compare to the bf16 baseline (README: ndcg@10≈0.4750,
recall@10≈0.4275, avg_calls≈2.88 on n=200, thinking off):

```bash
bash scripts/run_eval.sh   # points the agent at the vLLM endpoint above
```

A small ndcg drop (≲1 pt) is expected and acceptable; a large drop usually means
an expert was under-calibrated — raise `NUM_SAMPLES`, or move a layer into the
ignore list.

## Caveats / things to check on your stack

- **Memory during quantization**: GPTQ loads the model in bf16 (~70 GB) and
  onloads layers sequentially. `device_map="auto"` spreads it; use ≥1 80 GB GPU,
  or 2 to be comfortable. RTN (`--rtn`) is much lighter.
- **vLLM version**: confirm your vLLM build serves compressed-tensors W8A16 for
  the Qwen3-Next hybrid arch (GatedDeltaNet). The repo's pinned vLLM is 0.18.0
  (`install_cu129py312.sh`); if it predates hybrid+compressed-tensors support,
  bump vLLM for serving only.
- **transformers/llmcompressor compatibility**: needs `llmcompressor>=0.9.0`
  (Qwen3-Next calibration support). Keep this in a separate env from training.
