# Quantizing the actor for A100 inference (INT8)

Post-training INT8 quantization of the merged Qwen3.5-35B-A3B actor, for serving
on a single A100 with vLLM. One script (`quantize/quantize.py`), two schemes via
`--scheme`, so you can build both and A/B test them.

## Schemes

| `--scheme` | What | A100 fit |
|---|---|---|
| **W8A16** (default, safe) | int8 weights, **bf16 activations** | ~2x weight-memory savings (~35 GB, fits one 80 GB A100), decode speedup from halved weight bandwidth, activations untouched so the RL-tuned policy is preserved. No #2059 exposure. |
| **W8A8** (test) | int8 weights + **int8 per-token-dynamic activations** | Uses A100's int8 tensor cores for a prefill/throughput **compute** speedup. SmoothQuant OFF by default (so [#2059](https://github.com/vllm-project/llm-compressor/issues/2059) can't trigger — only Linear layers are quantized, norms are never touched, and dynamic activation quant needs no calib stats). Opt into `--smoothquant` to chase accuracy, accepting the #2059 risk. |
| FP8-dynamic (not built) | int8-ish fp8 | The validated Qwen3-Next recipe, but A100 (Ampere) has **no FP8 tensor cores** — memory savings only, no compute speedup. |

Both schemes share the ignore list and the calibration builder below; W8A8 just
also quantizes activations.

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

# 2a. W8A16 (weight-only, safe)
MODEL=checkpoints/gspo_qwen3_moe/global_step_40/actor/huggingface \
QUESTIONS=data/calib/questions.jsonl \
OUTPUT=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 \
PROMPT_VERSION=v2_search_only NUM_SAMPLES=512 \
bash scripts/quantize_w8a16.sh

# 2b. W8A8 (int8 activations too — the A100 compute path)
MODEL=checkpoints/gspo_qwen3_moe/global_step_40/actor/huggingface \
QUESTIONS=data/calib/questions.jsonl \
OUTPUT=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
PROMPT_VERSION=v2_search_only NUM_SAMPLES=512 \
bash scripts/quantize_w8a8.sh
#   SMOOTHQUANT=1 bash scripts/quantize_w8a8.sh   # opt-in, see #2059 risk
#   --rtn on either wrapper skips GPTQ (W8A8 keeps dynamic int8 activations).

# 3. Serve on A100 (TP=1, vLLM auto-detects the quant from the checkpoint)
MODEL=checkpoints/quantized/qwen3_5_35b_a3b_w8a16 bash scripts/serve_vllm_int8.sh
MODEL=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 PORT=8001 bash scripts/serve_vllm_int8.sh
```

## Validate / A-B test the two schemes

Quality only matters if retrieval holds. Serve each build (on different ports
above) and run the agent eval against each, comparing to the bf16 baseline
(README: ndcg@10≈0.4750, recall@10≈0.4275, avg_calls≈2.88 on n=200, thinking off):

```bash
# point the eval at each endpoint (set whatever your run_eval.sh reads for the URL)
VLLM_BASE_URL=http://localhost:8000/v1 bash scripts/run_eval.sh   # W8A16
VLLM_BASE_URL=http://localhost:8001/v1 bash scripts/run_eval.sh   # W8A8
```

Expected ranking: **W8A16 ≥ W8A8** on ndcg (W8A8 trades a little accuracy for the
int8-compute throughput win). A small drop (≲1 pt) is fine. If W8A8 drops hard,
try `SMOOTHQUANT=1`, raise `NUM_SAMPLES`, or fall back to W8A16. Compare
throughput with vLLM's `/metrics` or a fixed-load `benchmark_serving` run to see
whether W8A8's compute speedup is worth the accuracy cost on your traffic.

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
