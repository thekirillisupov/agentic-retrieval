# Expert pruning of the MoE actor

Structured pruning of the Qwen3.5-35B-A3B MoE: drop the experts the RL-tuned
policy never (or barely) routes to, shrinking the checkpoint and the expert
weight traffic without touching the layers that do the work. The **validation
domain is the train parquet** (`grpo_train.parquet`) — routing statistics and
the held-out scoring slice both come from it, so expert importance is measured
on exactly the agentic-retrieval distribution the model serves.

The default candidate is the quantized build
`checkpoints/quantized/qwen3_5_35b_a3b_w8a8`. Pruning composes cleanly with it:

- routing is decided by `mlp.gate` / `shared_expert_gate`, which the quantizer
  leaves in bf16 (`quantize/quantize.py:IGNORE`), so router statistics collected
  on the W8A8 candidate are exact, not a bf16 proxy;
- the pruner works at the safetensors-name level, so per-expert
  compressed-tensors artifacts (`weight`, `weight_scale`, ...) travel with
  their expert automatically, and the same code prunes the bf16 actor
  (`model.layers.*`) and the vLLM-wrapped W8A8 checkpoint
  (`model.language_model.layers.*`, `text_config`-nested config).

## Pipeline

```
[1] prune.collect_stats   forward passes over the STATS slice of grpo_train.parquet,
                          hooks on every *.mlp.gate router; per layer/expert:
                          top-k selection count + routed probability mass
        │                 (renormalised iff norm_topk_prob, i.e. the weight the
        │                  expert actually gets in the forward pass)
        ▼   router_stats.json
[2] prune.select          uniform keep-count per layer (the config has a single
                          num_experts, per-layer counts don't fit the format);
        │                 --keep N or --coverage p (max over layers of the
        │                  per-layer minimum); floor = router top_k
        ▼   prune_plan.json
[3] prune.apply           shard-by-shard safetensors rewrite on CPU: drop pruned
                          experts, renumber kept ones contiguously, slice router
        │                 gate rows, update num_experts in config.json
        ▼   pruned checkpoint
[4] prune.validate        teacher-forced NLL on the HELD-OUT tail slice of the
                          same parquet (never seen by [1]); candidates ranked
                          against the unpruned baseline
```

One command:

```bash
# inference/quant venv (llmcompressor ships compressed-tensors, needed to load W8A8)
bash scripts/prune_experts.sh                    # COVERAGE=0.99 default
KEEP=64 bash scripts/prune_experts.sh            # fixed keep per layer
COVERAGE=0.995 NUM_SAMPLES=1024 bash scripts/prune_experts.sh

# knobs (env): MODEL TRAIN_PARQUET WORKDIR KEEP|COVERAGE METRIC
#              NUM_SAMPLES VAL_SAMPLES MAX_SEQ_LEN SEED PROMPT_VERSION
```

Outputs land in `checkpoints/pruned/<candidate>/`:
`router_stats.json`, `prune_plan.json`, `model/` (the pruned checkpoint,
with an `expert_pruning.json` provenance stamp), `validation.json`.

## Train parquet as the validation domain

Rows are shuffled once with `--seed`; the stats pass reads the head
(`--num-samples`), validation reads the reserved tail (`--val-samples`).
`collect_stats` takes `--val-samples` precisely so the reservation is fixed at
stats time — `validate` must be called with the **same**
`--num-samples/--val-samples/--seed` or it refuses to make an honest split
(overlap raises). Prompts are rendered exactly like quantization calibration
(`quantize/calibration.py`): chat template + per-row `prompt_version` tool
schemas + `add_generation_prompt`, `enable_thinking=False`.

## Comparing several candidates

`prune.validate` scores any number of checkpoints in one run (loaded
sequentially, first one is the baseline for deltas):

```bash
python -m prune.validate \
  --train-parquet data/processed/unioned/grpo_train.parquet \
  --models base=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
           keep96=checkpoints/pruned/qwen3_5_35b_a3b_w8a8_k96/model \
           keep64=checkpoints/pruned/qwen3_5_35b_a3b_w8a8_k64/model \
  --output checkpoints/pruned/sweep_validation.json
```

To sweep retention levels, rerun `prune.select` + `prune.apply` at each
`--keep` (stage [1] is the expensive one and is reused — stats do not depend
on the keep level).

## Accept gate

On-domain NLL is a cheap smoke test that catches a broken prune; it is **not**
the accept criterion. Gate on the agent eval, same as the quantization A/B
(`quantize/README.md`):

```bash
MODEL=checkpoints/pruned/qwen3_5_35b_a3b_w8a8/model bash scripts/serve_vllm_int8.sh
bash scripts/run_eval.sh     # compare ndcg@10 / recall@10 to the unpruned candidate
```

A pruned checkpoint is a drop-in for vLLM: `num_experts` is updated in the
config, experts are contiguous, and the quantization_config regexes are
index-agnostic.

## Caveats

- **Uniform keep only.** HF/vLLM configs carry one `num_experts`; a layer that
  needs more experts than others drags the whole model up via `--coverage`.
- **Dead-slot warning.** `prune.select` warns when kept slots never fired
  during the stats pass — raise `--num-samples` (or use trajectory-style
  multi-turn stats data) before trusting the plan.
- **Domain lock-in.** Importance is measured on the train parquet only; experts
  that serve out-of-domain traffic will be pruned. That is the point of using
  the train distribution as the validation domain — but don't ship the pruned
  build to traffic that looks nothing like it.
- **Prune-then-quantize vs quantize-then-prune.** This pipeline prunes the
  already-quantized candidate (no requantization needed, GPTQ calibration
  stays valid because surviving expert weights are untouched). If you instead
  prune the bf16 actor, requantize afterwards so GPTQ calibrates on the pruned
  routing.
- `tests/test_expert_pruning.py` pins the selection/remap/config logic and runs
  without torch.
