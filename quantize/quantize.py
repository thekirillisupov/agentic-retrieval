"""Weight-only INT8 (W8A16) quantization of the agentic-retrieval actor for A100.

Why W8A16 (and not W8A8) for *this* model:
  * Qwen3.5-35B-A3B is a Qwen3-Next-family hybrid: GatedDeltaNet linear-attention
    (SSM) layers interleaved with gated attention, plus a 3B-active MoE. Pure
    W8A8-INT8 needs SmoothQuant to tame activation outliers, and SmoothQuant
    corrupts the GatedDeltaNet norm params on this architecture (open llm-compressor
    bug vllm-project/llm-compressor#2059). Weight-only quant does NOT touch
    activations, so that failure mode cannot occur.
  * On A100 weight-only INT8 still gives ~2x weight-memory savings (35B int8
    weights ~= 35 GB, fits one 80 GB A100, TP=1) and a decode-time speedup from
    halved weight-memory bandwidth, with bf16 activations preserving accuracy of
    the RL-tuned policy.

Ignore list (kept identical in spirit to the official Qwen3-Next recipe):
  * lm_head                          — output head, always left in high precision.
  * re:.*mlp.gate$                   — MoE router; tiny and routing-critical, must
                                       stay full precision or expert selection drifts.
  * re:.*mlp.shared_expert_gate$     — shared-expert gate, same reasoning.
  * re:.*linear_attn.*               — GatedDeltaNet / SSM block (conv1d, in_proj,
                                       out_proj, A_log, dt_bias): fragile, not a
                                       standard Linear matmul, leave in bf16.

The MoE *experts* themselves ARE quantized (that is where the weight budget is).
GPTQ needs tokens to route through each expert during calibration, so use a
healthy num_calibration_samples (default 512); raise it if you see experts with
poor reconstruction.

Run (from repo root, in an inference/quant env — NOT the Megatron training env):
  python -m quantize.quantize \
    --model checkpoints/gspo_qwen3_moe/global_step_N/actor/huggingface \
    --questions data/calib/questions.jsonl \
    --output checkpoints/quantized/qwen3_5_35b_a3b_w8a16 \
    --prompt-version v2_search_only --num-samples 512

Requires: llmcompressor>=0.9.0 (Qwen3-Next support), transformers, torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# GatedDeltaNet (SSM) block, MoE router gates, and the LM head are left in bf16.
IGNORE = [
    "lm_head",
    "re:.*mlp.gate$",
    "re:.*mlp.shared_expert_gate$",
    "re:.*linear_attn.*",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="HF checkpoint dir (merged actor, e.g. .../actor/huggingface) or hub id.",
    )
    ap.add_argument(
        "--questions",
        required=True,
        help="Generated calibration questions (.jsonl/.json/.txt/.parquet).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Where to write the compressed-tensors checkpoint.",
    )
    ap.add_argument("--question-field", default="question")
    ap.add_argument(
        "--prompt-version",
        default="v2_search_only",
        help="Which agent prompt profile to render calibration prompts with "
        "(v2_search_only for single-source rows, v2 for ckr/get_neighbours rows).",
    )
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rtn",
        action="store_true",
        help="Skip GPTQ; use plain round-to-nearest weight quant (no calibration "
        "forward passes). Fastest and zero-risk, slightly lower accuracy than GPTQ.",
    )
    ap.add_argument(
        "--dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian dampening; raise (e.g. 0.05) if you hit numerical issues.",
    )
    args = ap.parse_args()

    # Imported lazily so `--help` works without a GPU / heavy deps installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    from quantize.calibration import build_calibration_dataset, load_questions

    print(f"[quantize] loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.rtn:
        # Weight-only RTN: no calibration data needed.
        recipe = QuantizationModifier(
            targets="Linear", scheme="W8A16", ignore=IGNORE
        )
        oneshot(model=model, recipe=recipe)
    else:
        questions = load_questions(args.questions, field=args.question_field)
        print(f"[quantize] loaded {len(questions)} calibration questions")
        ds = build_calibration_dataset(
            questions,
            tokenizer,
            prompt_version=args.prompt_version,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
        )
        recipe = GPTQModifier(
            targets="Linear",
            scheme="W8A16",
            ignore=IGNORE,
            dampening_frac=args.dampening_frac,
        )
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.num_samples,
        )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[quantize] saving compressed checkpoint -> {out}")
    model.save_pretrained(str(out), save_compressed=True)
    tokenizer.save_pretrained(str(out))
    print("[quantize] done. Serve with scripts/serve_vllm_int8.sh")


if __name__ == "__main__":
    main()
