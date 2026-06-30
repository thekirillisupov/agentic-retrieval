"""INT8 quantization of the agentic-retrieval actor for A100. Two schemes (--scheme):

  W8A16 (default, safe): int8 weights + bf16 activations.
  W8A8 (--scheme W8A8): int8 weights + int8 per-token-dynamic activations.

Qwen3.5-35B-A3B is a Qwen3-Next-family hybrid: GatedDeltaNet linear-attention
(SSM) layers interleaved with gated attention, plus a 3B-active MoE.

  * W8A16 touches no activations, so the SmoothQuant norm-corruption failure
    mode on this arch (llm-compressor#2059) cannot occur. ~2x weight-memory
    savings (35B int8 weights ~= 35 GB, fits one 80 GB A100, TP=1) and a
    decode-time bandwidth speedup, bf16 activations preserving the RL policy.
  * W8A8 additionally quantizes activations to int8, using A100's int8 tensor
    cores for a compute (prefill/throughput) speedup. We default SmoothQuant OFF
    so #2059 cannot trigger (norms are never quantized — only Linear layers are,
    and per-token-dynamic activation quant needs no calibration stats). Opt into
    SmoothQuant with --smoothquant to A/B test accuracy, accepting the #2059 risk.

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
    --scheme W8A16 \
    --prompt-version v2_search_only --num-samples 512
  # W8A8 variant: --scheme W8A8 --output .../qwen3_5_35b_a3b_w8a8

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
    ap.add_argument(
        "--scheme",
        default="W8A16",
        choices=["W8A16", "W8A8"],
        help="W8A16 = int8 weights, bf16 activations (safe default). "
        "W8A8 = int8 weights + int8 per-token-dynamic activations; uses A100's "
        "int8 tensor cores for a compute speedup, at higher accuracy risk on this "
        "GatedDeltaNet hybrid (see --smoothquant and llm-compressor#2059).",
    )
    ap.add_argument(
        "--smoothquant",
        action="store_true",
        help="W8A8 only: prepend SmoothQuant to migrate activation outliers into "
        "weights. Can improve W8A8 accuracy, BUT corrupts GatedDeltaNet norm params "
        "on Qwen3-Next (llm-compressor#2059). OFF by default; enable to A/B test.",
    )
    ap.add_argument("--smoothquant-strength", type=float, default=0.8)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rtn",
        action="store_true",
        help="Skip GPTQ; use plain round-to-nearest weight quant. For W8A8 the "
        "activations are still int8 per-token-dynamic (no calibration needed). "
        "Fastest and zero-risk on weights; slightly lower accuracy than GPTQ.",
    )
    ap.add_argument(
        "--dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian dampening; raise (e.g. 0.05) if you hit numerical issues.",
    )
    args = ap.parse_args()

    if args.smoothquant and args.scheme != "W8A8":
        ap.error("--smoothquant only applies to --scheme W8A8")
    if args.smoothquant and args.rtn:
        ap.error("--smoothquant needs GPTQ calibration; drop --rtn")

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

    print(f"[quantize] scheme={args.scheme} gptq={not args.rtn} "
          f"smoothquant={args.smoothquant}")

    # W8A16 RTN and W8A8-dynamic RTN both need no calibration forward passes
    # (W8A8 activations are quantized per-token at runtime, not from calib stats).
    if args.rtn:
        recipe = QuantizationModifier(
            targets="Linear", scheme=args.scheme, ignore=IGNORE
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
        modifiers: list = []
        if args.smoothquant:
            # Opt-in only: see #2059 warning on --smoothquant. Shares the ignore
            # list so it never folds scales into the GatedDeltaNet block.
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

            modifiers.append(
                SmoothQuantModifier(
                    smoothing_strength=args.smoothquant_strength, ignore=IGNORE
                )
            )
        modifiers.append(
            GPTQModifier(
                targets="Linear",
                scheme=args.scheme,
                ignore=IGNORE,
                dampening_frac=args.dampening_frac,
            )
        )
        oneshot(
            model=model,
            dataset=ds,
            recipe=modifiers if len(modifiers) > 1 else modifiers[0],
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
