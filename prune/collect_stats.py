"""Stage 1: collect per-layer expert routing statistics on the validation domain.

Hooks every MoE router (``*.mlp.gate`` Linear — bf16 even in the W8A8 build,
it is on the quantization ignore list) and accumulates, per layer and expert:

  count — how often the expert was in the router's top-k
  mass  — accumulated top-k router probability (renormalised iff the config
          sets norm_topk_prob, so mass reflects the weight the expert actually
          gets in the forward pass)

The candidate model IS the model being profiled — routing is measured on the
exact checkpoint that will be pruned (the W8A8 candidate), not on a bf16 proxy.

Two data sources (same convention as quantize/quantize.py):

  --train-parquet   grpo_train.parquet ``prompt`` column: single-turn
                    [system, user] prompts. NO agentic loop runs and no tool
                    results are seen — this covers the prompt-encoding
                    distribution only.
  --trajectories    trajectory JSONL (``messages_full``): recorded multi-turn
                    rollouts including tool calls AND tool results. Preferred
                    for pruning: it routes tokens through the experts that only
                    fire while the model processes search results, which the
                    parquet prompts never activate.

Run (same env as quantize/ — needs llmcompressor's compressed-tensors for the
W8A8 candidate):

  python -m prune.collect_stats \
    --model checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
    --trajectories trajectories_data/gspo_qwen3_moe/65.jsonl \
    --output checkpoints/pruned/qwen3_5_35b_a3b_w8a8/router_stats.json \
    --num-samples 512 --val-samples 128 --max-seq-len 8192
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.mlp\.gate$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF checkpoint dir (the candidate).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--train-parquet",
        help="grpo_train.parquet — single-turn prompts (no tool results seen).",
    )
    src.add_argument(
        "--trajectories",
        help="Trajectory JSONL with full multi-turn rollouts (messages_full), "
        "e.g. trajectories_data/gspo_qwen3_moe/65.jsonl. Includes tool results "
        "— the higher-fidelity source for expert pruning.",
    )
    ap.add_argument("--output", required=True, help="Where to write router_stats.json.")
    ap.add_argument("--num-samples", type=int, default=512,
                    help="Rows used for the stats pass (front of the shuffled pool).")
    ap.add_argument("--val-samples", type=int, default=128,
                    help="Rows RESERVED for prune.validate (tail of the shuffled "
                    "pool). Reserved here so the two stages cannot overlap.")
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Truncation length; raise (e.g. 8192) for --trajectories, "
                    "multi-turn rollouts are much longer than bare prompts.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-field", default="prompt",
                    help="Parquet column with message arrays (--train-parquet only).")
    ap.add_argument("--messages-field", default="messages_full",
                    help="JSONL key with the full rollout (--trajectories only).")
    ap.add_argument("--prompt-version", default="v2_search_only",
                    help="Fallback when a row carries no prompt_version.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from prune.common import load_model, load_split, moe_params

    print(f"[stats] loading model: {args.model}")
    model = load_model(args.model, dtype=args.dtype, device_map=args.device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    num_experts, top_k, norm_topk = moe_params(model.config)
    print(f"[stats] num_experts={num_experts} top_k={top_k} norm_topk_prob={norm_topk}")

    # --- hook the routers -------------------------------------------------
    counts: dict[int, torch.Tensor] = {}
    mass: dict[int, torch.Tensor] = {}
    tokens: dict[int, int] = {}

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            logits = output
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.detach().float().reshape(-1, num_experts)
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = probs.topk(top_k, dim=-1)
            if norm_topk:
                top_p = top_p / top_p.sum(dim=-1, keepdim=True)
            flat_i = top_i.reshape(-1)
            flat_p = top_p.reshape(-1)
            counts[layer] += torch.bincount(flat_i, minlength=num_experts).cpu()
            mass[layer] += torch.bincount(
                flat_i, weights=flat_p, minlength=num_experts
            ).double().cpu()
            tokens[layer] += logits.shape[0]

        return hook

    handles = []
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if not m:
            continue
        out_features = getattr(module, "out_features", None)
        if out_features is not None and out_features != num_experts:
            print(f"[stats] skip {name}: out_features={out_features} != {num_experts}")
            continue
        layer = int(m.group(1))
        counts[layer] = torch.zeros(num_experts, dtype=torch.long)
        mass[layer] = torch.zeros(num_experts, dtype=torch.double)
        tokens[layer] = 0
        handles.append(module.register_forward_hook(make_hook(layer)))
    if not handles:
        raise RuntimeError("no '*.mlp.gate' router modules found — wrong model?")
    print(f"[stats] hooked {len(handles)} router layers")

    # --- forward pass over the stats slice --------------------------------
    source = args.train_parquet or args.trajectories
    ds = load_split(
        source,
        tokenizer,
        split="stats",
        num_stats=args.num_samples,
        num_val=args.val_samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        prompt_field=args.prompt_field,
        messages_field=args.messages_field,
        default_prompt_version=args.prompt_version,
    )
    device = getattr(model, "device", "cpu")
    with torch.no_grad():
        for i, row in enumerate(ds):
            input_ids = torch.tensor([row["input_ids"]], device=device)
            model(input_ids=input_ids)
            if (i + 1) % 32 == 0:
                print(f"[stats] {i + 1}/{len(ds)} sequences")

    for h in handles:
        h.remove()

    # --- save --------------------------------------------------------------
    stats = {
        "model": args.model,
        "source": source,
        "source_kind": "trajectories" if args.trajectories else "train_parquet",
        "num_experts": num_experts,
        "top_k": top_k,
        "norm_topk_prob": norm_topk,
        "num_samples": len(ds),
        "val_samples_reserved": args.val_samples,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "layers": {
            str(layer): {
                "count": counts[layer].tolist(),
                "mass": mass[layer].tolist(),
                "tokens": tokens[layer],
            }
            for layer in sorted(counts)
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats) + "\n")

    never = sum(int((counts[l] == 0).sum()) for l in counts)
    total_slots = len(counts) * num_experts
    print(f"[stats] wrote {out} | {never}/{total_slots} expert slots never routed to")


if __name__ == "__main__":
    main()
