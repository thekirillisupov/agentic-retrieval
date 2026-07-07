"""Stage 4: rank candidates by NLL on the held-out slice of the train parquet.

The train parquet is the pipeline's validation domain: rows are shuffled with
the same seed as the stats pass and the TAIL slice (reserved by
``collect_stats --val-samples``) is scored, so validation never sees the rows
that drove expert selection.

Each candidate (the unpruned W8A8 baseline, pruned variants at different keep
levels, ...) is loaded in turn and scored with teacher-forced mean NLL per
token over the rendered agentic prompts. Lower is better; the delta against
the first (baseline) model is the pruning cost on-domain.

  python -m prune.validate \
    --train-parquet data/processed/unioned/grpo_train.parquet \
    --models base=checkpoints/quantized/qwen3_5_35b_a3b_w8a8 \
             pruned=checkpoints/pruned/qwen3_5_35b_a3b_w8a8/model \
    --num-samples 512 --val-samples 128 \
    --output checkpoints/pruned/qwen3_5_35b_a3b_w8a8/validation.json

NLL on-domain is a cheap smoke test, not the accept gate. The accept gate is
the agent eval: serve the pruned checkpoint (scripts/serve_vllm_int8.sh) and
run scripts/run_eval.sh, comparing ndcg@10 / recall@10 to the unpruned
candidate (see prune/README.md).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _score_model(model_dir: str, ds, *, dtype: str, device_map: str) -> dict:
    import torch

    from prune.common import free_model, load_model

    model = load_model(model_dir, dtype=dtype, device_map=device_map)
    device = getattr(model, "device", "cpu")
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for row in ds:
            input_ids = torch.tensor([row["input_ids"]], device=device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            n = input_ids.shape[1] - 1  # teacher-forced predictions
            total_nll += float(out.loss) * n
            total_tokens += n
    free_model(model)
    nll = total_nll / max(total_tokens, 1)
    return {"nll": nll, "ppl": math.exp(nll), "tokens": total_tokens}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument(
        "--models",
        required=True,
        nargs="+",
        metavar="NAME=PATH",
        help="Candidates to score, first one is the baseline for deltas.",
    )
    ap.add_argument("--output", required=True, help="Where to write validation.json.")
    ap.add_argument("--num-samples", type=int, default=512,
                    help="MUST match collect_stats --num-samples (fixes the split).")
    ap.add_argument("--val-samples", type=int, default=128,
                    help="MUST match collect_stats --val-samples.")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0,
                    help="MUST match collect_stats --seed.")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--prompt-version", default="v2_search_only")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    candidates: list[tuple[str, str]] = []
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"--models entries must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        candidates.append((name, path))

    from transformers import AutoTokenizer

    from prune.common import load_split

    # One tokenizer for everyone: pruning does not touch the tokenizer, and a
    # shared tokenization keeps the token counts comparable across candidates.
    tokenizer = AutoTokenizer.from_pretrained(candidates[0][1], trust_remote_code=True)
    ds = load_split(
        args.train_parquet,
        tokenizer,
        split="val",
        num_stats=args.num_samples,
        num_val=args.val_samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        prompt_field=args.prompt_field,
        default_prompt_version=args.prompt_version,
    )
    print(f"[validate] {len(ds)} held-out sequences")

    results: dict[str, dict] = {}
    for name, path in candidates:
        print(f"[validate] scoring {name}: {path}")
        results[name] = {"path": path, **_score_model(
            path, ds, dtype=args.dtype, device_map=args.device_map
        )}
        print(f"[validate] {name}: nll={results[name]['nll']:.4f} "
              f"ppl={results[name]['ppl']:.3f}")

    base_name = candidates[0][0]
    base_nll = results[base_name]["nll"]
    for name in results:
        results[name]["delta_nll_vs_" + base_name] = results[name]["nll"] - base_nll

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "train_parquet": args.train_parquet,
        "val_samples": args.val_samples,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "baseline": base_name,
        "results": results,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n[validate] report -> {out}")
    width = max(len(n) for n, _ in candidates)
    print(f"{'model'.ljust(width)}  {'nll':>8}  {'ppl':>8}  {'Δnll':>8}")
    for name, _ in candidates:
        r = results[name]
        print(f"{name.ljust(width)}  {r['nll']:8.4f}  {r['ppl']:8.3f}  "
              f"{r['delta_nll_vs_' + base_name]:+8.4f}")
    print("\nAccept gate: serve the winner and run scripts/run_eval.sh "
          "(ndcg@10 vs the unpruned candidate).")


if __name__ == "__main__":
    main()
