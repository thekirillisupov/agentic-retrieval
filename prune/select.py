"""Stage 2: router stats -> prune plan (which experts each layer keeps).

Pure CPU, instant. Pick the retention with either:

  --keep N          keep the top-N experts per layer (uniform N — the config
                    carries a single num_experts, per-layer counts don't fit
                    the checkpoint format), or
  --coverage 0.99   keep the smallest uniform N such that EVERY layer retains
                    >= 99% of its observed router probability mass.

  python -m prune.select \
    --stats checkpoints/pruned/qwen3_5_35b_a3b_w8a8/router_stats.json \
    --coverage 0.99 \
    --output checkpoints/pruned/qwen3_5_35b_a3b_w8a8/prune_plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prune.plan import VALID_METRICS, select_experts, summarize_plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", required=True, help="router_stats.json from collect_stats.")
    ap.add_argument("--output", required=True, help="Where to write prune_plan.json.")
    how = ap.add_mutually_exclusive_group(required=True)
    how.add_argument("--keep", type=int, help="Uniform experts kept per layer.")
    how.add_argument("--coverage", type=float,
                     help="Min router-mass fraction every layer must retain (0..1].")
    ap.add_argument("--metric", default="mass", choices=list(VALID_METRICS),
                    help="Ranking signal: routed probability mass (default) or "
                    "raw top-k selection count.")
    ap.add_argument("--min-keep", type=int, default=None,
                    help="Lower bound on kept experts (default: the router's top_k).")
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    plan = select_experts(
        stats,
        keep=args.keep,
        coverage=args.coverage,
        metric=args.metric,
        min_keep=args.min_keep,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan) + "\n")
    print(f"[select] wrote {out}")
    print(summarize_plan(plan))


if __name__ == "__main__":
    main()
