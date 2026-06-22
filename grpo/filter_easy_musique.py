"""Drop musique rows whose gold_doc_ids are already all in top-k retrieval,
then optionally subsample the remainder at a fixed fraction while preserving
the original hop-count proportions (2-hop / 3-hop / 4-hop).

Two-pass filter:
  1. Remove easy questions — every gold doc already in top-k semantic results.
  2. (Optional) Proportional subsample — keep ``--subsample-frac`` of each
     hop-count group so relative difficulty distribution is unchanged.

Usage (standalone):
    python -m grpo.filter_easy_musique \
        --train data/processed/musique/grpo_train.parquet \
        --val   data/processed/musique/grpo_val.parquet \
        --top-k 8 \
        --subsample-frac 0.5 \
        --tool-server http://localhost:8100 \
        --out-dir data/processed/musique   # overwrites in-place by default

The script queries the already-running tool server (see scripts/serve_tool.sh)
so no GPU / embedder setup is needed here.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

_CLIENT_TAG_RE = re.compile(r"^<client>(.*)</client>$", re.DOTALL)
_USER_TAG_RE = re.compile(r"^<user>(.*)</user>$", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_question(prompt_messages) -> str:
    """Pull the raw question text out of a stored prompt messages array."""
    for msg in prompt_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
        else:
            # numpy record / structured object
            role = str(getattr(msg, "role", ""))
            content = str(getattr(msg, "content", ""))
        if role != "user":
            continue
        stripped = content.strip()
        m = _CLIENT_TAG_RE.match(stripped)
        if m:
            return m.group(1).strip()
        m = _USER_TAG_RE.match(stripped)
        if m:
            return m.group(1).strip()
        return stripped
    return ""


def _gold_ids(row: pd.Series) -> list[str]:
    ei = row["extra_info"]
    if isinstance(ei, dict):
        raw = ei.get("gold_doc_ids", [])
    else:
        raw = []
    return [str(x) for x in raw]


def _search_top_k(
    query: str,
    top_k: int,
    server: str,
    source: str,
    session: requests.Session,
    retries: int = 3,
) -> list[str]:
    """Return the top-k doc_ids from the tool server for *query*."""
    url = server.rstrip("/") + "/local_search"
    payload = {"query": query, "top_k": top_k, "source": source}
    for attempt in range(retries):
        try:
            resp = session.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return [hit["doc_id"] for hit in resp.json()["results"]]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("search attempt %d failed (%s), retrying…", attempt + 1, exc)
            time.sleep(1)
    return []


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def _is_easy(gold: list[str], top_k_ids: list[str]) -> bool:
    """True when every gold doc is already present in the top-k results."""
    if not gold:
        return False
    top_k_set = set(top_k_ids)
    return all(g in top_k_set for g in gold)


def filter_easy(
    df: pd.DataFrame,
    *,
    top_k: int,
    server: str,
    source: str,
    batch_log_every: int = 500,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter *df* in place, returning (filtered_df, stats_dict)."""
    session = requests.Session()

    keep_mask: list[bool] = []
    n_easy = 0

    for i, (_, row) in enumerate(df.iterrows()):
        question = _extract_question(row["prompt"])
        gold = _gold_ids(row)

        if not question or not gold:
            keep_mask.append(True)
            continue

        top_k_ids = _search_top_k(question, top_k, server, source, session)
        easy = _is_easy(gold, top_k_ids)
        keep_mask.append(not easy)
        if easy:
            n_easy += 1

        if (i + 1) % batch_log_every == 0:
            log.info(
                "  processed %d/%d rows, easy so far: %d (%.1f%%)",
                i + 1,
                len(df),
                n_easy,
                100 * n_easy / (i + 1),
            )

    filtered = df[keep_mask].reset_index(drop=True)
    stats = {
        "total": len(df),
        "easy_removed": n_easy,
        "kept": len(filtered),
        "easy_pct": round(100 * n_easy / max(1, len(df)), 2),
        "top_k": top_k,
        "source": source,
        "server": server,
    }
    return filtered, stats


# ---------------------------------------------------------------------------
# Proportional subsample
# ---------------------------------------------------------------------------

_HOP_RE = re.compile(r"^(\d+)hop")


def _hop_count(row: pd.Series) -> str:
    """Extract hop label from question_id, e.g. '2hop__...' -> '2hop'."""
    qid = str(row["extra_info"].get("question_id", ""))
    m = _HOP_RE.match(qid)
    return m.group(1) + "hop" if m else "unknown"


def subsample_proportional(
    df: pd.DataFrame,
    frac: float,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Keep *frac* of each hop-count group, preserving relative proportions.

    Returns (subsampled_df, per_hop_stats).
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"subsample_frac must be in (0, 1), got {frac}")

    df = df.copy()
    df["_hop"] = df.apply(_hop_count, axis=1)

    groups = df["_hop"].value_counts().sort_index()
    log.info("hop distribution before subsample: %s", groups.to_dict())

    rng = np.random.default_rng(seed)
    kept_parts: list[pd.DataFrame] = []
    per_hop: dict[str, dict] = {}

    for hop, group_df in df.groupby("_hop"):
        n_keep = max(1, round(len(group_df) * frac))
        idx = rng.choice(len(group_df), size=n_keep, replace=False)
        kept = group_df.iloc[sorted(idx)]
        kept_parts.append(kept)
        per_hop[str(hop)] = {"before": len(group_df), "kept": n_keep}
        log.info("  %s: %d -> %d (%.1f%%)", hop, len(group_df), n_keep, 100 * n_keep / len(group_df))

    result = pd.concat(kept_parts, ignore_index=True).drop(columns=["_hop"])
    return result, per_hop


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/musique/grpo_train.parquet"),
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("data/processed/musique/grpo_val.parquet"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tool-server", default="http://localhost:8100")
    parser.add_argument("--source", default="musique")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for filtered parquets. Defaults to same dir as --train (in-place).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split(s) to filter (default: both).",
    )
    parser.add_argument(
        "--subsample-frac",
        type=float,
        default=None,
        help=(
            "After removing easy rows, keep this fraction of each hop-count "
            "group (e.g. 0.5 keeps 50%% per group, preserving proportions). "
            "Default: no subsampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out_dir = args.out_dir or args.train.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict] = {}

    splits: list[tuple[str, Path]] = []
    if args.split in ("train", "both"):
        splits.append(("train", args.train))
    if args.split in ("val", "both"):
        splits.append(("val", args.val))

    for split_name, src_path in splits:
        log.info("loading %s split from %s", split_name, src_path)
        df = pd.read_parquet(src_path)
        log.info("  %d rows, filtering with top_k=%d …", len(df), args.top_k)

        filtered, stats = filter_easy(
            df,
            top_k=args.top_k,
            server=args.tool_server,
            source=args.source,
        )
        log.info(
            "  %s easy filter: %d -> %d rows (removed %d / %.1f%%)",
            split_name,
            stats["total"],
            stats["kept"],
            stats["easy_removed"],
            stats["easy_pct"],
        )

        if args.subsample_frac is not None:
            log.info(
                "  subsampling %.0f%% per hop group (seed=%d) …",
                args.subsample_frac * 100,
                args.seed,
            )
            filtered, per_hop = subsample_proportional(
                filtered, frac=args.subsample_frac, seed=args.seed
            )
            stats["subsample_frac"] = args.subsample_frac
            stats["subsample_seed"] = args.seed
            stats["subsample_per_hop"] = per_hop
            stats["final"] = len(filtered)
            log.info(
                "  %s after subsample: %d rows", split_name, len(filtered)
            )
        else:
            stats["final"] = stats["kept"]

        all_stats[split_name] = stats

        out_path = out_dir / src_path.name
        filtered.to_parquet(out_path, index=False)
        log.info(
            "  %s: %d -> %d final rows -> %s",
            split_name,
            stats["total"],
            stats["final"],
            out_path,
        )

    stats_path = out_dir / "filter_easy_stats.json"
    stats_path.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("stats written to %s", stats_path)

    log.info("done.")


if __name__ == "__main__":
    main()
