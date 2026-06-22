"""Build the unioned GRPO parquet.

Two modes:

  * Generic (preferred) -- concatenate the per-source GRPO parquets::

        python -m grpo.build_union_data --sources musique sbol rnd --prompt-version v2

    Reads  data/processed/<src>/grpo_{train,val}.parquet for each source.
    Writes data/processed/unioned/grpo_{train,val}.parquet + stats.json.

  * Legacy (no --sources) -- MuSiQue train/val parquet + SBOL train.jsonl
    (95/5 split). Kept for backward compatibility.

Every row is guaranteed an ``extra_info["source"]`` so the rollout loop pins each
row's searches to the right corpus index; ``--prompt-version`` (optional) re-stamps
all rows to one prompt version.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from grpo.data_prep import (
    DEFAULT_DATA_SOURCE,
    build,
    source_from_data_source,
    update_prompt_messages,
)

log = logging.getLogger(__name__)

SBOL_DATA_SOURCE = "sbol_retrieval"
DEFAULT_MUSIQUE_TRAIN = Path("data/processed/musique/grpo_train.parquet")
DEFAULT_MUSIQUE_VAL = Path("data/processed/musique/grpo_val.parquet")
DEFAULT_SBOL_TRAIN = Path("data/processed/sbol/train.jsonl")
DEFAULT_OUT_DIR = Path("data/processed/unioned")

# Generic per-source parquet union (preferred): each source ships its own
# data/processed/<src>/grpo_{train,val}.parquet, and we concatenate them.
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SOURCES = ["musique", "sbol", "rnd"]


def _ensure_source(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every row's extra_info carries a `source` derived from data_source."""

    def _fix(row: pd.Series) -> dict:
        info = dict(row["extra_info"]) if row.get("extra_info") is not None else {}
        if not info.get("source"):
            info["source"] = source_from_data_source(str(row["data_source"]))
        return info

    df = df.copy()
    df["extra_info"] = df.apply(_fix, axis=1)
    return df


def _apply_prompt_version(df: pd.DataFrame, prompt_version: str) -> pd.DataFrame:
    """Rewrite every row's baked prompt + extra_info to a single prompt version."""
    df = df.copy()
    df["prompt"] = df["prompt"].apply(
        lambda messages: update_prompt_messages(list(messages), prompt_version)
    )
    df["extra_info"] = df["extra_info"].apply(
        lambda info: {**dict(info), "prompt_version": prompt_version}
    )
    return df


def union_parquets(
    *,
    sources: list[str],
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    prompt_version: str | None = None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, object]:
    """Concatenate per-source ``grpo_{split}.parquet`` files into one union.

    For each split, reads ``{processed_dir}/{src}/grpo_{split}.parquet`` for every
    source, guarantees each row carries ``extra_info["source"]`` (so the rollout
    loop pins searches to the right corpus index), optionally re-stamps every row
    to a single ``prompt_version``, then writes ``{out_dir}/grpo_{split}.parquet``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {
        "sources": sources,
        "prompt_version": prompt_version,
        "processed_dir": str(processed_dir),
        "counts": {},
    }
    counts: dict[str, dict[str, int]] = {}

    for split in splits:
        frames: list[pd.DataFrame] = []
        for src in sources:
            path = processed_dir / src / f"grpo_{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing {split} parquet for {src!r}: {path}")
            df = pd.read_parquet(path)
            if prompt_version:
                df = _apply_prompt_version(df, prompt_version)
            df = _ensure_source(df)
            counts.setdefault(src, {})[split] = len(df)
            frames.append(df)
        union = pd.concat(frames, ignore_index=True)
        out_path = out_dir / f"grpo_{split}.parquet"
        union.to_parquet(out_path, index=False)
        stats[f"union_{split}"] = len(union)
        per_source = union["extra_info"].apply(lambda x: dict(x)["source"]).value_counts()
        log.info(
            "union %s = %d rows -> %s (by source: %s)",
            split,
            len(union),
            out_path,
            per_source.to_dict(),
        )

    stats["counts"] = counts
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def build_union(
    *,
    musique_train: Path,
    musique_val: Path,
    sbol_train: Path,
    out_dir: Path,
    prompt_version: str = "v1",
    sbol_val_frac: float = 0.05,
    seed: int = 0,
) -> dict[str, int | float]:
    if not musique_train.exists():
        raise FileNotFoundError(f"missing MuSiQue train parquet: {musique_train}")
    if not musique_val.exists():
        raise FileNotFoundError(f"missing MuSiQue val parquet: {musique_val}")
    if not sbol_train.exists():
        raise FileNotFoundError(f"missing SBOL train jsonl: {sbol_train}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "grpo_train.parquet"
    out_val = out_dir / "grpo_val.parquet"

    musique_train_df = pd.read_parquet(musique_train)
    musique_val_df = pd.read_parquet(musique_val)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sbol_train_pq = tmp_path / "sbol_train.parquet"
        sbol_val_pq = tmp_path / "sbol_val.parquet"
        n_sbol_train, n_sbol_val = build(
            sbol_train,
            sbol_train_pq,
            sbol_val_pq,
            prompt_version=prompt_version,
            data_source=SBOL_DATA_SOURCE,
            val_frac=sbol_val_frac,
            seed=seed,
        )
        sbol_train_df = pd.read_parquet(sbol_train_pq)
        sbol_val_df = pd.read_parquet(sbol_val_pq)

    train_df = pd.concat([musique_train_df, sbol_train_df], ignore_index=True)
    val_df = pd.concat([musique_val_df, sbol_val_df], ignore_index=True)

    # Backfill extra_info["source"] for rows read from pre-existing parquets
    # (e.g. MuSiQue built before per-row routing existed). The rollout loop
    # pins each row's searches to this corpus index.
    train_df = _ensure_source(train_df)
    val_df = _ensure_source(val_df)

    train_df.to_parquet(out_train, index=False)
    val_df.to_parquet(out_val, index=False)

    stats = {
        "musique_train": len(musique_train_df),
        "musique_val": len(musique_val_df),
        "sbol_train": n_sbol_train,
        "sbol_val": n_sbol_val,
        "sbol_val_frac": sbol_val_frac,
        "sbol_train_total": n_sbol_train + n_sbol_val,
        "union_train": len(train_df),
        "union_val": len(val_df),
        "seed": seed,
        "prompt_version": prompt_version,
        "musique_data_source": DEFAULT_DATA_SOURCE,
        "sbol_data_source": SBOL_DATA_SOURCE,
        "sources": {
            "musique_train": str(musique_train),
            "musique_val": str(musique_val),
            "sbol_train": str(sbol_train),
        },
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log.info(
        "union train=%d (musique %d + sbol %d), val=%d (musique %d + sbol %d) -> %s",
        stats["union_train"],
        stats["musique_train"],
        stats["sbol_train"],
        stats["union_val"],
        stats["musique_val"],
        stats["sbol_val"],
        out_dir,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Generic per-source parquet union (preferred). When --sources is given we
    # concatenate data/processed/<src>/grpo_{train,val}.parquet for each source.
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help=(
            "Sources to union from per-source grpo_{train,val}.parquet, e.g. "
            f"--sources {' '.join(DEFAULT_SOURCES)}. Enables the generic mode."
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    # Legacy musique+sbol(jsonl) mode (used when --sources is omitted).
    parser.add_argument("--musique-train", type=Path, default=DEFAULT_MUSIQUE_TRAIN)
    parser.add_argument("--musique-val", type=Path, default=DEFAULT_MUSIQUE_VAL)
    parser.add_argument("--sbol-train", type=Path, default=DEFAULT_SBOL_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--prompt-version",
        default=None,
        help=(
            "If set, re-stamp every unioned row to this prompt version. In the "
            "legacy mode it defaults to 'v1' when omitted."
        ),
    )
    parser.add_argument(
        "--sbol-val-frac",
        type=float,
        default=0.05,
        help="Fraction of SBOL train.jsonl held out for validation (default 0.05).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.sources is not None:
        sources = args.sources or DEFAULT_SOURCES
        union_parquets(
            sources=sources,
            processed_dir=args.processed_dir,
            out_dir=args.out_dir,
            prompt_version=args.prompt_version,
        )
        return

    build_union(
        musique_train=args.musique_train,
        musique_val=args.musique_val,
        sbol_train=args.sbol_train,
        out_dir=args.out_dir,
        prompt_version=args.prompt_version or "v1",
        sbol_val_frac=args.sbol_val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
