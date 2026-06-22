"""Build CKR GRPO train/val parquet from one or more verified JSONL files.

Each record in a verified JSONL (e.g. v2_verified.jsonl, v3_verified.jsonl) has:
    id              -- unique question identifier
    question        -- the question text
    answer          -- the gold answer
    gold_chunk_ids      -- list of "file_name::index" chunk ids (legacy)
    positive_chunk_ids  -- expanded positive set (preferred for per-doc datasets)

Chunk ids use the "file_name::index" format, while the FAISS index
stores doc_ids as "document_id_index".  This script loads ckr_index.jsonl
to build that mapping, deduplicates across all input files by id, then
calls grpo.data_prep.build to produce veRL-ready parquet files.

Usage (from repo root):

    # single file
    python -m indexing.compose_ckr_train \\
        --input data/raw/ckr/v2_verified.jsonl \\
        --ckr-index data/raw/ckr/ckr_index.jsonl \\
        --out-dir data/processed/ckr --prompt-version v2

    # merge multiple files
    python -m indexing.compose_ckr_train \\
        --input data/raw/ckr/v2_verified.jsonl data/raw/ckr/v3_verified.jsonl \\
        --ckr-index data/raw/ckr/ckr_index.jsonl \\
        --out-dir data/processed/ckr --prompt-version v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CKR_DATA_SOURCE = "ckr_retrieval"
CKR_PER_DOC_FILTERED_DATA_SOURCE = "ckr_per_doc_filtered_retrieval"
DEFAULT_VAL_FRAC = 0.1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_chunk_id_map(index_path: Path) -> dict[str, str]:
    """Build mapping: 'file_name::index' -> 'document_id_index'.

    ckr_index.jsonl columns used:
        file_name   -- e.g. "abc123.html"
        index       -- 0-based chunk position within that file
        document_id -- UUID for the source document

    The FAISS index doc_id is built as "{document_id}_{index}"
    (see indexing/compose_ckr_index.py), so that is what gold_doc_ids
    must contain.
    """
    mapping: dict[str, str] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = f"{d['file_name']}::{d['index']}"
            doc_id = f"{d['document_id']}_{d['index']}"
            mapping[key] = doc_id
    return mapping


def convert_records(
    records: list[dict[str, Any]],
    chunk_id_map: dict[str, str],
    *,
    gold_chunk_field: str = "gold_chunk_ids",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert verified JSONL records to standard grpo.data_prep train format.

    Returns (converted_rows, stats).  Rows without any resolvable gold
    doc_ids are skipped.
    """
    out: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "total": len(records),
        "kept": 0,
        "skipped_no_gold": 0,
        "records_with_missing_chunks": 0,
    }

    for rec in records:
        raw_ids: list[str] = rec.get(gold_chunk_field) or []
        gold_doc_ids: list[str] = []
        missing = 0
        for chunk_id in raw_ids:
            doc_id = chunk_id_map.get(chunk_id)
            if doc_id is None:
                log.warning("chunk_id not in index: %s", chunk_id)
                missing += 1
            elif doc_id not in gold_doc_ids:
                gold_doc_ids.append(doc_id)

        if missing:
            stats["records_with_missing_chunks"] += 1

        if not gold_doc_ids:
            stats["skipped_no_gold"] += 1
            continue

        out.append(
            {
                "question_id": rec["id"],
                "question": rec["question"],
                "answer": rec.get("answer", ""),
                "gold_doc_ids": gold_doc_ids,
            }
        )
        stats["kept"] += 1

    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/raw/ckr/v2_verified.jsonl")],
        help="One or more verified JSONL files; records are merged and deduplicated by id.",
    )
    parser.add_argument(
        "--ckr-index",
        type=Path,
        default=Path("data/raw/ckr/ckr_index.jsonl"),
        help="Path to ckr_index.jsonl (used to resolve doc_ids)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/ckr"),
        help="Output directory; receives ckr_train.jsonl, grpo_{train,val}.parquet",
    )
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument(
        "--data-source",
        default=CKR_DATA_SOURCE,
        help=(
            "veRL reward_fn_key / W&B grouping tag. Use "
            f"{CKR_PER_DOC_FILTERED_DATA_SOURCE!r} for per-doc filtered CKR "
            "(still routes to the ckr index)."
        ),
    )
    parser.add_argument(
        "--gold-chunk-field",
        default="gold_chunk_ids",
        choices=("gold_chunk_ids", "positive_chunk_ids"),
        help=(
            "JSONL field with gold chunk ids in file_name::index format "
            "(use positive_chunk_ids for per-doc filtered datasets)."
        ),
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of rows held out for validation (default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    log.info("loading CKR index from %s", args.ckr_index)
    chunk_id_map = build_chunk_id_map(args.ckr_index)
    log.info("loaded %d chunk-id mappings", len(chunk_id_map))

    seen_ids: set[str] = set()
    raw_records: list[dict[str, Any]] = []
    for path in args.input:
        file_records = load_jsonl(path)
        before = len(raw_records)
        for rec in file_records:
            rid = rec["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                raw_records.append(rec)
        log.info(
            "loaded %s: %d records, %d new (total %d)",
            path,
            len(file_records),
            len(raw_records) - before,
            len(raw_records),
        )

    train_records, conv_stats = convert_records(
        raw_records, chunk_id_map, gold_chunk_field=args.gold_chunk_field
    )
    log.info("conversion: %s", conv_stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = args.out_dir / "ckr_train.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("wrote %d train records to %s", len(train_records), train_jsonl)

    from grpo.data_prep import build as build_grpo_parquet

    n_train, n_val = build_grpo_parquet(
        train_jsonl,
        args.out_dir / "grpo_train.parquet",
        args.out_dir / "grpo_val.parquet",
        prompt_version=args.prompt_version,
        data_source=args.data_source,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    summary: dict[str, Any] = {
        "input": [str(p) for p in args.input],
        "ckr_index": str(args.ckr_index),
        "out_dir": str(args.out_dir),
        "prompt_version": args.prompt_version,
        "data_source": args.data_source,
        "gold_chunk_field": args.gold_chunk_field,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "conversion": conv_stats,
        "grpo_train_rows": n_train,
        "grpo_val_rows": n_val,
    }
    stats_path = args.out_dir / "compose_ckr_stats.json"
    stats_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info(
        "done: %d train + %d val rows -> %s (stats: %s)",
        n_train,
        n_val,
        args.out_dir,
        stats_path,
    )


if __name__ == "__main__":
    main()
