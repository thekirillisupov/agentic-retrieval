"""Stratified sample of MuSiQue dialogue queries for train / eval.

Reads ``data/raw/musique_dialogs/musique_dialogues.jsonl``, resolves
``gold_chunk_ids`` to ``gold_doc_ids`` via ``source_to_doc_id.json``, and
draws disjoint train / eval sets while preserving the original hop-count mix
(2-hop / 3-hop / 4-hop as difficulty strata).

Outputs under ``data/processed/musique_dialogs/`` by default:
    musique_dialog_train.jsonl
    musique_dialog_eval.jsonl
    grpo_train.parquet
    grpo_val.parquet
    sample_stats.json

Usage (from repo root):

    python -m indexing.sample_musique_dialogs \\
        --train-size 2000 --eval-size 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from grpo.data_prep import DEFAULT_DATA_SOURCE, _row

log = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/raw/musique_dialogs/musique_dialogues.jsonl")
DEFAULT_MAPPING = Path("data/processed/musique/source_to_doc_id.json")
DEFAULT_OUT_DIR = Path("data/processed/musique_dialogs")
MUSIQUE_DIALOG_DATA_SOURCE = "musique_dialog_retrieval"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_to_doc_id(chunk_id: str, source_to_doc_id: dict[str, str]) -> str | None:
    file_name, idx = chunk_id.rsplit("::", 1)
    for split in ("train", "dev"):
        key = f"{split}:{file_name}:{idx}"
        doc_id = source_to_doc_id.get(key)
        if doc_id is not None:
            return doc_id
    return None


def resolve_gold_doc_ids(
    record: dict[str, Any],
    source_to_doc_id: dict[str, str],
    *,
    gold_chunk_field: str = "gold_chunk_ids",
) -> list[str]:
    gold_doc_ids: list[str] = []
    for chunk_id in record.get(gold_chunk_field) or []:
        doc_id = chunk_to_doc_id(str(chunk_id), source_to_doc_id)
        if doc_id is None:
            continue
        if doc_id not in gold_doc_ids:
            gold_doc_ids.append(doc_id)
    return gold_doc_ids


def hop_count(record: dict[str, Any]) -> int:
    return int(record["verification"]["musique"]["num_hops"])


def convert_record(
    record: dict[str, Any],
    source_to_doc_id: dict[str, str],
    *,
    gold_chunk_field: str = "gold_chunk_ids",
) -> dict[str, Any] | None:
    gold_doc_ids = resolve_gold_doc_ids(
        record, source_to_doc_id, gold_chunk_field=gold_chunk_field
    )
    if not gold_doc_ids:
        return None
    return {
        "question_id": record["id"],
        "question": record["question"],
        "answer": record.get("answer", ""),
        "gold_doc_ids": gold_doc_ids,
        "num_hops": hop_count(record),
        "difficulty": record.get("difficulty", ""),
    }


def stratified_sample(
    pool: list[dict[str, Any]],
    n: int,
    rng: random.Random,
    *,
    group_key: str = "num_hops",
) -> list[dict[str, Any]]:
    """Sample *n* rows preserving proportions of ``group_key``."""
    if n > len(pool):
        raise ValueError(f"requested {n} samples but only {len(pool)} rows available")

    by_group: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_group[row[group_key]].append(row)

    groups = sorted(by_group, key=lambda g: (isinstance(g, str), g))
    total = len(pool)
    targets: dict[Any, int] = {}
    assigned = 0
    for i, group in enumerate(groups):
        if i == len(groups) - 1:
            targets[group] = n - assigned
        else:
            count = round(n * len(by_group[group]) / total)
            targets[group] = count
            assigned += count

    picked: list[dict[str, Any]] = []
    for group in groups:
        group_pool = by_group[group]
        k = min(targets[group], len(group_pool))
        if k < targets[group]:
            log.warning(
                "hop group %s: wanted %d samples, only %d available",
                group,
                targets[group],
                k,
            )
        picked.extend(rng.sample(group_pool, k))

    rng.shuffle(picked)
    return picked


def group_counts(rows: list[dict[str, Any]], key: str = "num_hops") -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[key])] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def to_parquet(
    records: list[dict[str, Any]],
    out_path: Path,
    *,
    prompt_version: str,
    data_source: str,
) -> int:
    rows = [_row(rec, prompt_version, data_source=data_source) for rec in records]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def sample(
    input_path: Path,
    mapping_path: Path,
    out_dir: Path,
    *,
    train_size: int = 2000,
    eval_size: int = 200,
    seed: int = 0,
    prompt_version: str = "v2_search_only",
    data_source: str = MUSIQUE_DIALOG_DATA_SOURCE,
    gold_chunk_field: str = "gold_chunk_ids",
    write_parquet: bool = True,
) -> dict[str, Any]:
    source_to_doc_id: dict[str, str] = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_records = load_jsonl(input_path)

    converted: list[dict[str, Any]] = []
    skipped_no_gold = 0
    for rec in raw_records:
        row = convert_record(rec, source_to_doc_id, gold_chunk_field=gold_chunk_field)
        if row is None:
            skipped_no_gold += 1
            continue
        converted.append(row)

    rng = random.Random(seed)
    eval_rows = stratified_sample(converted, eval_size, rng)
    eval_ids = {row["question_id"] for row in eval_rows}
    train_pool = [row for row in converted if row["question_id"] not in eval_ids]
    train_rows = stratified_sample(train_pool, train_size, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "musique_dialog_train.jsonl"
    eval_jsonl = out_dir / "musique_dialog_eval.jsonl"
    write_jsonl(train_rows, train_jsonl)
    write_jsonl(eval_rows, eval_jsonl)

    stats: dict[str, Any] = {
        "input": str(input_path),
        "mapping": str(mapping_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "total_converted": len(converted),
        "skipped_no_gold": skipped_no_gold,
        "pool_hop_counts": group_counts(converted),
        "train_hop_counts": group_counts(train_rows),
        "eval_hop_counts": group_counts(eval_rows),
        "prompt_version": prompt_version,
        "data_source": data_source,
        "gold_chunk_field": gold_chunk_field,
    }

    if write_parquet:
        n_train = to_parquet(
            train_rows,
            out_dir / "grpo_train.parquet",
            prompt_version=prompt_version,
            data_source=data_source,
        )
        n_eval = to_parquet(
            eval_rows,
            out_dir / "grpo_val.parquet",
            prompt_version=prompt_version,
            data_source=data_source,
        )
        stats["grpo_train_rows"] = n_train
        stats["grpo_val_rows"] = n_eval

    stats_path = out_dir / "sample_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log.info(
        "sampled train=%d eval=%d (pool=%d) -> %s",
        len(train_rows),
        len(eval_rows),
        len(converted),
        out_dir,
    )
    log.info("train hop counts: %s", stats["train_hop_counts"])
    log.info("eval hop counts: %s", stats["eval_hop_counts"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-version", default="v2_search_only")
    parser.add_argument(
        "--data-source",
        default=MUSIQUE_DIALOG_DATA_SOURCE,
        help=(
            "veRL data_source tag. Rows still route to the musique index via "
            f"{DEFAULT_DATA_SOURCE!r} suffix stripping unless overridden."
        ),
    )
    parser.add_argument(
        "--gold-chunk-field",
        default="gold_chunk_ids",
        choices=("gold_chunk_ids", "positive_chunk_ids"),
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Only write jsonl splits, skip grpo parquet export.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sample(
        args.input,
        args.mapping,
        args.out_dir,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        prompt_version=args.prompt_version,
        data_source=args.data_source,
        gold_chunk_field=args.gold_chunk_field,
        write_parquet=not args.no_parquet,
    )


if __name__ == "__main__":
    main()
