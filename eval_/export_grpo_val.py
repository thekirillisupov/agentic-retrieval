"""Export grpo_val.parquet to eval_.run_eval-compatible JSONL.

Each output row: question_id, question, gold_doc_ids, source.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def export_grpo_val(parquet_path: Path, jsonl_path: Path) -> int:
    df = pd.read_parquet(parquet_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            extra = row["extra_info"]
            if isinstance(extra, str):
                extra = json.loads(extra)
            prompt = row["prompt"]
            if isinstance(prompt, str):
                prompt = json.loads(prompt)
            question = next(m["content"] for m in prompt if m["role"] == "user")
            out = {
                "question_id": extra["question_id"],
                "question": question,
                "gold_doc_ids": [str(doc_id) for doc_id in extra["gold_doc_ids"]],
                "source": extra.get("source", "ckr"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Input grpo_val.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output eval.jsonl",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = export_grpo_val(args.parquet, args.out)
    log.info("wrote %d rows to %s", n, args.out)


if __name__ == "__main__":
    main()
