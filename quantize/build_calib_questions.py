"""Extract calibration questions from a veRL GRPO parquet into .jsonl.

Reads the ``prompt`` column (system + user messages), strips the
``<client>``/``<user>`` wrapper from the user turn, and writes one JSON
object per line with a ``question`` field (plus row metadata for filtering).

Usage (from repo root)::

    python -m quantize.build_calib_questions \\
        --in data/processed/unioned/grpo_train.parquet \\
        --out data/calib/questions.jsonl

Pass the output to ``quantize.quantize --questions`` (default field
``question``). Pick ``--prompt-version`` to match the corpus you serve:
``v2_search_only`` for musique/sbol/rnd rows, ``v2`` for ckr.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantize.calibration import extract_question_from_prompt


def build_rows(df: pd.DataFrame, *, dedupe: bool = True) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for _, row in df.iterrows():
        question = extract_question_from_prompt(row["prompt"])
        if not question:
            continue
        if dedupe:
            if question in seen:
                continue
            seen.add(question)
        extra = row.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        rows.append(
            {
                "question": question,
                "question_id": extra.get("question_id", ""),
                "source": extra.get("source", ""),
                "prompt_version": extra.get("prompt_version", ""),
                "data_source": row.get("data_source", ""),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        default="data/processed/unioned/grpo_train.parquet",
        help="veRL GRPO parquet (train split).",
    )
    ap.add_argument(
        "--out",
        default="data/calib/questions.jsonl",
        help="Output .jsonl (one object per line with a question field).",
    )
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate question strings (default: dedupe by question text).",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    rows = build_rows(df, dedupe=not args.no_dedupe)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[build_calib_questions] {len(rows)} questions "
        f"from {len(df)} parquet rows -> {out}"
    )


if __name__ == "__main__":
    main()
