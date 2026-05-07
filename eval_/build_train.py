"""Turn raw MuSiQue *train* examples into per-question records with gold doc_ids.

Mirror of `build_eval.py` but for the train split. Used as the source dataset for
GRPO rollouts. Each output line:
    {
      "question_id": "...",
      "question": "...",
      "answer": "...",
      "gold_doc_ids": ["musique_p_...", ...]
    }

Source-id format must match parse_musique: f"train:{question_id}:{paragraph_idx}".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

TRAIN_FILENAME = "musique_ans_v1.0_train.jsonl"


def build(raw_dir: Path, processed_dir: Path, out_path: Path, min_gold: int = 1) -> int:
    mapping_path = processed_dir / "source_to_doc_id.json"
    source_to_doc_id: dict[str, str] = json.loads(mapping_path.read_text())

    train_path = raw_dir / TRAIN_FILENAME
    if not train_path.exists():
        raise FileNotFoundError(f"missing train file: {train_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_no_gold = 0
    n_skipped = 0
    with train_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            if not example.get("answerable", True):
                n_skipped += 1
                continue
            qid = example["id"]
            gold: list[str] = []
            for p in example["paragraphs"]:
                if not p.get("is_supporting", False):
                    continue
                key = f"train:{qid}:{p['idx']}"
                doc_id = source_to_doc_id.get(key)
                if doc_id is None:
                    continue
                if doc_id not in gold:
                    gold.append(doc_id)
            if len(gold) < min_gold:
                n_no_gold += 1
                continue

            record = {
                "question_id": qid,
                "question": example["question"],
                "answer": example.get("answer", ""),
                "gold_doc_ids": gold,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    log.info(
        "wrote %d train examples to %s (%d without gold, %d unanswerable)",
        n_written, out_path, n_no_gold, n_skipped,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/musique")
    parser.add_argument("--processed-dir", default="data/processed/musique")
    parser.add_argument(
        "--out", default="data/processed/musique/musique_train.jsonl"
    )
    parser.add_argument("--min-gold", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(Path(args.raw_dir), Path(args.processed_dir), Path(args.out), min_gold=args.min_gold)


if __name__ == "__main__":
    main()
