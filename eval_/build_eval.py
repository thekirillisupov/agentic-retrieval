"""Turn raw MuSiQue dev examples into eval records with gold doc_ids.

Reads:
    {raw_dir}/musique_ans_v1.0_dev.jsonl
    {processed_dir}/source_to_doc_id.json

Writes:
    {processed_dir}/musique_dev_eval.jsonl

Each output line:
    {
      "question_id": "...",
      "question": "...",
      "answer": "...",
      "gold_doc_ids": ["musique_p_...", ...]
    }

Source-id format must match parse_musique: f"dev:{question_id}:{paragraph_idx}".
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEV_FILENAME = "musique_ans_v1.0_dev.jsonl"


def build(raw_dir: Path, processed_dir: Path, out_path: Path) -> int:
    mapping_path = processed_dir / "source_to_doc_id.json"
    source_to_doc_id: dict[str, str] = json.loads(mapping_path.read_text())

    dev_path = raw_dir / DEV_FILENAME
    if not dev_path.exists():
        raise FileNotFoundError(f"missing dev file: {dev_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_missing = 0
    with dev_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
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
                key = f"dev:{qid}:{p['idx']}"
                doc_id = source_to_doc_id.get(key)
                if doc_id is None:
                    n_missing += 1
                    continue
                if doc_id not in gold:
                    gold.append(doc_id)

            record = {
                "question_id": qid,
                "question": example["question"],
                "answer": example.get("answer", ""),
                "gold_doc_ids": gold,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    log.info(
        "wrote %d eval examples to %s (%d unanswerable skipped, %d missing source-id mappings)",
        n_written, out_path, n_skipped, n_missing,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/musique")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument(
        "--out", default="data/processed/musique_dev_eval.jsonl"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build(Path(args.raw_dir), Path(args.processed_dir), Path(args.out))


if __name__ == "__main__":
    main()
