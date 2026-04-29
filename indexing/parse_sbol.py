"""Parse SBOL FAQ index into corpus.jsonl and eval.jsonl.

Reads data/raw/sbol/faq_index_28_apr.json and produces:
  {out_dir}/corpus.jsonl  — one passage per FAQ entry
  {out_dir}/eval.jsonl    — one eval row per alternative_question

Document text format:
    Вопрос:{question}
    Раздел:{sections}
    Ответ:{answer}

doc_id is derived from the FAQ entry's metadata.question_id:
    sbol_<question_id>

Eval rows use alternative_questions as the retrieval query and the
parent FAQ doc_id as the single gold document.  Entries with no
alternative_questions produce no eval rows.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse(raw_path: Path, out_dir: Path) -> tuple[int, int]:
    """Return (num_corpus, num_eval)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    faq_data: list[dict] = json.loads(raw_path.read_text(encoding="utf-8"))
    log.info("loaded %d FAQ entries from %s", len(faq_data), raw_path)

    corpus_path = out_dir / "corpus.jsonl"
    eval_path = out_dir / "eval.jsonl"

    num_corpus = 0
    num_eval = 0

    with corpus_path.open("w", encoding="utf-8") as cf, \
         eval_path.open("w", encoding="utf-8") as ef:

        for entry in faq_data:
            meta = entry.get("metadata", {})
            qid = meta.get("question_id")
            if not qid:
                log.warning("entry missing question_id, skipping: %s", entry.get("question", "")[:60])
                continue

            doc_id = f"sbol_{qid}"
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            sections = entry.get("sections", meta.get("sections", ""))

            text = f"Вопрос:{question}\nРаздел:{sections}\nОтвет:{answer}"

            cf.write(json.dumps({"doc_id": doc_id, "title": question, "text": text}, ensure_ascii=False))
            cf.write("\n")
            num_corpus += 1

            for i, alt_q in enumerate(meta.get("alternative_questions", [])):
                if not alt_q:
                    continue
                row = {
                    "question_id": f"{doc_id}_alt_{i}",
                    "question": alt_q,
                    "gold_doc_ids": [doc_id],
                }
                ef.write(json.dumps(row, ensure_ascii=False))
                ef.write("\n")
                num_eval += 1

    log.info("wrote %d corpus entries to %s", num_corpus, corpus_path)
    log.info("wrote %d eval rows to %s", num_eval, eval_path)
    return num_corpus, num_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SBOL FAQ JSON into corpus.jsonl + eval.jsonl."
    )
    parser.add_argument(
        "--raw",
        default="data/raw/sbol/faq_index_28_apr.json",
        help="path to the raw FAQ JSON file",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/sbol",
        help="output directory for corpus.jsonl and eval.jsonl",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parse(Path(args.raw), Path(args.out_dir))


if __name__ == "__main__":
    main()
