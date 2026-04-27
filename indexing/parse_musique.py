"""Parse raw MuSiQue jsonl into a deduplicated corpus.

Inputs:
    musique_ans_v1.0_train.jsonl
    musique_ans_v1.0_dev.jsonl
(Located in --raw-dir; we ignore the _full_ files — only answerable questions.)

Outputs (in --out-dir):
    corpus.jsonl              one record per unique passage: {doc_id, title, text}
    source_to_doc_id.json     {f"{split}:{question_id}:{paragraph_idx}": doc_id}
    stats.json                {num_raw, num_dedup, by_split: {...}}

Dedup is content-based (sha256 of "title\\n\\ntext" lowercased). The same paragraph
appears in many MuSiQue examples; dedup typically shrinks the raw count by ~10–100×.

doc_id format: `musique_p_NNNNNN` (zero-padded counter, stable order = first
appearance during the train→dev sweep).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


SPLITS = [
    ("train", "musique_ans_v1.0_train.jsonl"),
    ("dev", "musique_ans_v1.0_dev.jsonl"),
]


def _content_hash(title: str, text: str) -> str:
    payload = (title.strip() + "\n\n" + text.strip()).lower()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_examples(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse(raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_to_doc_id: dict[str, str] = {}
    corpus: list[dict] = []
    source_to_doc_id: dict[str, str] = {}
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"raw": 0, "examples": 0})

    counter = 0
    for split, filename in SPLITS:
        path = raw_dir / filename
        if not path.exists():
            log.warning("missing split file: %s — skipping", path)
            continue
        log.info("parsing %s", path)
        for example in _iter_examples(path):
            qid = example["id"]
            by_split[split]["examples"] += 1
            for p in example["paragraphs"]:
                idx = p["idx"]
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                by_split[split]["raw"] += 1

                h = _content_hash(title, text)
                doc_id = hash_to_doc_id.get(h)
                if doc_id is None:
                    doc_id = f"musique_p_{counter:06d}"
                    counter += 1
                    hash_to_doc_id[h] = doc_id
                    corpus.append({"doc_id": doc_id, "title": title, "text": text})

                source_to_doc_id[f"{split}:{qid}:{idx}"] = doc_id

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for record in corpus:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    mapping_path = out_dir / "source_to_doc_id.json"
    mapping_path.write_text(json.dumps(source_to_doc_id, ensure_ascii=False))

    stats = {
        "num_dedup": len(corpus),
        "num_raw": sum(s["raw"] for s in by_split.values()),
        "num_examples": sum(s["examples"] for s in by_split.values()),
        "by_split": dict(by_split),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    log.info(
        "parsed %d unique passages from %d raw paragraphs (%d examples)",
        stats["num_dedup"],
        stats["num_raw"],
        stats["num_examples"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/musique")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parse(Path(args.raw_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
