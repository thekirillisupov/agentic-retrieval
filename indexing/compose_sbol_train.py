"""Classify SBOL questions by retrieval difficulty and build dialog train data.

1. Run each eval question against the existing SBOL FAISS index (top-k=3).
2. Questions whose gold doc is in top-k are *simple*; the rest are *complex*.
3. Write train.jsonl with only complex questions, each prefixed by 1–2
   question–answer turns sampled from simple questions in the same Раздел.

Outputs under data/processed/sbol/:
    simple.jsonl          — simple questions (excluded from training)
    train.jsonl           — complex questions wrapped in <client>/<bot> dialog
    compose_sbol_stats.json

Usage (from repo root):

    python -m indexing.compose_sbol_train \\
        --eval data/processed/sbol/eval.jsonl \\
        --index-dir indexes/sbol \\
        --raw data/raw/sbol/faq_index_28_apr.json \\
        --out-dir data/processed/sbol
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from tool_server.embedder import E5Embedder
from tool_server.index import FaissIndex

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_PREFIX_MIN = 1
DEFAULT_PREFIX_MAX = 2


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


def load_faq_meta(raw_path: Path) -> dict[str, dict[str, str]]:
    """Map question_id (sbol_<uuid>) -> {section, answer, question}."""
    faq_data: list[dict] = json.loads(raw_path.read_text(encoding="utf-8"))
    meta: dict[str, dict[str, str]] = {}
    for entry in faq_data:
        qid = entry.get("metadata", {}).get("question_id")
        if not qid:
            continue
        doc_id = f"sbol_{qid}"
        sections = entry.get("sections", entry.get("metadata", {}).get("sections", ""))
        meta[doc_id] = {
            "section": sections,
            "answer": entry.get("answer", ""),
            "question": entry.get("question", ""),
        }
    return meta


def classify_retrieval(
    eval_rows: list[dict[str, Any]],
    index: FaissIndex,
    embedder: E5Embedder,
    *,
    top_k: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Split eval rows into simple (gold in top-k) and complex."""
    simple: list[dict[str, Any]] = []
    complex_: list[dict[str, Any]] = []
    stats = {
        "total": len(eval_rows),
        "simple": 0,
        "complex": 0,
        "missing_gold_in_index": 0,
    }

    questions = [r["question"] for r in eval_rows]
    query_embs = embedder.encode_queries(questions, batch_size=batch_size)

    for i, row in enumerate(tqdm(eval_rows, desc="classify retrieval")):
        gold_ids = set(row.get("gold_doc_ids") or [])
        if not gold_ids:
            complex_.append(row)
            stats["complex"] += 1
            continue

        q = query_embs[i : i + 1].astype(np.float32)
        _, ids = index.index.search(q, top_k)
        retrieved_ids = {
            index.metadata[idx].doc_id
            for idx in ids[0]
            if idx >= 0
        }

        hit = bool(gold_ids & retrieved_ids)
        if not hit:
            for gid in gold_ids:
                if gid not in index.doc_id_to_pos:
                    stats["missing_gold_in_index"] += 1
                    break

        enriched = dict(row)
        enriched["retrieval_top_k"] = list(retrieved_ids)
        enriched["is_simple"] = hit

        if hit:
            simple.append(enriched)
            stats["simple"] += 1
        else:
            complex_.append(enriched)
            stats["complex"] += 1

    return simple, complex_, stats


def _section_key(section: str) -> str:
    return section.strip()


def _top_level_section(section: str) -> str:
    return section.split("/", 1)[0].strip() if section else ""


def _pick_prefixes(
    candidates: list[dict[str, Any]],
    *,
    exclude_id: str,
    rng: random.Random,
    min_n: int,
    max_n: int,
) -> list[dict[str, Any]]:
    pool = [c for c in candidates if c["question_id"] != exclude_id]
    if not pool:
        return []
    n = rng.randint(min_n, max_n)
    n = min(n, len(pool))
    return rng.sample(pool, n)


def format_dialog(prefixes: list[dict[str, Any]], final_question: str) -> str:
    parts: list[str] = []
    for row in prefixes:
        parts.append(f"<client>{row['question']}</client>")
        parts.append(f"<bot>{row['answer']}</bot>")
    parts.append(f"<client>{final_question}</client>")
    return "\n".join(parts)


def build_train_rows(
    simple_rows: list[dict[str, Any]],
    complex_rows: list[dict[str, Any]],
    faq_meta: dict[str, dict[str, str]],
    *,
    seed: int,
    prefix_min: int,
    prefix_max: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_top_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in simple_rows:
        section = _section_key(faq_meta.get(row["question_id"], {}).get("section", ""))
        top = _top_level_section(section)
        enriched = dict(row)
        enriched["section"] = section
        enriched["answer"] = faq_meta.get(row["question_id"], {}).get("answer", "")
        by_section[section].append(enriched)
        if top:
            by_top_section[top].append(enriched)

    train_rows: list[dict[str, Any]] = []
    stats = {
        "complex_total": len(complex_rows),
        "train_built": 0,
        "no_prefix_same_section": 0,
        "used_top_level_fallback": 0,
    }

    for row in complex_rows:
        qid = row["question_id"]
        meta = faq_meta.get(qid, {})
        section = _section_key(meta.get("section", ""))
        final_q = row["question"]

        prefixes = _pick_prefixes(
            by_section.get(section, []),
            exclude_id=qid,
            rng=rng,
            min_n=prefix_min,
            max_n=prefix_max,
        )
        if not prefixes:
            top = _top_level_section(section)
            prefixes = _pick_prefixes(
                by_top_section.get(top, []),
                exclude_id=qid,
                rng=rng,
                min_n=prefix_min,
                max_n=prefix_max,
            )
            if prefixes:
                stats["used_top_level_fallback"] += 1
            else:
                stats["no_prefix_same_section"] += 1

        dialog = format_dialog(prefixes, final_q)
        train_rows.append(
            {
                "question_id": qid,
                "question": dialog,
                "gold_doc_ids": list(row["gold_doc_ids"]),
                "answer": meta.get("answer", ""),
                "section": section,
                "prefix_question_ids": [p["question_id"] for p in prefixes],
                "is_simple": False,
            }
        )
        stats["train_built"] += 1

    return train_rows, stats


def compose(
    *,
    eval_path: Path,
    index_dir: Path,
    raw_path: Path,
    out_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    prefix_min: int = DEFAULT_PREFIX_MIN,
    prefix_max: int = DEFAULT_PREFIX_MAX,
    seed: int = 0,
    device: str = "cuda:0",
    batch_size: int = 64,
    write_grpo: bool = False,
    prompt_version: str = "v2_search_only",
    val_frac: float = 0.05,
) -> dict[str, Any]:
    eval_rows = load_jsonl(eval_path)
    faq_meta = load_faq_meta(raw_path)

    loaded = FaissIndex.load(index_dir, use_gpu=False)
    embedder = E5Embedder(
        name=str(loaded.config.get("embedder", "BAAI/bge-m3")),
        device=device,
        query_prefix="",
        passage_prefix="",
        pooling="cls",
        max_length=8192,
    )
    log.info("loaded index from %s (%d docs)", index_dir, loaded.num_docs)

    simple_rows, complex_rows, cls_stats = classify_retrieval(
        eval_rows,
        loaded,
        embedder,
        top_k=top_k,
        batch_size=batch_size,
    )

    for row in simple_rows:
        meta = faq_meta.get(row["question_id"], {})
        row["section"] = meta.get("section", "")
        row["answer"] = meta.get("answer", "")

    train_rows, train_stats = build_train_rows(
        simple_rows,
        complex_rows,
        faq_meta,
        seed=seed,
        prefix_min=prefix_min,
        prefix_max=prefix_max,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(simple_rows, out_dir / "simple.jsonl")
    write_jsonl(train_rows, out_dir / "train.jsonl")

    grpo_stats: dict[str, int] = {}
    if write_grpo:
        from grpo.data_prep import build as build_grpo_parquet

        n_train, n_val = build_grpo_parquet(
            out_dir / "train.jsonl",
            out_dir / "grpo_train.parquet",
            out_dir / "grpo_val.parquet",
            prompt_version=prompt_version,
            data_source="sbol_retrieval",
            val_frac=val_frac,
            seed=seed,
        )
        grpo_stats = {"grpo_train": n_train, "grpo_val": n_val}

    summary: dict[str, Any] = {
        "eval_path": str(eval_path),
        "index_dir": str(index_dir),
        "raw_path": str(raw_path),
        "out_dir": str(out_dir),
        "top_k": top_k,
        "prefix_min": prefix_min,
        "prefix_max": prefix_max,
        "seed": seed,
        "classification": cls_stats,
        "train_build": train_stats,
        **grpo_stats,
    }
    stats_path = out_dir / "compose_sbol_stats.json"
    stats_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info(
        "simple=%d complex=%d train=%d -> %s",
        cls_stats["simple"],
        cls_stats["complex"],
        train_stats["train_built"],
        out_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("data/processed/sbol/eval.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/sbol"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/sbol/faq_index_28_apr.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/sbol"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--prefix-min", type=int, default=DEFAULT_PREFIX_MIN)
    parser.add_argument("--prefix-max", type=int, default=DEFAULT_PREFIX_MAX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--write-grpo", action="store_true")
    parser.add_argument("--prompt-version", default="v2_search_only")
    parser.add_argument("--val-frac", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    compose(
        eval_path=args.eval,
        index_dir=args.index_dir,
        raw_path=args.raw,
        out_dir=args.out_dir,
        top_k=args.top_k,
        prefix_min=args.prefix_min,
        prefix_max=args.prefix_max,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        write_grpo=args.write_grpo,
        prompt_version=args.prompt_version,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
