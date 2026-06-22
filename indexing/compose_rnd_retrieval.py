"""Build rnd FAISS index and filter MIRACL / Mr.TyDi training data.

1. Collect unique passages from ru/en MIRACL and ru/en Mr.TyDi JSONL files.
2. Encode with BGE-M3 and write indexes/rnd_index/ (union of all four sources).
3. Keep only queries whose positive passage is NOT in retrieval top-k (default k=3).
4. Write data/processed/rnd/ artifacts (like musique/):
   - rnd_train.jsonl — filtered rows with gold_doc_ids
   - grpo_train.parquet / grpo_val.parquet — veRL-ready (10% val by default)
   - per-source filtered *.jsonl (optional)

Usage (from agentic-retrieval repo root, conda env cu129py312):

    python -m indexing.compose_rnd_retrieval \\
        --input-dir /home/jovyan/isupov/data/giga_emb_bank_sft \\
        --processed-dir data/processed/rnd \\
        --index-dir indexes/rnd_index \\
        --top-k 3 --skip-index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

from indexing.embedder_batch import encode_passages_streaming
from tool_server.embedder import E5Embedder
from tool_server.index import DocMeta, build_index, save_index

log = logging.getLogger(__name__)

DATASET_FILES = (
    "ru_miracl.jsonl",
    "en_miracl.jsonl",
    "ru_mr_tydi.jsonl",
    "en_mr_tydi.jsonl",
)

# Query instructions aligned with gigachat-embeddings/configs/instruct_bank_sft.json
QUERY_INSTRUCTIONS: dict[str, str] = {
    "ru_miracl": "Найди абзац текста с ответом на вопрос",
    "en_miracl": "Найди абзац текста с ответом на вопрос",
    "ru_mr_tydi": "Найди абзац текста с ответом на вопрос",
    "en_mr_tydi": "Найди абзац текста с ответом на вопрос",
}

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_TOP_K = 3
DEFAULT_VAL_FRAC = 0.1
DEFAULT_PROCESSED_DIR = Path("data/processed/rnd")
RND_DATA_SOURCE = "rnd_retrieval"


def _dataset_key(filename: str) -> str:
    return Path(filename).stem


def _doc_id(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"rnd_{h}"


def _title_from_passage(text: str) -> str:
    if "\n\n" in text:
        return text.split("\n\n", 1)[0].strip()
    if "\n" in text:
        return text.split("\n", 1)[0].strip()
    return text[:120].strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_corpus(
    input_dir: Path,
    files: tuple[str, ...] = DATASET_FILES,
) -> tuple[list[DocMeta], dict[str, int]]:
    """Deduplicate passages across datasets; preserve first-seen doc_id."""
    text_to_pos: dict[str, int] = {}
    metadata: list[DocMeta] = []
    per_source: dict[str, int] = {}

    for fname in files:
        path = input_dir / fname
        if not path.exists():
            raise FileNotFoundError(path)
        key = _dataset_key(fname)
        added = 0
        for row in load_jsonl(path):
            for text in row.get("pos", []) + row.get("neg", []):
                if not text or text in text_to_pos:
                    continue
                doc_id = _doc_id(text)
                metadata.append(
                    DocMeta(doc_id=doc_id, title=_title_from_passage(text), text=text)
                )
                text_to_pos[text] = len(metadata) - 1
                added += 1
        per_source[key] = added
        log.info("%s: %d new unique passages", key, added)

    log.info("corpus total: %d unique passages", len(metadata))
    return metadata, per_source


def build_rnd_index(
    corpus: list[DocMeta],
    out_dir: Path,
    *,
    embedder_name: str = DEFAULT_EMBEDDER,
    device: str = "cuda:0",
    batch_size: int = 32,
    max_length: int = 8192,
) -> E5Embedder:
    embedder = E5Embedder(
        name=embedder_name,
        device=device,
        max_length=max_length,
        query_prefix="",
        passage_prefix="",
        pooling="cls",
    )
    texts = [m.text for m in corpus]
    embeddings = encode_passages_streaming(embedder, texts, batch_size=batch_size)
    index = build_index(dim=embedder.dim, use_gpu=False)
    index.add(embeddings)
    save_index(
        index,
        corpus,
        out_dir=out_dir,
        embedder_name=embedder.name,
        dim=embedder.dim,
    )
    stats = {
        "out_dir": str(out_dir),
        "embedder": embedder.name,
        "dim": embedder.dim,
        "num_docs": len(corpus),
        "sources": list(QUERY_INSTRUCTIONS.keys()),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("wrote rnd index: %d docs -> %s", len(corpus), out_dir)
    return embedder


def _format_query(query: str, dataset_key: str) -> str:
    instruct = QUERY_INSTRUCTIONS.get(dataset_key, "")
    if instruct:
        return f"{instruct}\n{query}"
    return query


def _has_relevant_in_topk(
    pos_texts: list[str],
    retrieved_texts: list[str],
    top_k: int,
) -> bool:
    pos_set = set(pos_texts)
    for text in retrieved_texts[:top_k]:
        if text in pos_set:
            return True
    return False


def filter_dataset(
    rows: list[dict[str, Any]],
    embedder: E5Embedder,
    text_to_pos: dict[str, int],
    corpus_texts: list[str],
    index: faiss.Index,
    *,
    dataset_key: str,
    top_k: int,
    query_batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats = {"total": len(rows), "kept": 0, "dropped_has_pos_in_topk": 0, "dropped_no_pos": 0}

    queries = [_format_query(r["query"], dataset_key) for r in rows]
    query_embs = embedder.encode_queries(queries, batch_size=query_batch_size)

    for i, row in enumerate(tqdm(rows, desc=f"filter {dataset_key}", leave=False)):
        pos = row.get("pos") or []
        if not pos:
            stats["dropped_no_pos"] += 1
            continue

        q = query_embs[i : i + 1].astype(np.float32)
        scores, ids = index.search(q, top_k)
        retrieved = []
        for idx in ids[0]:
            if idx < 0:
                continue
            retrieved.append(corpus_texts[idx])

        if _has_relevant_in_topk(pos, retrieved, top_k):
            stats["dropped_has_pos_in_topk"] += 1
            continue

        out = {
            "query": row["query"],
            "pos": pos,
            "neg": row.get("neg") or [],
        }
        kept.append(out)
        stats["kept"] += 1

    return kept, stats


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_train_records(
    kept: list[dict[str, Any]],
    *,
    source: str,
    text_to_doc_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Map filtered query/pos/neg rows to musique-style train jsonl records."""
    records: list[dict[str, Any]] = []
    for i, row in enumerate(kept):
        pos_texts = row.get("pos") or []
        gold = []
        for text in pos_texts:
            doc_id = text_to_doc_id.get(text)
            if doc_id and doc_id not in gold:
                gold.append(doc_id)
        if not gold:
            continue
        qid = f"{source}:{i}"
        records.append(
            {
                "question_id": qid,
                "question": row["query"],
                "answer": "",
                "gold_doc_ids": gold,
                "source": source,
            }
        )
    return records


def export_corpus_jsonl(corpus: list[DocMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in corpus:
            f.write(
                json.dumps(
                    {"doc_id": m.doc_id, "title": m.title, "text": m.text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_grpo_parquet(
    train_records: list[dict[str, Any]],
    processed_dir: Path,
    *,
    val_frac: float,
    seed: int,
    prompt_version: str,
) -> tuple[int, int]:
    from grpo.data_prep import build as build_grpo_parquet

    tmp_jsonl = processed_dir / "rnd_train.jsonl"
    write_jsonl(train_records, tmp_jsonl)
    n_train, n_val = build_grpo_parquet(
        tmp_jsonl,
        processed_dir / "grpo_train.parquet",
        processed_dir / "grpo_val.parquet",
        prompt_version=prompt_version,
        data_source=RND_DATA_SOURCE,
        val_frac=val_frac,
        seed=seed,
    )
    return n_train, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/jovyan/isupov/data/giga_emb_bank_sft"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output dir (musique-style): rnd_train.jsonl, grpo_{train,val}.parquet",
    )
    parser.add_argument(
        "--out-data-dir",
        type=Path,
        default=None,
        help="Optional: also write per-source filtered query/pos/neg JSONL here",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("indexes/rnd_index"),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Reuse existing faiss.index + metadata.jsonl under --index-dir",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Only build the index, do not write filtered outputs",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Validation fraction of filtered rows (default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-version", default="v1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    corpus, per_source = collect_corpus(args.input_dir)
    corpus_texts = [m.text for m in corpus]
    text_to_pos = {t: i for i, t in enumerate(corpus_texts)}
    text_to_doc_id = {m.text: m.doc_id for m in corpus}

    index_dir = args.index_dir
    if args.skip_index:
        import faiss as _faiss

        from tool_server.index import FaissIndex

        loaded = FaissIndex.load(index_dir, use_gpu=False)
        index = loaded.index
        embedder = E5Embedder(
            name=str(loaded.config.get("embedder", args.embedder)),
            device=args.device,
            pooling="cls",
        )
        log.info("loaded existing index from %s (%d docs)", index_dir, loaded.num_docs)
    else:
        embedder = build_rnd_index(
            corpus,
            index_dir,
            embedder_name=args.embedder,
            device=args.device,
            batch_size=args.batch_size,
        )
        index = faiss.read_index(str(index_dir / "faiss.index"))

    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_corpus_jsonl(corpus, processed_dir / "corpus.jsonl")

    if args.skip_filter:
        return

    all_filter_stats: dict[str, Any] = {}
    all_train_records: list[dict[str, Any]] = []
    for fname in DATASET_FILES:
        key = _dataset_key(fname)
        rows = load_jsonl(args.input_dir / fname)
        filtered, fstats = filter_dataset(
            rows,
            embedder,
            text_to_pos,
            corpus_texts,
            index,
            dataset_key=key,
            top_k=args.top_k,
            query_batch_size=args.query_batch_size,
        )
        all_filter_stats[key] = fstats
        records = to_train_records(
            filtered, source=key, text_to_doc_id=text_to_doc_id
        )
        all_train_records.extend(records)
        fstats["train_records"] = len(records)

        if args.out_data_dir is not None:
            write_jsonl(filtered, args.out_data_dir / fname)

        log.info(
            "%s: kept %d / %d (dropped top-%d hits: %d, no pos: %d), "
            "%d train records",
            key,
            fstats["kept"],
            fstats["total"],
            args.top_k,
            fstats["dropped_has_pos_in_topk"],
            fstats["dropped_no_pos"],
            len(records),
        )

    n_train, n_val = write_grpo_parquet(
        all_train_records,
        processed_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        prompt_version=args.prompt_version,
    )

    summary = {
        "input_dir": str(args.input_dir),
        "processed_dir": str(processed_dir),
        "out_data_dir": str(args.out_data_dir) if args.out_data_dir else None,
        "index_dir": str(index_dir),
        "top_k": args.top_k,
        "val_frac": args.val_frac,
        "embedder": args.embedder,
        "corpus_per_source": per_source,
        "corpus_total": len(corpus),
        "filter": all_filter_stats,
        "combined_train_records": len(all_train_records),
        "grpo_train_rows": n_train,
        "grpo_val_rows": n_val,
    }
    summary_path = processed_dir / "compose_rnd_stats.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info(
        "wrote %d train + %d val parquet rows under %s; summary %s",
        n_train,
        n_val,
        processed_dir,
        summary_path,
    )


if __name__ == "__main__":
    main()
