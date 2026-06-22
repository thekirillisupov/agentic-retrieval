"""Build FAISS index from data/raw/ckr/ckr_index.jsonl.

Each line in ckr_index.jsonl has:
    file_name   -- source HTML filename (adjacency key)
    index       -- 0-based chunk position within that file
    raw_text    -- passage text
    title       -- breadcrumb-style document title
    document_id -- UUID identifying the source document

The script maps these to DocMeta so that get_neighbours works:
    doc_id    = "{document_id}_{index}"  (unique per chunk)
    title     = title
    text      = raw_text
    file_name = file_name
    index     = index

Usage (from agentic-retrieval repo root):

    python -m indexing.compose_ckr_index \\
        --input  data/raw/ckr/ckr_index.jsonl \\
        --out-dir indexes/ckr_index \\
        --device cuda:0 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from indexing.embedder_batch import encode_passages_streaming
from tool_server.embedder import E5Embedder
from tool_server.index import DocMeta, build_index, save_index

log = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_OUT_DIR = Path("indexes/ckr_index")


def load_ckr_corpus(path: Path) -> list[DocMeta]:
    rows: list[DocMeta] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = f"{d['document_id']}_{d['index']}"
            rows.append(
                DocMeta(
                    doc_id=doc_id,
                    title=d["title"],
                    text=d["raw_text"],
                    file_name=d["file_name"],
                    index=d["index"],
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/ckr/ckr_index.jsonl"),
        help="Path to ckr_index.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write faiss.index / metadata.jsonl / config.json",
    )
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    corpus = load_ckr_corpus(args.input)
    log.info("loaded %d chunks from %s", len(corpus), args.input)

    embedder = E5Embedder(
        name=args.embedder,
        device=args.device,
        max_length=args.max_length,
        query_prefix="",
        passage_prefix="",
        pooling="cls",
    )

    sample = [m.text for m in corpus[: min(len(corpus), 2000)]]
    truncated = embedder.count_truncated(sample)
    pct = 100 * truncated / max(1, len(sample))
    log.info("truncation sample: %d/%d (%.2f%%)", truncated, len(sample), pct)
    if pct > 1.0:
        log.warning("> 1%% of passages truncated — consider shorter chunks or larger max_length")

    texts = [m.text for m in corpus]
    embeddings = encode_passages_streaming(embedder, texts, batch_size=args.batch_size)

    index = build_index(dim=embedder.dim, use_gpu=False)
    index.add(embeddings)

    save_index(
        index,
        corpus,
        out_dir=args.out_dir,
        embedder_name=embedder.name,
        dim=embedder.dim,
    )
    log.info(
        "index built: %d vectors, dim=%d -> %s",
        index.ntotal,
        embedder.dim,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
