"""Build the faiss index from corpus.jsonl.

Reads corpus.jsonl produced by parse_musique, encodes all passages in batches,
and writes faiss.index + metadata.jsonl + config.json under --out-dir.

The corpus is read fully into memory — at MuSiQue scale this is fine (~50k rows).
For larger corpora switch to a streaming write of metadata.jsonl + chunked
embedding append; the embedder's batched API already handles the GPU side.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from indexing.embedder_batch import encode_passages_streaming
from tool_server.embedder import E5Embedder
from tool_server.index import DocMeta, build_index, save_index

log = logging.getLogger(__name__)


def load_corpus(path: Path) -> list[DocMeta]:
    rows: list[DocMeta] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                DocMeta(
                    d["doc_id"],
                    d["title"],
                    d["text"],
                    d.get("file_name"),
                    d.get("index"),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus", default="data/processed/musique/corpus.jsonl")
    parser.add_argument("--out-dir", default=None, help="overrides index.dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    emb_cfg = cfg["embedder"]
    idx_cfg = cfg["index"]
    out_dir = Path(args.out_dir or idx_cfg["dir"])

    corpus = load_corpus(Path(args.corpus))
    log.info("loaded %d passages from %s", len(corpus), args.corpus)

    embedder = E5Embedder(
        name=emb_cfg["name"],
        device=emb_cfg["device"],
        max_length=emb_cfg["max_length"],
        query_prefix=emb_cfg["query_prefix"],
        passage_prefix=emb_cfg["passage_prefix"],
        pooling=emb_cfg.get("pooling", "mean"),
    )

    # Truncation sanity check: the spec calls out >1% truncation as a smell.
    sample = [m.text for m in corpus[: min(len(corpus), 2000)]]
    truncated = embedder.count_truncated(sample)
    pct = 100 * truncated / max(1, len(sample))
    log.info("truncation sample: %d/%d (%.2f%%)", truncated, len(sample), pct)
    if pct > 1.0:
        log.warning("> 1%% of passages truncated — consider chunking or longer embedder")

    texts = [m.text for m in corpus]
    embeddings = encode_passages_streaming(
        embedder, texts, batch_size=emb_cfg["batch_size"]
    )

    index = build_index(
        dim=embedder.dim,
        use_gpu=idx_cfg.get("use_gpu", False),
        gpu_id=idx_cfg.get("gpu_id", 3),
    )
    index.add(embeddings)

    save_index(
        index,
        corpus,
        out_dir=out_dir,
        embedder_name=embedder.name,
        dim=embedder.dim,
    )
    log.info("index built: %d vectors, dim=%d, dir=%s", index.ntotal, embedder.dim, out_dir)


if __name__ == "__main__":
    main()
