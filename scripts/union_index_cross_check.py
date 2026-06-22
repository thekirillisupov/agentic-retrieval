"""
Union two FAISS indexes (sbol + musique) and estimate cross-index contamination
in top-k=10 results for both eval sets.

For each query from eval_sbol:   how many of top-10 passages are from musique?
For each query from eval_musique: how many of top-10 passages are from sbol?
"""

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tool_server.index import FaissIndex
from tool_server.embedder import E5Embedder

# ── Config ────────────────────────────────────────────────────────────────────
SBOL_INDEX_DIR   = ROOT / "indexes/sbol"
MUSIQUE_INDEX_DIR = ROOT / "indexes/musique"
SBOL_EVAL        = ROOT / "data/processed/sbol/eval.jsonl"
MUSIQUE_EVAL     = ROOT / "data/processed/musique/musique_dev_eval.jsonl"
TOP_K            = 10
DEVICE           = "cuda:0"
EMBEDDER_NAME    = "BAAI/bge-m3"
BATCH_SIZE       = 64


def load_eval(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f]


def merge_indexes(idx_a: FaissIndex, idx_b: FaissIndex) -> tuple[faiss.Index, list]:
    """Reconstruct vectors from both indexes and build a single merged IndexFlatIP."""
    dim = idx_a.dim
    assert idx_b.dim == dim, f"Dim mismatch: {idx_a.dim} vs {idx_b.dim}"

    vecs_a = idx_a.index.reconstruct_n(0, idx_a.index.ntotal)
    vecs_b = idx_b.index.reconstruct_n(0, idx_b.index.ntotal)

    merged_vecs = np.vstack([vecs_a, vecs_b]).astype(np.float32)
    merged_meta = idx_a.metadata + idx_b.metadata  # DocMeta list

    merged_index = faiss.IndexFlatIP(dim)
    merged_index.add(merged_vecs)
    print(f"Merged index: {merged_index.ntotal} vectors "
          f"({idx_a.index.ntotal} sbol + {idx_b.index.ntotal} musique)")
    return merged_index, merged_meta


def encode_queries(embedder: E5Embedder, questions: list[str]) -> np.ndarray:
    t0 = time.time()
    embs = embedder.encode_queries(questions, batch_size=BATCH_SIZE)
    print(f"Encoded {len(questions)} queries in {time.time()-t0:.1f}s → shape {embs.shape}")
    return embs


def cross_index_stats(
    merged_index: faiss.Index,
    merged_meta: list,
    query_embs: np.ndarray,
    query_origin: str,   # "sbol" or "musique"
    top_k: int = TOP_K,
) -> dict:
    """
    Search merged index and count how many top-k hits come from the *other* dataset.
    Returns per-query counts and summary statistics.
    """
    scores, ids = merged_index.search(query_embs.astype(np.float32), top_k)

    other_prefix = "musique" if query_origin == "sbol" else "sbol"
    per_query_cross = []

    for row_ids in ids:
        cross = sum(
            1 for pos in row_ids
            if pos >= 0 and merged_meta[pos].doc_id.startswith(other_prefix)
        )
        per_query_cross.append(cross)

    arr = np.array(per_query_cross)
    return {
        "query_origin":        query_origin,
        "other_index":         other_prefix,
        "num_queries":         len(arr),
        "top_k":               top_k,
        "mean_cross_per_query": float(arr.mean()),
        "median_cross":        float(np.median(arr)),
        "pct_queries_any_cross": float((arr > 0).mean() * 100),
        "cross_distribution":  {
            str(k): int((arr == k).sum()) for k in range(top_k + 1) if (arr == k).sum() > 0
        },
    }


def main():
    print("Loading indexes…")
    idx_sbol    = FaissIndex.load(SBOL_INDEX_DIR)
    idx_musique = FaissIndex.load(MUSIQUE_INDEX_DIR)
    print(f"  sbol:    {idx_sbol.num_docs} docs")
    print(f"  musique: {idx_musique.num_docs} docs")

    merged_index, merged_meta = merge_indexes(idx_sbol, idx_musique)

    print("\nLoading embedder…")
    embedder = E5Embedder(
        name=EMBEDDER_NAME,
        device=DEVICE,
        query_prefix="",
        passage_prefix="",
        pooling="cls",
        max_length=512,
    )

    # ── SBOL eval ─────────────────────────────────────────────────────────────
    sbol_eval = load_eval(SBOL_EVAL)
    sbol_questions = [e["question"] for e in sbol_eval]
    print(f"\n[sbol eval] {len(sbol_questions)} questions")
    sbol_embs = encode_queries(embedder, sbol_questions)
    sbol_stats = cross_index_stats(merged_index, merged_meta, sbol_embs, "sbol")

    # ── MuSiQue eval ──────────────────────────────────────────────────────────
    musique_eval = load_eval(MUSIQUE_EVAL)
    musique_questions = [e["question"] for e in musique_eval]
    print(f"\n[musique eval] {len(musique_questions)} questions")
    musique_embs = encode_queries(embedder, musique_questions)
    musique_stats = cross_index_stats(merged_index, merged_meta, musique_embs, "musique")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for stats in [sbol_stats, musique_stats]:
        print(f"\nQueries from [{stats['query_origin']}] searching merged index (top_k={stats['top_k']})")
        print(f"  Passages from [{stats['other_index']}] in top-{stats['top_k']}:")
        print(f"    mean per query      : {stats['mean_cross_per_query']:.3f} / {stats['top_k']}")
        print(f"    median per query    : {stats['median_cross']:.1f}")
        print(f"    % queries with ≥1  : {stats['pct_queries_any_cross']:.1f}%")
        print(f"    distribution (# cross → # queries):")
        for k, cnt in sorted(stats["cross_distribution"].items(), key=lambda x: int(x[0])):
            bar = "█" * min(cnt // max(1, len(sbol_questions) // 50), 40)
            print(f"      {int(k):2d} cross: {cnt:5d} queries  {bar}")

    out_path = ROOT / "data/processed/union_cross_index_stats.json"
    with out_path.open("w") as f:
        json.dump({"sbol": sbol_stats, "musique": musique_stats}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved stats → {out_path}")


if __name__ == "__main__":
    main()
