"""Single-shot RAG baseline: one tool call, no agent.

If the agent doesn't beat this on NDCG@10, the bottleneck is the harness/prompt,
not the retriever or the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from agent.harness import ToolServerClient
from eval_.metrics import aggregate, evaluate_one

log = logging.getLogger(__name__)


def load_eval(path: Path, subset_size: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if subset_size is not None and subset_size > 0:
        rows = rows[:subset_size]
    return rows


def run(
    eval_path: Path,
    tool_url: str,
    *,
    top_k: int = 10,
    subset_size: int | None = None,
    timeout_s: float = 30.0,
) -> dict:
    eval_rows = load_eval(eval_path, subset_size=subset_size)
    tool = ToolServerClient(tool_url, timeout_s=timeout_s)

    per_example = []
    metrics = []
    try:
        for row in eval_rows:
            response = tool.local_search(query=row["question"], top_k=top_k)
            predicted = [r["doc_id"] for r in response.get("results", [])]
            gold = set(row["gold_doc_ids"])
            m = evaluate_one(predicted, gold, k=top_k)
            metrics.append(m)
            per_example.append(
                {
                    "question_id": row["question_id"],
                    "predicted": predicted,
                    "gold_doc_ids": row["gold_doc_ids"],
                    "ndcg": m.ndcg,
                    "recall": m.recall,
                    "precision": m.precision,
                }
            )
    finally:
        tool.close()

    agg = aggregate(metrics)
    return {
        "n": len(metrics),
        "top_k": top_k,
        "ndcg": agg.ndcg,
        "recall": agg.recall,
        "precision": agg.precision,
        "per_example": per_example,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--out", default="data/processed/baseline_results.json")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())

    eval_path = Path(args.eval_path or cfg["eval"]["eval_dataset_path"])
    top_k = args.top_k or cfg["eval"]["ndcg_k"]
    subset = cfg["eval"].get("subset_size")
    tool_url = cfg["tool_server"]["url"]
    timeout_s = cfg["tool_server"].get("timeout_s", 30)

    summary = run(
        eval_path,
        tool_url,
        top_k=top_k,
        subset_size=subset,
        timeout_s=timeout_s,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info(
        "baseline n=%d top_k=%d  ndcg@k=%.4f  recall@k=%.4f  precision@k=%.4f",
        summary["n"],
        summary["top_k"],
        summary["ndcg"],
        summary["recall"],
        summary["precision"],
    )


if __name__ == "__main__":
    main()
