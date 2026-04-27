"""Retrieval metrics. Binary relevance — gold is a set of doc_ids per question.

Bare implementations (no sklearn) so the eval has no surprises around tie-breaking
or sort orders. Predictions arrive already in rank order; metrics treat them that
way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MetricResult:
    ndcg: float
    recall: float
    precision: float


def precision_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = predicted[:k]
    if not topk:
        return 0.0
    hits = sum(1 for p in topk if p in gold)
    return hits / k


def recall_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for p in predicted[:k] if p in gold)
    return hits / len(gold)


def dcg_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(predicted[:k], start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def idcg_at_k(num_gold: int, k: int) -> float:
    n = min(num_gold, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, n + 1))


def ndcg_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    idcg = idcg_at_k(len(gold), k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(predicted, gold, k) / idcg


def evaluate_one(
    predicted: list[str], gold: set[str], k: int = 10
) -> MetricResult:
    return MetricResult(
        ndcg=ndcg_at_k(predicted, gold, k),
        recall=recall_at_k(predicted, gold, k),
        precision=precision_at_k(predicted, gold, k),
    )


def aggregate(results: list[MetricResult]) -> MetricResult:
    if not results:
        return MetricResult(0.0, 0.0, 0.0)
    n = len(results)
    return MetricResult(
        ndcg=sum(r.ndcg for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        precision=sum(r.precision for r in results) / n,
    )
