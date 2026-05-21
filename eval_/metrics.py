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
    precision: float
    recall: float
    f1: float


def _dedup_ranked(predicted: list[str]) -> list[str]:
    """Return predicted with duplicates removed, preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in predicted:
        if doc_id not in seen:
            result.append(doc_id)
            seen.add(doc_id)
    return result


def simple_precision(predicted: list[str], gold: set[str]) -> float:
    predicted = _dedup_ranked(predicted)
    if not predicted:
        return 0.0
    hits = sum(1 for p in predicted if p in gold)
    return hits / len(predicted)


def simple_recall(predicted: list[str], gold: set[str]) -> float:
    predicted = _dedup_ranked(predicted)
    if not gold:
        return 0.0
    hits = sum(1 for p in predicted if p in gold)
    return hits / len(gold)


def simple_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F_beta score: (1+beta^2) * P * R / (beta^2 * P + R).

    beta>1 weights recall more (e.g. beta=2), beta<1 weights precision more.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def dcg_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(_dedup_ranked(predicted)[:k], start=1):
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
    # dcg_at_k already deduplicates; pass through for consistency
    return dcg_at_k(predicted, gold, k) / idcg


def evaluate_one(predicted: list[str], gold: set[str], k: int = 10) -> MetricResult:
    p = simple_precision(predicted, gold)
    r = simple_recall(predicted, gold)
    return MetricResult(
        ndcg=ndcg_at_k(predicted, gold, k),
        precision=p,
        recall=r,
        f1=simple_f1(p, r),
    )


def aggregate(results: list[MetricResult]) -> MetricResult:
    if not results:
        return MetricResult(0.0, 0.0, 0.0, 0.0)
    n = len(results)
    return MetricResult(
        ndcg=sum(r.ndcg for r in results) / n,
        precision=sum(r.precision for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        f1=sum(r.f1 for r in results) / n,
    )
