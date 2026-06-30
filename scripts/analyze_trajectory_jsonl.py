#!/usr/bin/env python3
"""Recall stats for GRPO trajectory JSONL dumps (step checkpoints).

Computes per-index:
  - recall@k on the final <answer> list
  - trajectory recall: gold docs seen anywhere in tool outputs (even if dropped
    from the final answer)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from agent.parser import parse_answer
from eval_.metrics import simple_recall

CKR_DOC_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+$",
    re.IGNORECASE,
)


def detect_index(gold_ids: list[str]) -> str:
    for doc_id in gold_ids:
        if doc_id.startswith("musique_"):
            return "musique"
        if doc_id.startswith("sbol_"):
            return "sbol"
        if doc_id.startswith("rnd_"):
            return "rnd"
        if doc_id.startswith("ckr_") or CKR_DOC_RE.match(doc_id):
            return "ckr"
    return "unknown"


def dedup_preserve_order(doc_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
    return out


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(dedup_preserve_order(ranked)[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def retrieved_doc_ids(messages_full: list[dict]) -> list[str]:
    """All doc_ids returned by search tools, in encounter order."""
    retrieved: list[str] = []
    seen: set[str] = set()
    for msg in messages_full:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("doc_id")
            if isinstance(doc_id, str) and doc_id and doc_id not in seen:
                retrieved.append(doc_id)
                seen.add(doc_id)
    return retrieved


def final_ranked_doc_ids(messages_full: list[dict]) -> list[str]:
    for msg in reversed(messages_full):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        answer = parse_answer(content)
        if answer:
            return answer
    return []


@dataclass
class IndexStats:
    n: int = 0
    n_gold: int = 0
    answer_r3_sum: float = 0.0
    answer_r5_sum: float = 0.0
    answer_r3_hits: int = 0
    answer_r5_hits: int = 0
    traj_r_sum: float = 0.0
    traj_r_hits: int = 0
    full_traj_recall: int = 0
    any_traj_recall: int = 0
    seen_not_in_answer: int = 0
    seen_in_answer: int = 0
    not_seen: int = 0

    def add(
        self,
        *,
        gold: list[str],
        ranked: list[str],
        retrieved: list[str],
    ) -> None:
        gold_set = set(gold)
        n_gold = len(gold_set)
        retrieved_set = set(retrieved)
        ranked_set = set(ranked)

        self.n += 1
        self.n_gold += n_gold
        self.answer_r3_sum += recall_at_k(ranked, gold_set, 3)
        self.answer_r5_sum += recall_at_k(ranked, gold_set, 5)
        self.answer_r3_hits += sum(1 for g in gold_set if g in set(dedup_preserve_order(ranked)[:3]))
        self.answer_r5_hits += sum(1 for g in gold_set if g in set(dedup_preserve_order(ranked)[:5]))
        self.traj_r_sum += simple_recall(retrieved, gold_set)
        self.traj_r_hits += sum(1 for g in gold_set if g in retrieved_set)
        if n_gold and gold_set <= retrieved_set:
            self.full_traj_recall += 1
        if gold_set & retrieved_set:
            self.any_traj_recall += 1

        for g in gold_set:
            if g not in retrieved_set:
                self.not_seen += 1
            elif g in ranked_set:
                self.seen_in_answer += 1
            else:
                self.seen_not_in_answer += 1


def summarize(stats: IndexStats) -> dict[str, float | int]:
    n = stats.n
    ng = stats.n_gold
    return {
        "n": n,
        "n_gold": ng,
        "answer_recall@3_macro": stats.answer_r3_sum / n if n else 0.0,
        "answer_recall@5_macro": stats.answer_r5_sum / n if n else 0.0,
        "answer_recall@3_micro": stats.answer_r3_hits / ng if ng else 0.0,
        "answer_recall@5_micro": stats.answer_r5_hits / ng if ng else 0.0,
        "trajectory_recall_macro": stats.traj_r_sum / n if n else 0.0,
        "trajectory_recall_micro": stats.traj_r_hits / ng if ng else 0.0,
        "trajectory_full_recall_pct": 100.0 * stats.full_traj_recall / n if n else 0.0,
        "trajectory_any_recall_pct": 100.0 * stats.any_traj_recall / n if n else 0.0,
        "gold_seen_in_answer_pct": 100.0 * stats.seen_in_answer / ng if ng else 0.0,
        "gold_seen_not_in_answer_pct": 100.0 * stats.seen_not_in_answer / ng if ng else 0.0,
        "gold_not_seen_pct": 100.0 * stats.not_seen / ng if ng else 0.0,
    }


def analyze_jsonl(path: Path) -> dict[str, IndexStats]:
    by_index: dict[str, IndexStats] = defaultdict(IndexStats)
    unknown = 0

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip line {line_no}: {exc}", file=sys.stderr)
                continue

            gold = list(row.get("gold_ids") or [])
            if not gold:
                continue

            index = detect_index(gold)
            if index == "unknown":
                unknown += 1

            messages = row.get("messages_full") or []
            ranked = final_ranked_doc_ids(messages)
            retrieved = retrieved_doc_ids(messages)
            by_index[index].add(gold=gold, ranked=ranked, retrieved=retrieved)

    if unknown:
        print(f"warning: {unknown} trajectories with unknown index", file=sys.stderr)
    return by_index


def print_report(path: Path, by_index: dict[str, IndexStats]) -> None:
    total_n = sum(s.n for s in by_index.values())
    print(f"File: {path}")
    print(f"Trajectories: {total_n}")
    print()

    header = (
        f"{'index':<8} {'n':>5}  "
        f"{'ans@3':>7} {'ans@5':>7}  "
        f"{'traj_r':>7} {'full%':>6} {'any%':>6}  "
        f"{'in_ans%':>8} {'tool_only%':>10} {'miss%':>6}"
    )
    print(header)
    print("-" * len(header))

    for index in sorted(by_index):
        s = summarize(by_index[index])
        print(
            f"{index:<8} {s['n']:>5}  "
            f"{s['answer_recall@3_macro']:>6.1%} {s['answer_recall@5_macro']:>6.1%}  "
            f"{s['trajectory_recall_macro']:>6.1%} {s['trajectory_full_recall_pct']:>5.1f}% {s['trajectory_any_recall_pct']:>5.1f}%  "
            f"{s['gold_seen_in_answer_pct']:>7.1f}% {s['gold_seen_not_in_answer_pct']:>9.1f}% {s['gold_not_seen_pct']:>5.1f}%"
        )

    print()
    print("Columns:")
    print("  ans@3, ans@5     macro recall@k on final <answer> (mean per trajectory)")
    print("  traj_r           macro trajectory recall (gold in any tool output)")
    print("  full% / any%     % trajectories with all / any gold retrieved in tools")
    print("  in_ans%          % gold docs in final answer")
    print("  tool_only%       % gold docs seen in tools but omitted from final answer")
    print("  miss%            % gold docs never returned by search")

    print()
    print("Micro-averaged (per gold doc):")
    for index in sorted(by_index):
        s = summarize(by_index[index])
        print(
            f"  {index}: answer@3={s['answer_recall@3_micro']:.1%}, "
            f"answer@5={s['answer_recall@5_micro']:.1%}, "
            f"trajectory={s['trajectory_recall_micro']:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Trajectory JSONL file (e.g. trajectories_data/.../65.jsonl)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    if not args.jsonl.is_file():
        print(f"error: not a file: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    by_index = analyze_jsonl(args.jsonl)
    if args.json:
        payload = {idx: summarize(stats) for idx, stats in sorted(by_index.items())}
        print(json.dumps({"file": str(args.jsonl), "by_index": payload}, indent=2))
    else:
        print_report(args.jsonl, by_index)


if __name__ == "__main__":
    main()
