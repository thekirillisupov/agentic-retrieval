#!/usr/bin/env python3
"""Find trajectory JSONs where ranked_doc_ids != gold_doc_ids."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def detect_domain(doc_ids: list[str]) -> str | None:
    """Infer domain from doc_id prefix: musique or sbol."""
    for doc_id in doc_ids:
        if doc_id.startswith("musique_"):
            return "musique"
        if doc_id.startswith("sbol_"):
            return "sbol"
    return None


def get_retrieved_doc_ids(data: dict) -> set[str]:
    """All unique doc_ids returned by local_search across the trajectory."""
    retrieved: set[str] = set()
    for tc in data.get("tool_calls", []):
        for doc_id in tc.get("result_summary", {}).get("top_doc_ids", []):
            retrieved.add(doc_id)
    return retrieved


def get_tool_usage_stats(data: dict) -> dict[str, int]:
    """Count tool usage in a trajectory by tool name."""
    counts: Counter[str] = Counter()
    for tc in data.get("tool_calls", []):
        tool_name = tc.get("tool")
        if isinstance(tool_name, str) and tool_name:
            counts[tool_name] += 1
    return dict(sorted(counts.items()))


def gold_retrieval_stats(gold: list[str], retrieved: set[str], ranked: list[str]) -> dict:
    ranked_set = set(ranked)
    not_seen = [d for d in gold if d not in retrieved]
    seen = [d for d in gold if d in retrieved]
    seen_in_answer = [d for d in seen if d in ranked_set]
    seen_missing_from_answer = [d for d in seen if d not in ranked_set]
    n_gold = len(gold)
    return {
        "gold_seen": seen,
        "gold_missed": not_seen,
        "gold_seen_in_answer": seen_in_answer,
        "gold_seen_missing_from_answer": seen_missing_from_answer,
        "n_gold": n_gold,
        "n_seen": len(seen),
        "n_not_seen": len(not_seen),
        "n_seen_in_answer": len(seen_in_answer),
        "n_seen_missing_from_answer": len(seen_missing_from_answer),
        "recall": len(seen) / n_gold if n_gold else 0.0,
        "full_recall": len(seen) == n_gold if n_gold else False,
        "any_seen": len(seen) > 0,
    }


def analyze_trajectories(directory: str, domain: str | None = None) -> tuple[list[dict], list[dict], int, int]:
    dirpath = Path(directory)
    if not dirpath.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_trajectories: list[dict] = []
    mismatches: list[dict] = []
    total = 0
    filtered_out = 0

    for fpath in sorted(dirpath.glob("*.json")):
        with open(fpath) as f:
            data = json.load(f)

        total += 1
        output = data.get("output", {})
        ranked = output.get("ranked_doc_ids", [])
        gold = output.get("gold_doc_ids", [])
        traj_domain = detect_domain(gold) or detect_domain(ranked)

        if domain and traj_domain != domain:
            filtered_out += 1
            continue

        retrieved = get_retrieved_doc_ids(data)
        tool_usage = get_tool_usage_stats(data)
        ret_stats = gold_retrieval_stats(gold, retrieved, ranked)

        entry = {
            "file": fpath.name,
            "trajectory_id": data.get("trajectory_id"),
            "domain": traj_domain,
            "query": data.get("input", {}).get("messages", [{}])[0].get("content", ""),
            "ranked_doc_ids": ranked,
            "gold_doc_ids": gold,
            "retrieved_doc_ids": sorted(retrieved),
            "tools_used": list(tool_usage.keys()),
            "tool_counts": tool_usage,
            "num_tool_calls": sum(tool_usage.values()),
            "missing_from_ranked": [d for d in gold if d not in ranked],
            "extra_in_ranked": [d for d in ranked if d not in gold],
            **ret_stats,
        }
        all_trajectories.append(entry)

        if ranked != gold:
            mismatches.append(entry)

    return all_trajectories, mismatches, total, filtered_out


def print_retrieval_stats(trajectories: list[dict]) -> None:
    if not trajectories:
        return

    n = len(trajectories)
    total_gold = sum(t["n_gold"] for t in trajectories)
    total_seen = sum(t["n_seen"] for t in trajectories)
    full_recall = sum(1 for t in trajectories if t["full_recall"])
    any_seen = sum(1 for t in trajectories if t["any_seen"])
    zero_seen = sum(1 for t in trajectories if not t["any_seen"])

    print("Gold-in-retrieval (model saw relevant doc via local_search):")
    print(f"  Trajectories with all gold docs retrieved: {full_recall}/{n}  ({full_recall / n * 100:.1f}%)")
    print(f"  Trajectories with at least 1 gold doc seen:  {any_seen}/{n}  ({any_seen / n * 100:.1f}%)")
    print(f"  Trajectories with 0 gold docs seen:          {zero_seen}/{n}  ({zero_seen / n * 100:.1f}%)")
    if total_gold:
        print(f"  Gold docs seen (micro):                    {total_seen}/{total_gold}  ({total_seen / total_gold * 100:.1f}%)")
        print(f"  Mean gold recall per trajectory:         {sum(t['recall'] for t in trajectories) / n * 100:.1f}%")

    # Per-recall bucket (e.g. 0/2, 1/2, 2/2)
    buckets: dict[str, int] = {}
    for t in trajectories:
        key = f"{t['n_seen']}/{t['n_gold']}"
        buckets[key] = buckets.get(key, 0) + 1
    print("  Recall distribution:", ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))

    by_domain: dict[str, list[dict]] = {}
    for t in trajectories:
        d = t["domain"] or "unknown"
        by_domain.setdefault(d, []).append(t)

    if len(by_domain) > 1:
        print("  By domain:")
        for d, ts in sorted(by_domain.items()):
            dn = len(ts)
            d_gold = sum(t["n_gold"] for t in ts)
            d_seen = sum(t["n_seen"] for t in ts)
            d_full = sum(1 for t in ts if t["full_recall"])
            d_any = sum(1 for t in ts if t["any_seen"])
            micro = f"{d_seen / d_gold * 100:.1f}%" if d_gold else "n/a"
            print(
                f"    {d}: full={d_full}/{dn} ({d_full / dn * 100:.1f}%), "
                f"any={d_any}/{dn} ({d_any / dn * 100:.1f}%), "
                f"gold_seen={d_seen}/{d_gold} ({micro})"
            )


def print_final_gold_outcomes(trajectories: list[dict]) -> None:
    """Print per-gold-doc outcome counts across all trajectories."""
    if not trajectories:
        return

    total_gold = sum(t["n_gold"] for t in trajectories)
    n_not_seen = sum(t["n_not_seen"] for t in trajectories)
    n_seen_in_answer = sum(t["n_seen_in_answer"] for t in trajectories)
    n_seen_missing = sum(t["n_seen_missing_from_answer"] for t in trajectories)

    def pct(x: int) -> str:
        return f"{x / total_gold * 100:.1f}%" if total_gold else "n/a"

    print("=" * 80)
    print("Final statistics (per gold document):")
    print(f"  Not seen in retrieval:                    {n_not_seen}/{total_gold}  ({pct(n_not_seen)})")
    print(f"  Seen and in final answer:                 {n_seen_in_answer}/{total_gold}  ({pct(n_seen_in_answer)})")
    print(f"  Seen but missing from final answer:       {n_seen_missing}/{total_gold}  ({pct(n_seen_missing)})")

    by_domain: dict[str, list[dict]] = {}
    for t in trajectories:
        d = t["domain"] or "unknown"
        by_domain.setdefault(d, []).append(t)

    if len(by_domain) > 1:
        print("  By domain:")
        for d, ts in sorted(by_domain.items()):
            dg = sum(t["n_gold"] for t in ts)
            print(
                f"    {d}: not_seen={sum(t['n_not_seen'] for t in ts)}/{dg}, "
                f"seen_in_answer={sum(t['n_seen_in_answer'] for t in ts)}/{dg}, "
                f"seen_missing={sum(t['n_seen_missing_from_answer'] for t in ts)}/{dg}"
            )


def print_tool_usage_stats(trajectories: list[dict]) -> None:
    """Print how often tools were used and which ones."""
    if not trajectories:
        return

    total_calls = sum(t["num_tool_calls"] for t in trajectories)
    aggregate: Counter[str] = Counter()
    for t in trajectories:
        aggregate.update(t["tool_counts"])

    print("Tool usage:")
    print(f"  Total tool calls: {total_calls}")
    if aggregate:
        print("  Tools and counts:", ", ".join(f"{tool}={cnt}" for tool, cnt in sorted(aggregate.items())))
    else:
        print("  Tools and counts: none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find trajectory JSONs where ranked_doc_ids != gold_doc_ids.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="trajectories_data/2026-05-25/170_steps",
        help="Directory with trajectory JSON files",
    )
    parser.add_argument(
        "--domain",
        choices=["musique", "sbol"],
        help="Only include trajectories from this domain",
    )
    parser.add_argument(
        "--mismatches-only",
        action="store_true",
        help="Only print per-trajectory details for ranked != gold cases",
    )
    args = parser.parse_args()

    all_trajectories, mismatches, total, filtered_out = analyze_trajectories(
        args.directory, args.domain
    )

    scanned = len(all_trajectories)
    print(f"Directory: {args.directory}")
    print(f"Total files: {total}")
    if args.domain:
        print(f"Domain filter: {args.domain}  (skipped {filtered_out} files)")
    print(f"Scanned: {scanned}")
    print("=" * 80)

    print_retrieval_stats(all_trajectories)
    print("=" * 80)
    print_tool_usage_stats(all_trajectories)

    print("=" * 80)
    print(f"Mismatches (ranked != gold): {len(mismatches)}  ({len(mismatches) / scanned * 100:.1f}% of scanned)" if scanned else "Mismatches: 0")

    by_domain: dict[str, int] = {}
    for m in mismatches:
        d = m["domain"] or "unknown"
        by_domain[d] = by_domain.get(d, 0) + 1
    if by_domain:
        print("By domain:", ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items())))

    if args.mismatches_only:
        to_print = mismatches
    else:
        to_print = all_trajectories

    for i, m in enumerate(to_print, 1):
        domain_tag = f"[{m['domain']}]" if m["domain"] else "[unknown]"
        recall_tag = f"gold_in_retrieval={m['n_seen']}/{m['n_gold']}"
        mismatch_tag = " MISMATCH" if m["ranked_doc_ids"] != m["gold_doc_ids"] else ""
        print(f"\n[{i}] {m['file']} {domain_tag} {recall_tag}{mismatch_tag}")
        print(f"    Query:   {m['query'][:120]}")
        if m["tools_used"]:
            tools_line = ", ".join(f"{tool}={m['tool_counts'][tool]}" for tool in m["tools_used"])
            print(f"    Tools:   {tools_line}")
        else:
            print("    Tools:   none")
        print(f"    Ranked:  {m['ranked_doc_ids']}")
        print(f"    Gold:    {m['gold_doc_ids']}")
        if m["gold_seen"]:
            print(f"    Seen:    {m['gold_seen']}")
        if m["gold_missed"]:
            print(f"    Not seen:{m['gold_missed']}")
        if m["gold_seen_in_answer"]:
            print(f"    Seen in answer:      {m['gold_seen_in_answer']}")
        if m["gold_seen_missing_from_answer"]:
            print(f"    Seen, not in answer: {m['gold_seen_missing_from_answer']}")
        if m["missing_from_ranked"]:
            print(f"    Missing from answer: {m['missing_from_ranked']}")
        if m["extra_in_ranked"]:
            print(f"    Extra in answer:     {m['extra_in_ranked']}")

    print_final_gold_outcomes(all_trajectories)


if __name__ == "__main__":
    main()
