"""Evaluate agentic-retrieval via the HTTP /retrieve endpoint.

Loads eval JSONL (e.g. from eval_.import_ckiz266), POSTs each question to
``/retrieve`` with ``include_trajectory=true``, and reports recall@k / hit@k.

For CKIZ-266 gold labels (page + file UUIDs), each ranked passage contributes
matchable ids from both ``doc_id`` (GigaSearch ``faq_id``) and ``file_name``
stem so recall is not sensitive to id format alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_.metrics import hit_at_k, recall_at_k

log = logging.getLogger(__name__)

DEFAULT_KS = (3, 5, 8)


def load_eval(path: Path, subset_size: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if subset_size is not None and subset_size > 0:
        rows = rows[:subset_size]
    return rows


def _file_stem(file_name: str | None) -> str | None:
    if not file_name:
        return None
    name = str(file_name).strip()
    for suffix in (".html", ".htm"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def ranked_match_sets(response: dict[str, Any]) -> list[set[str]]:
    """One set of matchable ids per ranked position (doc_id + file stem)."""
    items: list[set[str]] = []
    passages = response.get("ranked_passages") or []
    if passages:
        for passage in passages:
            ids: set[str] = set()
            doc_id = passage.get("doc_id")
            if doc_id is not None and str(doc_id).strip():
                ids.add(str(doc_id).strip())
            stem = _file_stem(passage.get("file_name"))
            if stem:
                ids.add(stem)
            items.append(ids)
        return items

    for doc_id in response.get("ranked_doc_ids") or []:
        if doc_id is not None and str(doc_id).strip():
            items.append({str(doc_id).strip()})
    return items


def flatten_ranked_ids(items: list[set[str]]) -> list[str]:
    """Collapse per-position id sets into a ranked list for metrics helpers."""
    ranked: list[str] = []
    seen: set[str] = set()
    for ids in items:
        for doc_id in ids:
            if doc_id not in seen:
                ranked.append(doc_id)
                seen.add(doc_id)
    return ranked


def recall_at_k_items(items: list[set[str]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    matched: set[str] = set()
    for ids in items[:k]:
        matched.update(ids & gold)
    return len(matched) / len(gold)


def hit_at_k_items(items: list[set[str]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    for ids in items[:k]:
        if ids & gold:
            return 1.0
    return 0.0


@dataclass
class EvalStats:
    n: int = 0
    errors: int = 0
    recall_sums: dict[int, float] = field(default_factory=dict)
    hit_sums: dict[int, float] = field(default_factory=dict)
    page_recall_sums: dict[int, float] = field(default_factory=dict)
    page_hit_sums: dict[int, float] = field(default_factory=dict)
    total_gold: int = 0
    gold_hits: dict[int, int] = field(default_factory=dict)

    def add(
        self,
        *,
        items: list[set[str]],
        gold_doc_ids: set[str],
        gold_page_id: str | None,
        ks: tuple[int, ...],
    ) -> None:
        ranked = flatten_ranked_ids(items)
        self.n += 1
        self.total_gold += len(gold_doc_ids)
        for k in ks:
            self.recall_sums[k] = self.recall_sums.get(k, 0.0) + recall_at_k_items(
                items, gold_doc_ids, k
            )
            self.hit_sums[k] = self.hit_sums.get(k, 0.0) + hit_at_k_items(
                items, gold_doc_ids, k
            )
            self.gold_hits[k] = self.gold_hits.get(k, 0) + sum(
                1 for g in gold_doc_ids if g in set(flatten_ranked_ids(items[:k]))
            )
            if gold_page_id:
                page_gold = {gold_page_id}
                self.page_recall_sums[k] = self.page_recall_sums.get(k, 0.0) + recall_at_k(
                    ranked, page_gold, k
                )
                self.page_hit_sums[k] = self.page_hit_sums.get(k, 0.0) + hit_at_k(
                    ranked, page_gold, k
                )

    def summarize(self, ks: tuple[int, ...]) -> dict[str, Any]:
        n = self.n or 1
        out: dict[str, Any] = {
            "n": self.n,
            "errors": self.errors,
            "n_gold": self.total_gold,
        }
        for k in ks:
            out[f"recall@{k}"] = self.recall_sums.get(k, 0.0) / n
            out[f"hit@{k}"] = self.hit_sums.get(k, 0.0) / n
            out[f"page_recall@{k}"] = self.page_recall_sums.get(k, 0.0) / n
            out[f"page_hit@{k}"] = self.page_hit_sums.get(k, 0.0) / n
            ng = self.total_gold or 1
            out[f"recall@{k}_micro"] = self.gold_hits.get(k, 0) / ng
        return out


def retrieve(
    *,
    base_url: str,
    question: str,
    search_params: dict[str, Any],
    prompt_version: str,
    include_trajectory: bool,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/retrieve"
    payload = {
        "messages": [{"role": "user", "content": question}],
        "search_params": search_params,
        "prompt_version": prompt_version,
        "include_trajectory": include_trajectory,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_eval(
    eval_path: Path,
    *,
    base_url: str,
    prompt_version: str,
    search_params: dict[str, Any],
    include_trajectory: bool,
    timeout_s: float,
    subset_size: int | None,
    ks: tuple[int, ...],
    sleep_s: float,
    trajectories_path: Path | None,
) -> dict[str, Any]:
    rows = load_eval(eval_path, subset_size=subset_size)
    stats = EvalStats()
    per_example: list[dict[str, Any]] = []
    traj_f = None
    if trajectories_path is not None:
        trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        traj_f = trajectories_path.open("w", encoding="utf-8")

    try:
        for i, row in enumerate(rows):
            question_id = row["question_id"]
            question = row["question"]
            gold_doc_ids = set(row.get("gold_doc_ids") or [])
            gold_page_id = row.get("gold_page_id")

            t0 = time.perf_counter()
            try:
                response = retrieve(
                    base_url=base_url,
                    question=question,
                    search_params=search_params,
                    prompt_version=prompt_version,
                    include_trajectory=include_trajectory,
                    timeout_s=timeout_s,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                stats.errors += 1
                log.error("[%s] retrieve failed: %s", question_id, exc)
                per_example.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "error": str(exc),
                    }
                )
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            items = ranked_match_sets(response)
            ranked = flatten_ranked_ids(items)
            metrics_row = {
                f"recall@{k}": recall_at_k_items(items, gold_doc_ids, k) for k in ks
            }
            metrics_row.update(
                {f"hit@{k}": hit_at_k_items(items, gold_doc_ids, k) for k in ks}
            )
            if gold_page_id:
                metrics_row.update(
                    {
                        f"page_recall@{k}": recall_at_k(ranked, {gold_page_id}, k)
                        for k in ks
                    }
                )
                metrics_row.update(
                    {f"page_hit@{k}": hit_at_k(ranked, {gold_page_id}, k) for k in ks}
                )

            stats.add(
                items=items,
                gold_doc_ids=gold_doc_ids,
                gold_page_id=gold_page_id,
                ks=ks,
            )

            example = {
                "question_id": question_id,
                "question": question,
                "gold_page_id": gold_page_id,
                "gold_doc_ids": sorted(gold_doc_ids),
                "ranked_doc_ids": response.get("ranked_doc_ids") or [],
                "ranked_match_ids": ranked,
                "num_tool_calls": response.get("num_tool_calls"),
                "stopped_reason": response.get("stopped_reason"),
                "latency_ms": latency_ms,
                **metrics_row,
            }
            per_example.append(example)

            if traj_f is not None:
                traj_f.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "question": question,
                            "gold_doc_ids": sorted(gold_doc_ids),
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            log.info(
                "[%d/%d] %s  recall@3=%.0f%%  page_hit@3=%s  calls=%s",
                i + 1,
                len(rows),
                question_id,
                100 * metrics_row.get("recall@3", 0.0),
                metrics_row.get("page_hit@3"),
                response.get("num_tool_calls"),
            )

            if sleep_s > 0 and i + 1 < len(rows):
                time.sleep(sleep_s)
    finally:
        if traj_f is not None:
            traj_f.close()

    summary = stats.summarize(ks)
    return {
        "eval_path": str(eval_path),
        "base_url": base_url,
        "prompt_version": prompt_version,
        "search_params": search_params,
        "ks": list(ks),
        "summary": summary,
        "per_example": per_example,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/processed/ckiz266/eval.jsonl"),
        help="Eval JSONL (from eval_.import_ckiz266)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8090",
        help="Agent service base URL",
    )
    parser.add_argument(
        "--prompt-version",
        default="v2_search_only",
        help="Prompt profile (v2, v2_search_only, …)",
    )
    parser.add_argument(
        "--search-params",
        default="{}",
        help='JSON object forwarded as search_params (default: {})',
    )
    parser.add_argument(
        "--include-trajectory",
        action="store_true",
        help="Ask /retrieve to include full trajectory in each response",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--ks",
        default="3,5,8",
        help="Comma-separated k values for recall@k (default: 3,5,8)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/ckiz266/results.json"),
        help="Summary + per-example metrics JSON",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="Optional JSONL path to save full /retrieve responses",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    search_params = json.loads(args.search_params)

    results = run_eval(
        args.eval,
        base_url=args.url,
        prompt_version=args.prompt_version,
        search_params=search_params,
        include_trajectory=args.include_trajectory,
        timeout_s=args.timeout_s,
        subset_size=args.subset_size,
        ks=ks,
        sleep_s=args.sleep_s,
        trajectories_path=args.trajectories,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    s = results["summary"]
    log.info(
        "n=%d  recall@3=%.1f%%  recall@5=%.1f%%  recall@8=%.1f%%  "
        "page_hit@3=%.1f%%  page_hit@5=%.1f%%  page_hit@8=%.1f%%",
        s["n"],
        100 * s.get("recall@3", 0.0),
        100 * s.get("recall@5", 0.0),
        100 * s.get("recall@8", 0.0),
        100 * s.get("page_hit@3", 0.0),
        100 * s.get("page_hit@5", 0.0),
        100 * s.get("page_hit@8", 0.0),
    )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
