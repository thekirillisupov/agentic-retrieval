"""Evaluate CKR JSONL sets via /retrieve or direct GigaSearch universal_search.

Gold ``gold_doc_ids`` are chunk ids ``{document_id}_{index}`` (file stem +
passage index). GigaSearch returns ``faq_id`` values, so ranked ids are mapped
through ``eval_.ckr_ids`` before computing ndcg@k / recall@k.

Agent (one search tool call, ``v2_search_only``):

    python -m eval_.run_ckr_eval \\
        --eval data/ckr_eval/ckr_eval.jsonl \\
        --url http://127.0.0.1:8090 \\
        --prompt-version v2_search_only \\
        --ks 3,5,16 \\
        --out data/processed/ckr_eval/ckr_eval_results.json

Direct GigaSearch (single search phase, no LLM — mirrors scripts/DEMO.ipynb):

    python -m eval_.run_ckr_eval \\
        --eval data/ckr_eval/ckr_eval.jsonl \\
        --gigasearch-direct \\
        --ks 3,5,16 \\
        --out data/processed/ckr_eval/ckr_eval_results.json

Both bundled sets:

    bash scripts/run_ckr_eval.sh
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

from eval_.ckr_ids import (
    build_doc_id_metadata_map,
    ckr_chunk_id,
    ranked_chunk_ids,
    ranked_match_sets,
    strip_client_tags,
)
from eval_.metrics import ndcg_at_k
from eval_.run_service_eval import (
    load_eval,
    recall_at_k_items,
)

log = logging.getLogger(__name__)

CKR_EVAL_DIR = Path("data/ckr_eval")
CKR_DATASETS = (
    "ckr_eval.jsonl",
    "ckr_per_doc_filtered_eval.jsonl",
)
DEFAULT_SEARCH_CONFIG = Path("configs/search_config.v.0.2.4.json")
DEFAULT_AGENT_CONFIG = Path("configs/ift_inference.yaml")


def default_search_params(config_path: Path = DEFAULT_SEARCH_CONFIG) -> dict[str, Any]:
    """Load GigaSearch pipeline config for /retrieve search_params."""
    return {"configuration": json.loads(config_path.read_text(encoding="utf-8"))}


def extract_tool_latency(response: dict[str, Any]) -> dict[str, Any]:
    """Pull tool-call counts and latencies from the harness trajectory."""
    traj = response.get("trajectory") or {}
    tool_calls = traj.get("tool_calls") or []
    tool_latencies = [
        int(tc["latency_ms"])
        for tc in tool_calls
        if tc.get("latency_ms") is not None
    ]
    search_latencies = [
        int(tc["latency_ms"])
        for tc in tool_calls
        if tc.get("tool") == "search" and tc.get("latency_ms") is not None
    ]
    total_tool_ms = sum(tool_latencies)
    total_search_ms = sum(search_latencies)
    n_tool_calls = response.get("num_tool_calls")
    if n_tool_calls is None:
        n_tool_calls = len(tool_calls)
    return {
        "num_tool_calls": int(n_tool_calls),
        "tool_latencies_ms": tool_latencies,
        "tool_latency_ms": total_tool_ms,
        "search_calls": len(search_latencies),
        "search_latencies_ms": search_latencies,
        "search_latency_ms": total_search_ms,
        "search_latency_ms_mean": (total_search_ms / len(search_latencies))
        if search_latencies
        else 0,
    }


def _dedup_ranked_ckr_ids(ranked_chunks: list[str | None]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk_id in ranked_chunks:
        if chunk_id and chunk_id not in seen:
            ranked.append(chunk_id)
            seen.add(chunk_id)
    return ranked


def _hits_to_ranked(
    hits: list[dict[str, Any]],
) -> tuple[list[str], list[set[str]], dict[str, dict[str, Any]], list[str | None]]:
    """Map GigaSearch hits to CKR chunk ids and per-rank match sets."""
    ranked_chunks: list[str | None] = []
    items: list[set[str]] = []
    metadata_map: dict[str, dict[str, Any]] = {}
    ranked_doc_ids: list[str] = []

    for hit in hits:
        doc_id = str(hit["doc_id"])
        ranked_doc_ids.append(doc_id)
        file_name = hit.get("file_name")
        index = hit.get("index")
        chunk_id = None
        if file_name is not None and index is not None:
            chunk_id = ckr_chunk_id(str(file_name), index)

        meta: dict[str, Any] = {}
        if file_name is not None:
            meta["file_name"] = str(file_name)
        if index is not None:
            meta["index"] = int(index)
        if chunk_id:
            meta["chunk_id"] = chunk_id
        if meta:
            metadata_map[doc_id] = meta

        ids = {doc_id}
        if chunk_id:
            ids.add(chunk_id)
        items.append(ids)
        ranked_chunks.append(chunk_id)

    return (
        _dedup_ranked_ckr_ids(ranked_chunks),
        items,
        metadata_map,
        ranked_chunks,
    )


def make_gigasearch_client(
    *,
    agent_config_path: Path = DEFAULT_AGENT_CONFIG,
    search_config_path: Path = DEFAULT_SEARCH_CONFIG,
) -> Any:
    """Build ``GigaSearchClient`` from inference yaml (scripts/DEMO.ipynb)."""
    import yaml

    from agent.harness import GigaSearchClient

    cfg = yaml.safe_load(agent_config_path.read_text(encoding="utf-8"))
    search_cfg = cfg.get("search") or {}
    gs_cfg = search_cfg.get("gigasearch") or {}
    config_path = Path(gs_cfg.get("config_path") or search_config_path)
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    return GigaSearchClient(
        url=str(search_cfg["url"]),
        source_uuid=str(gs_cfg["source_uuid"]),
        configuration=configuration,
        skill=str(gs_cfg.get("skill", "universal_search")),
        predict_path=str(gs_cfg.get("predict_path", "/predict")),
        timeout_s=float(search_cfg.get("timeout_s", 30)),
        request_timeout_s=float(gs_cfg.get("request_timeout_s", 300)),
        tls=gs_cfg.get("tls"),
        response_schema=gs_cfg.get("response"),
    )


def _direct_search_latency(search_ms: int) -> dict[str, Any]:
    return {
        "num_tool_calls": 1,
        "tool_latencies_ms": [search_ms],
        "tool_latency_ms": search_ms,
        "search_calls": 1,
        "search_latencies_ms": [search_ms],
        "search_latency_ms": search_ms,
        "search_latency_ms_mean": float(search_ms),
    }


@dataclass
class EvalStats:
    n: int = 0
    errors: int = 0
    recall_sums: dict[int, float] = field(default_factory=dict)
    ndcg_sums: dict[int, float] = field(default_factory=dict)
    total_gold: int = 0
    tool_calls: list[int] = field(default_factory=list)
    tool_latency_ms: list[int] = field(default_factory=list)
    llm_latency_ms: list[int] = field(default_factory=list)

    def add(
        self,
        *,
        items: list[set[str]],
        ranked_ckr_ids: list[str],
        gold_doc_ids: set[str],
        ks: tuple[int, ...],
        latency_ms: int,
        tool_latency: dict[str, Any],
    ) -> None:
        self.n += 1
        self.total_gold += len(gold_doc_ids)
        tool_ms = int(tool_latency["tool_latency_ms"])
        self.tool_calls.append(int(tool_latency["num_tool_calls"]))
        self.tool_latency_ms.append(tool_ms)
        self.llm_latency_ms.append(max(0, latency_ms - tool_ms))
        for k in ks:
            self.recall_sums[k] = self.recall_sums.get(k, 0.0) + recall_at_k_items(
                items, gold_doc_ids, k
            )
            self.ndcg_sums[k] = self.ndcg_sums.get(k, 0.0) + ndcg_at_k(
                ranked_ckr_ids, gold_doc_ids, k
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
            out[f"ndcg@{k}"] = self.ndcg_sums.get(k, 0.0) / n
        n_ok = len(self.tool_calls) or 1
        out["tool_calls_mean"] = sum(self.tool_calls) / n_ok
        out["tool_latency_ms_mean"] = sum(self.tool_latency_ms) / n_ok
        out["llm_latency_ms_mean"] = sum(self.llm_latency_ms) / n_ok
        return out


def retrieve(
    *,
    base_url: str,
    question: str,
    search_params: dict[str, Any],
    prompt_version: str,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/retrieve"
    payload = {
        "messages": [{"role": "user", "content": question}],
        "search_params": search_params,
        "prompt_version": prompt_version,
        "include_trajectory": True,
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
    search_config_path: Path | None,
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
            question = strip_client_tags(row["question"])
            gold_doc_ids = set(row.get("gold_doc_ids") or [])

            t0 = time.perf_counter()
            try:
                response = retrieve(
                    base_url=base_url,
                    question=question,
                    search_params=search_params,
                    prompt_version=prompt_version,
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
            tool_latency = extract_tool_latency(response)
            items = ranked_match_sets(response)
            metadata_map = (
                build_doc_id_metadata_map(response["trajectory"])
                if response.get("trajectory")
                else {}
            )
            chunk_map = {
                doc_id: str(meta["chunk_id"])
                for doc_id, meta in metadata_map.items()
                if meta.get("chunk_id")
            }
            ranked_chunks = ranked_chunk_ids(
                response, metadata_map=metadata_map or None
            )
            ranked_ckr_ids = _dedup_ranked_ckr_ids(ranked_chunks)

            metrics_row = {
                f"recall@{k}": recall_at_k_items(items, gold_doc_ids, k) for k in ks
            }
            metrics_row.update(
                {
                    f"ndcg@{k}": ndcg_at_k(ranked_ckr_ids, gold_doc_ids, k)
                    for k in ks
                }
            )

            stats.add(
                items=items,
                ranked_ckr_ids=ranked_ckr_ids,
                gold_doc_ids=gold_doc_ids,
                ks=ks,
                latency_ms=latency_ms,
                tool_latency=tool_latency,
            )

            example = {
                "question_id": question_id,
                "question": question,
                "gold_doc_ids": sorted(gold_doc_ids),
                "ranked_doc_ids": response.get("ranked_doc_ids") or [],
                "ranked_chunk_ids": ranked_ckr_ids,
                "ranked_ckr_chunk_ids": ranked_chunks,
                "doc_id_to_chunk": chunk_map,
                "doc_id_metadata": metadata_map,
                "stopped_reason": response.get("stopped_reason"),
                "latency_ms": latency_ms,
                "llm_latency_ms": max(0, latency_ms - int(tool_latency["tool_latency_ms"])),
                **tool_latency,
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
                            "ranked_doc_ids": response.get("ranked_doc_ids") or [],
                            "ranked_ckr_chunk_ids": ranked_chunks,
                            "doc_id_to_chunk": chunk_map,
                            "doc_id_metadata": metadata_map,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            log.info(
                "[%d/%d] %s  ndcg@3=%.3f  ndcg@5=%.3f  recall@3=%.0f%%  recall@5=%.0f%%",
                i + 1,
                len(rows),
                question_id,
                metrics_row.get("ndcg@3", 0.0),
                metrics_row.get("ndcg@5", 0.0),
                100 * metrics_row.get("recall@3", 0.0),
                100 * metrics_row.get("recall@5", 0.0),
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
        "search_config_path": str(search_config_path) if search_config_path else None,
        "search_params": search_params,
        "ks": list(ks),
        "summary": summary,
        "per_example": per_example,
    }


def run_gigasearch_eval(
    eval_path: Path,
    *,
    client: Any,
    top_k: int,
    search_config_path: Path | None,
    timeout_s: float,
    subset_size: int | None,
    ks: tuple[int, ...],
    sleep_s: float,
    trajectories_path: Path | None,
) -> dict[str, Any]:
    """Evaluate via one direct ``universal_search`` call per question."""
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
            question = strip_client_tags(row["question"])
            gold_doc_ids = set(row.get("gold_doc_ids") or [])

            t0 = time.perf_counter()
            try:
                search_resp = client.local_search(question, top_k=top_k)
            except Exception as exc:
                stats.errors += 1
                log.error("[%s] gigasearch failed: %s", question_id, exc)
                per_example.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "error": str(exc),
                    }
                )
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            hits = search_resp.get("results") or []
            ranked_ckr_ids, items, metadata_map, ranked_chunks = _hits_to_ranked(hits)
            ranked_doc_ids = [str(h["doc_id"]) for h in hits]
            chunk_map = {
                doc_id: str(meta["chunk_id"])
                for doc_id, meta in metadata_map.items()
                if meta.get("chunk_id")
            }
            tool_latency = _direct_search_latency(latency_ms)

            metrics_row = {
                f"recall@{k}": recall_at_k_items(items, gold_doc_ids, k) for k in ks
            }
            metrics_row.update(
                {
                    f"ndcg@{k}": ndcg_at_k(ranked_ckr_ids, gold_doc_ids, k)
                    for k in ks
                }
            )

            stats.add(
                items=items,
                ranked_ckr_ids=ranked_ckr_ids,
                gold_doc_ids=gold_doc_ids,
                ks=ks,
                latency_ms=latency_ms,
                tool_latency=tool_latency,
            )

            example = {
                "question_id": question_id,
                "question": question,
                "gold_doc_ids": sorted(gold_doc_ids),
                "ranked_doc_ids": ranked_doc_ids,
                "ranked_chunk_ids": ranked_ckr_ids,
                "ranked_ckr_chunk_ids": ranked_chunks,
                "doc_id_to_chunk": chunk_map,
                "doc_id_metadata": metadata_map,
                "stopped_reason": "gigasearch_direct",
                "latency_ms": latency_ms,
                "llm_latency_ms": 0,
                **tool_latency,
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
                            "ranked_doc_ids": ranked_doc_ids,
                            "ranked_ckr_chunk_ids": ranked_chunks,
                            "doc_id_to_chunk": chunk_map,
                            "doc_id_metadata": metadata_map,
                            "response": search_resp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            log.info(
                "[%d/%d] %s  ndcg@3=%.3f  ndcg@5=%.3f  ndcg@16=%.3f  "
                "recall@3=%.0f%%  recall@5=%.0f%%",
                i + 1,
                len(rows),
                question_id,
                metrics_row.get("ndcg@3", 0.0),
                metrics_row.get("ndcg@5", 0.0),
                metrics_row.get("ndcg@16", 0.0),
                100 * metrics_row.get("recall@3", 0.0),
                100 * metrics_row.get("recall@5", 0.0),
            )

            if sleep_s > 0 and i + 1 < len(rows):
                time.sleep(sleep_s)
    finally:
        if traj_f is not None:
            traj_f.close()

    summary = stats.summarize(ks)
    return {
        "eval_path": str(eval_path),
        "mode": "gigasearch_direct",
        "top_k": top_k,
        "search_config_path": str(search_config_path) if search_config_path else None,
        "ks": list(ks),
        "summary": summary,
        "per_example": per_example,
    }


def _default_out_path(eval_path: Path, out_dir: Path) -> Path:
    stem = eval_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return out_dir / f"{stem}_results.json"


def _default_traj_path(eval_path: Path, out_dir: Path) -> Path:
    stem = eval_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return out_dir / f"{stem}_trajectories.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        type=Path,
        nargs="*",
        help="Eval JSONL path(s). With --all, defaults to both CKR sets.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Evaluate all sets in {CKR_EVAL_DIR}",
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
        default=None,
        help=(
            "JSON object forwarded as search_params. "
            "Default: load configs/search_config.v.0.2.4.json as "
            '{"configuration": ...}'
        ),
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=DEFAULT_SEARCH_CONFIG,
        help="GigaSearch configuration json when --search-params is omitted",
    )
    parser.add_argument(
        "--gigasearch-direct",
        action="store_true",
        help=(
            "Call GigaSearch universal_search directly (one search phase, no "
            "/retrieve agent). Uses configs/ift_inference.yaml gigasearch block."
        ),
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=DEFAULT_AGENT_CONFIG,
        help="Agent yaml with search.gigasearch settings (--gigasearch-direct)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="GigaSearch hits to request (default: max --ks, at least 16)",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--ks",
        default="3,5",
        help="Comma-separated k values for recall@k / ndcg@k (default: 3,5)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/ckr_eval"),
        help="Directory for --all results (per-dataset *_results.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Summary JSON for a single --eval run",
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
    if args.search_params is None:
        search_params = default_search_params(args.search_config)
        search_config_path = args.search_config
    else:
        search_params = json.loads(args.search_params)
        search_config_path = None

    if args.all:
        eval_paths = [CKR_EVAL_DIR / name for name in CKR_DATASETS]
    elif args.eval:
        eval_paths = args.eval
    else:
        parser.error("pass --eval PATH [PATH ...] or --all")

    if len(eval_paths) > 1 and args.out is not None:
        parser.error("--out applies to a single --eval; use --out-dir with multiple sets")

    all_summaries: dict[str, Any] = {}
    for eval_path in eval_paths:
        if not eval_path.exists():
            raise FileNotFoundError(eval_path)

        out_path = args.out if args.out is not None else _default_out_path(eval_path, args.out_dir)
        traj_path = args.trajectories
        if traj_path is None and len(eval_paths) == 1:
            traj_path = _default_traj_path(eval_path, args.out_dir)
        elif traj_path is None and len(eval_paths) > 1:
            traj_path = _default_traj_path(eval_path, args.out_dir)

        log.info("=== %s ===", eval_path)
        if args.gigasearch_direct:
            top_k = args.top_k if args.top_k is not None else max(max(ks), 16)
            if args.search_config is not None:
                search_config_path = args.search_config
            else:
                search_config_path = DEFAULT_SEARCH_CONFIG
            log.info("mode: gigasearch_direct  top_k=%d", top_k)
            if search_config_path is not None:
                log.info("search_config: %s", search_config_path)
            client = make_gigasearch_client(
                agent_config_path=args.agent_config,
                search_config_path=search_config_path,
            )
            results = run_gigasearch_eval(
                eval_path,
                client=client,
                top_k=top_k,
                search_config_path=search_config_path,
                timeout_s=args.timeout_s,
                subset_size=args.subset_size,
                ks=ks,
                sleep_s=args.sleep_s,
                trajectories_path=traj_path,
            )
        else:
            if search_config_path is not None:
                log.info("search_config: %s", search_config_path)
            results = run_eval(
                eval_path,
                base_url=args.url,
                prompt_version=args.prompt_version,
                search_params=search_params,
                search_config_path=search_config_path,
                timeout_s=args.timeout_s,
                subset_size=args.subset_size,
                ks=ks,
                sleep_s=args.sleep_s,
                trajectories_path=traj_path,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        s = results["summary"]
        log.info(
            "%s  n=%d  ndcg@3=%.4f  ndcg@5=%.4f  ndcg@16=%.4f  "
            "recall@3=%.1f%%  recall@5=%.1f%%  "
            "tool_calls=%.2f  tool_ms=%.0f  llm_ms=%.0f",
            eval_path.name,
            s["n"],
            s.get("ndcg@3", 0.0),
            s.get("ndcg@5", 0.0),
            s.get("ndcg@16", 0.0),
            100 * s.get("recall@3", 0.0),
            100 * s.get("recall@5", 0.0),
            s.get("tool_calls_mean", 0.0),
            s.get("tool_latency_ms_mean", 0.0),
            s.get("llm_latency_ms_mean", 0.0),
        )
        log.info("wrote %s", out_path)
        if traj_path is not None:
            log.info("wrote %s", traj_path)
        all_summaries[str(eval_path)] = s

    if len(eval_paths) > 1:
        combined_path = args.out_dir / "summary.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
        log.info("wrote combined summary %s", combined_path)


if __name__ == "__main__":
    main()
