"""Agent eval on MuSiQue dev. Optional budget sweep.

For each example: build a single-turn user message, run the harness, compare
ranked_doc_ids to gold. Trajectories go to {trajectories.output_dir}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from tqdm import tqdm

from agent.harness import AgentHarness, ToolServerClient
from agent.schemas import AgentInput, Message
from eval_.baseline import load_eval
from eval_.metrics import aggregate, evaluate_one
from trajectories.checker import maybe_check_consistency
from trajectories.writer import TrajectoryWriter

log = logging.getLogger(__name__)


def run_agent_eval(
    cfg: dict,
    *,
    max_tool_calls: int | None = None,
    subset_size: int | None = None,
    suffix: str = "",
) -> dict:
    eval_path = Path(cfg["eval"]["eval_dataset_path"])
    eval_rows = load_eval(
        eval_path, subset_size=subset_size or cfg["eval"].get("subset_size")
    )

    tool = ToolServerClient(
        cfg["tool_server"]["url"],
        timeout_s=cfg["tool_server"].get("timeout_s", 30),
    )
    harness = AgentHarness(
        model=cfg["model"]["name"],
        vllm_url=cfg["model"]["vllm_url"],
        api_key=cfg["model"].get("api_key", "EMPTY"),
        tool_client=tool,
        prompt_version=cfg["agent"].get("prompt_version", "v1"),
        max_tokens=cfg["model"].get("max_tokens", 4096),
        temperature=cfg["model"].get("temperature", 0.6),
        top_k_default=cfg["index"].get("top_k_default", 10),
        top_k_max=cfg["index"].get("top_k_max", 50),
    )

    out_dir = Path(cfg["trajectories"]["output_dir"])
    if suffix:
        out_dir = out_dir / suffix
    writer = TrajectoryWriter(out_dir)
    check_every = cfg["trajectories"].get("ti_to_check_every_n", 20)

    per_example = []
    metrics = []
    stop_counts: dict[str, int] = {}
    try:
        for i, row in enumerate(tqdm(eval_rows, desc="agent eval")):
            agent_input = AgentInput(
                messages=[Message(role="user", content=row["question"])],
                max_turns=cfg["agent"]["max_turns"],
                max_tool_calls=max_tool_calls or cfg["agent"]["max_tool_calls"],
                top_k_default=cfg["index"].get("top_k_default", 10),
            )
            try:
                result = harness.run(agent_input)
            except Exception as e:
                log.exception("agent run failed for %s", row["question_id"])
                stop_counts["error"] = stop_counts.get("error", 0) + 1
                per_example.append(
                    {
                        "question_id": row["question_id"],
                        "error": str(e),
                    }
                )
                continue

            writer.write(result.trajectory)
            if i % max(1, check_every) == 0:
                maybe_check_consistency(result.trajectory)

            stop_counts[result.stopped_reason] = stop_counts.get(
                result.stopped_reason, 0
            ) + 1

            gold = set(row["gold_doc_ids"])
            k = cfg["eval"]["ndcg_k"]
            m = evaluate_one(result.ranked_doc_ids, gold, k=k)
            metrics.append(m)
            per_example.append(
                {
                    "question_id": row["question_id"],
                    "predicted": result.ranked_doc_ids,
                    "gold_doc_ids": row["gold_doc_ids"],
                    "ndcg": m.ndcg,
                    "recall": m.recall,
                    "precision": m.precision,
                    "num_turns": result.trajectory.num_turns,
                    "num_tool_calls": result.trajectory.num_tool_calls,
                    "stopped_reason": result.stopped_reason,
                    "trajectory_id": result.trajectory.trajectory_id,
                }
            )
    finally:
        tool.close()

    agg = aggregate(metrics)
    summary = {
        "n": len(metrics),
        "top_k": cfg["eval"]["ndcg_k"],
        "max_tool_calls": max_tool_calls or cfg["agent"]["max_tool_calls"],
        "ndcg": agg.ndcg,
        "recall": agg.recall,
        "precision": agg.precision,
        "stopped_reason_counts": stop_counts,
        "avg_turns": (
            sum(p.get("num_turns", 0) for p in per_example) / max(1, len(per_example))
        ),
        "avg_tool_calls": (
            sum(p.get("num_tool_calls", 0) for p in per_example)
            / max(1, len(per_example))
        ),
        "per_example": per_example,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--budget-sweep", action="store_true")
    parser.add_argument("--out", default="data/processed/agent_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.budget_sweep:
        sweep = cfg["eval"].get("budget_sweep", [1, 3, 5, 8])
        runs = []
        for budget in sweep:
            log.info("=== budget=%d ===", budget)
            summary = run_agent_eval(
                cfg, max_tool_calls=budget, suffix=f"budget_{budget}"
            )
            runs.append({"max_tool_calls": budget, "summary": summary})
            log.info(
                "budget=%d  ndcg=%.4f  recall=%.4f  avg_calls=%.2f",
                budget,
                summary["ndcg"],
                summary["recall"],
                summary["avg_tool_calls"],
            )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"sweep": runs}, indent=2, ensure_ascii=False)
        )
    else:
        summary = run_agent_eval(cfg)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(
            "agent n=%d  ndcg@%d=%.4f  recall@%d=%.4f  avg_calls=%.2f",
            summary["n"],
            summary["top_k"],
            summary["ndcg"],
            summary["top_k"],
            summary["recall"],
            summary["avg_tool_calls"],
        )


if __name__ == "__main__":
    main()
