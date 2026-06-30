"""Run the agent on a single question and save a rich debug JSON.

Includes per-turn thinking (from  or <think> blocks),
tool choices, full message history, metrics (when gold is known), and trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from agent.harness import AgentHarness, ToolServerClient
from agent.parser import parse_answer
from agent.prompts import format_user_content
from agent.schemas import AgentInput, Message
from eval_.baseline import load_eval
from eval_.metrics import evaluate_one
from trajectories.writer import trajectory_to_dict

log = logging.getLogger(__name__)

THINKING_RE = re.compile(
    r"<(?:think|redacted_thinking)>(.*?)</(?:think|redacted_thinking)>",
    re.DOTALL | re.IGNORECASE,
)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def extract_thinking(content: str | None) -> str | None:
    """Pull reasoning text out of assistant content, if present."""
    if not content:
        return None
    parts = [m.group(1).strip() for m in THINKING_RE.finditer(content)]
    if not parts:
        return None
    return "\n\n".join(p for p in parts if p)


def strip_thinking(content: str | None) -> str | None:
    """Assistant visible text with thinking blocks removed."""
    if content is None:
        return None
    out = THINKING_RE.sub("", content)
    out = TOOL_CALL_BLOCK_RE.sub("", out)
    return out.strip() or None


def build_turns(
    messages: list[Message], tool_calls: list[Any]
) -> list[dict[str, Any]]:
    """One entry per model turn: thinking, tools, answer snippet."""
    traces_by_turn: dict[int, list[dict[str, Any]]] = {}
    for tc in tool_calls:
        traces_by_turn.setdefault(tc.turn, []).append(
            {
                "tool": tc.tool,
                "arguments": tc.arguments,
                "result_summary": tc.result_summary,
                "latency_ms": tc.latency_ms,
            }
        )

    turns: list[dict[str, Any]] = []
    turn_idx = 0
    for msg in messages:
        if msg.role != "assistant":
            continue
        turn_idx += 1
        content = msg.content
        openai_tools = msg.tool_calls or []
        turns.append(
            {
                "turn": turn_idx,
                "thinking": extract_thinking(content),
                "content_without_thinking": strip_thinking(content),
                "content_raw": content,
                "tool_calls_requested": [
                    {
                        "id": tc.get("id"),
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments"),
                    }
                    for tc in openai_tools
                ],
                "tools_executed": traces_by_turn.get(turn_idx, []),
            }
        )
    return turns


def resolve_example(
    *,
    question: str | None,
    question_id: str | None,
    index: int | None,
    eval_path: Path,
) -> dict[str, Any]:
    if question is not None:
        return {
            "question_id": question_id or "custom",
            "question": question,
            "gold_doc_ids": [],
            "answer": None,
        }

    rows = load_eval(eval_path, subset_size=None)
    if question_id is not None:
        for row in rows:
            if row["question_id"] == question_id:
                return row
        raise SystemExit(f"question_id not found in {eval_path}: {question_id}")

    if index is not None:
        if index < 0 or index >= len(rows):
            raise SystemExit(f"--index {index} out of range [0, {len(rows) - 1}]")
        return rows[index]

    raise SystemExit("Provide --question, --question-id, or --index")


def _format_passage(p: dict[str, Any], *, rank: int | None = None) -> str:
    lines = []
    prefix = f"#{rank + 1} " if rank is not None else ""
    lines.append(f"{prefix}{p['doc_id']}")
    if p.get("title"):
        lines.append(f"  title: {p['title']}")
    if p.get("best_score") is not None:
        lines.append(f"  score: {p['best_score']:.4f}")
    text = p.get("text") or ""
    lines.append(f"  text: {text}")
    return "\n".join(lines)


def _attach_answer_passages(
    turns: list[dict[str, Any]], ranked_passages: list[dict[str, Any]]
) -> None:
    by_id = {p["doc_id"]: p for p in ranked_passages}
    for turn in reversed(turns):
        doc_ids = parse_answer(turn.get("content_raw"))
        if doc_ids is None:
            continue
        turn["answer_passages"] = [
            by_id[did]
            if did in by_id
            else {"doc_id": did, "title": "", "text": "(not retrieved — id not in seen passages)"}
            for did in doc_ids
        ]
        return


def print_retrieve_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"question_id: {result['question_id']}")
    print(f"question:    {result['question']}")
    print(f"top_k:       {result['top_k']}")
    if result.get("latency_ms") is not None:
        print(f"latency_ms:  {result['latency_ms']}")
    print(f"ranked_doc_ids: {result['ranked_doc_ids']}")
    if result.get("gold_doc_ids"):
        print(f"gold_doc_ids:   {result['gold_doc_ids']}")
        m = result["metrics"]
        print(
            f"metrics @ {m['k']}: ndcg={m['ndcg']:.4f}  "
            f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}"
        )
    print("=" * 72)
    print("\n[retrieved passages]")
    for i, p in enumerate(result["ranked_passages"]):
        print(_format_passage(p, rank=i))
        if i + 1 < len(result["ranked_passages"]):
            print()
    print("\n" + "=" * 72)


def retrieve_only(
    cfg: dict,
    example: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Single local_search against the index — no vLLM / agent loop."""
    tool = ToolServerClient(
        cfg["tool_server"]["url"],
        timeout_s=cfg["tool_server"].get("timeout_s", 30),
    )
    source = example.get("source") or cfg["index"].get("default_source")
    try:
        response = tool.local_search(
            query=example["question"], top_k=top_k, source=source
        )
    finally:
        tool.close()

    results: list[dict[str, Any]] = response.get("results", [])
    ranked_doc_ids = [r["doc_id"] for r in results]
    ranked_passages = [
        {
            "doc_id": r["doc_id"],
            "title": r.get("title", ""),
            "text": r.get("text", ""),
            "best_score": float(r["score"]),
        }
        for r in results
    ]

    gold = example.get("gold_doc_ids", [])
    metrics = None
    if gold:
        m = evaluate_one(ranked_doc_ids, set(gold), k=top_k)
        metrics = {
            "k": top_k,
            "ndcg": m.ndcg,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
        }

    return {
        "mode": "retrieve_only",
        "question_id": example.get("question_id", "custom"),
        "question": example["question"],
        "answer": example.get("answer"),
        "gold_doc_ids": gold,
        "top_k": top_k,
        "latency_ms": response.get("latency_ms"),
        "ranked_doc_ids": ranked_doc_ids,
        "ranked_passages": ranked_passages,
        "metrics": metrics,
    }


def print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"question_id: {result['question_id']}")
    print(f"question:    {result['question']}")
    if result.get("answer"):
        print(f"dataset answer: {result['answer']}")
    print(f"stopped_reason: {result['stopped_reason']}")
    print(f"ranked_doc_ids: {result['ranked_doc_ids']}")
    if result.get("gold_doc_ids"):
        print(f"gold_doc_ids:   {result['gold_doc_ids']}")
        m = result["metrics"]
        print(
            f"metrics @ {m['k']}: ndcg={m['ndcg']:.4f}  "
            f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}"
        )
    print("=" * 72)

    for turn in result["turns"]:
        print(f"\n--- turn {turn['turn']} ---")
        if turn.get("thinking"):
            print("[thinking]")
            print(turn["thinking"])
        if turn.get("tool_calls_requested"):
            print("[tool calls requested]")
            for tc in turn["tool_calls_requested"]:
                print(f"  {tc['name']}({tc['arguments']})")
        if turn.get("tools_executed"):
            print("[tools executed]")
            for ex in turn["tools_executed"]:
                print(
                    f"  {ex['tool']}({ex['arguments']}) -> {ex['result_summary']}"
                )
        visible = turn.get("content_without_thinking")
        if visible:
            print("[assistant text]")
            print(visible[:2000] + ("..." if len(visible) > 2000 else ""))
        if turn.get("answer_passages"):
            print("[answer passages]")
            for i, p in enumerate(turn["answer_passages"]):
                print(_format_passage(p, rank=i))
                if i + 1 < len(turn["answer_passages"]):
                    print()

    if result.get("ranked_passages") and not any(
        t.get("answer_passages") for t in result["turns"]
    ):
        print("\n[answer passages]")
        for i, p in enumerate(result["ranked_passages"]):
            print(_format_passage(p, rank=i))
            if i + 1 < len(result["ranked_passages"]):
                print()

    print("\n" + "=" * 72)


def run_one(
    cfg: dict,
    example: dict[str, Any],
    *,
    max_tool_calls: int | None = None,
    gold_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    gold = gold_doc_ids if gold_doc_ids is not None else example.get("gold_doc_ids", [])

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
        use_id_map=cfg["agent"].get("use_id_map", False),
        tool_budget_feedback=cfg["agent"].get("tool_budget_feedback", False),
    )

    try:
        agent_input = AgentInput(
            messages=[Message(role="user", content=example["question"])],
            max_turns=cfg["agent"]["max_turns"],
            max_tool_calls=max_tool_calls or cfg["agent"]["max_tool_calls"],
            top_k_default=cfg["index"].get("top_k_default", 10),
            source=example.get("source") or cfg["index"].get("default_source"),
        )
        output = harness.run(agent_input, gold_doc_ids=gold or None)
    finally:
        tool.close()

    traj = output.trajectory
    k = cfg["eval"]["ndcg_k"]
    metrics = None
    if gold:
        m = evaluate_one(output.ranked_doc_ids, set(gold), k=k)
        metrics = {
            "k": k,
            "ndcg": m.ndcg,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
        }

    turns = build_turns(traj.messages_full, traj.tool_calls)
    ranked_passages = [asdict(p) for p in output.ranked_passages]
    _attach_answer_passages(turns, ranked_passages)

    return {
        "question_id": example.get("question_id", "custom"),
        "question": example["question"],
        "answer": example.get("answer"),
        "gold_doc_ids": gold,
        "ranked_doc_ids": output.ranked_doc_ids,
        "ranked_passages": ranked_passages,
        "answer_passages": next(
            (t["answer_passages"] for t in reversed(turns) if t.get("answer_passages")),
            ranked_passages,
        ),
        "stopped_reason": output.stopped_reason,
        "num_turns": traj.num_turns,
        "num_tool_calls": traj.num_tool_calls,
        "metrics": metrics,
        "turns": turns,
        "tool_calls": [asdict(tc) for tc in traj.tool_calls],
        "trajectory": trajectory_to_dict(traj),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent on one question; save thinking + tools to JSON."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/processed/one_question_debug.json")
    parser.add_argument("--model-name", default=None, help="Override model.name")
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument(
        "--question",
        default=None,
        help="Question text (use instead of eval row)",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Load question from eval jsonl by question_id",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Load Nth row from eval jsonl (0-based)",
    )
    parser.add_argument(
        "--gold-doc-ids",
        default=None,
        help="Comma-separated gold doc ids (for custom --question)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print turn-by-turn summary to stdout",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Run one local_search on the index; skip vLLM/agent (tool server only)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of passages to retrieve (default: index.top_k_default from config)",
    )
    parser.add_argument(
        "--wrap-client",
        action="store_true",
        help="Wrap question as <client>...</client> (v2_search_only conversation format)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.model_name:
        cfg["model"]["name"] = args.model_name

    eval_path = Path(cfg["eval"]["eval_dataset_path"])
    gold_override = None
    if args.gold_doc_ids:
        gold_override = [x.strip() for x in args.gold_doc_ids.split(",") if x.strip()]

    example = resolve_example(
        question=args.question,
        question_id=args.question_id,
        index=args.index,
        eval_path=eval_path,
    )
    if args.wrap_client:
        example = {
            **example,
            "question": format_user_content(example["question"], "v2_search_only"),
        }
    if gold_override is not None:
        example = {**example, "gold_doc_ids": gold_override}

    top_k = args.top_k or cfg["index"].get("top_k_default", 10)
    top_k_max = cfg["index"].get("top_k_max", 50)
    if top_k < 1 or top_k > top_k_max:
        raise SystemExit(f"--top-k must be in [1, {top_k_max}], got {top_k}")

    if args.retrieve_only:
        result = retrieve_only(cfg, example, top_k=top_k)
    else:
        result = run_one(
            cfg,
            example,
            max_tool_calls=args.max_tool_calls,
            gold_doc_ids=example.get("gold_doc_ids"),
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("wrote %s", out_path)

    if not args.quiet:
        if args.retrieve_only:
            print_retrieve_summary(result)
        else:
            print_summary(result)


if __name__ == "__main__":
    main()
