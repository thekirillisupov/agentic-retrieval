"""ReAct loop. Talks to vLLM (OpenAI-compatible) and the tool server (FastAPI)."""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from typing import Any

import httpx
from openai import OpenAI

from agent.parser import parse_answer
from agent.prompts import (
    format_user_content,
    get_prompt,
    get_tool_schemas,
    render_grep_results,
    render_neighbours_results,
    render_search_results_json,
)
from agent.schemas import (
    AgentInput,
    AgentOutput,
    Message,
    RankedPassage,
    SeenPassage,
    StoppedReason,
    ToolCallTrace,
    Trajectory,
    assign_serial_id,
)

log = logging.getLogger(__name__)


class ToolServerClient:
    """Thin httpx wrapper around the tool server.

    Kept here (not in tool_server/) so the harness has zero runtime dependency on
    the server's Python module — it only needs the URL.
    """

    def __init__(self, url: str, timeout_s: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def local_search(
        self, query: str, top_k: int, source: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "top_k": top_k}
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}/local_search", json=payload)
        r.raise_for_status()
        return r.json()

    def grep(
        self, pattern: str, top_k: int, source: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"pattern": pattern, "top_k": top_k}
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}/grep", json=payload)
        r.raise_for_status()
        return r.json()

    def get_neighbours(
        self, doc_id: str, window: int, source: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"doc_id": doc_id, "window": window}
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}/get_neighbours", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self.client.close()


class AgentHarness:
    def __init__(
        self,
        *,
        model: str,
        vllm_url: str,
        api_key: str,
        tool_client: ToolServerClient,
        prompt_version: str = "v1",
        max_tokens: int = 4096,
        temperature: float = 0.6,
        top_k_default: int = 10,
        top_k_max: int = 50,
        use_id_map: bool = False,
        tool_budget_feedback: bool = False,
    ) -> None:
        self.model = model
        self.client = OpenAI(base_url=vllm_url, api_key=api_key)
        self.tool_client = tool_client
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k_default = top_k_default
        self.top_k_max = top_k_max
        self.use_id_map = use_id_map
        # Mirror of the training agent_loop knob (configs.agent_loop.retrieval_react
        # .tool_budget_feedback). When True, suffix each tool response with
        # `[calls used: N/max_tool_calls]` and, once the budget is spent, grant one
        # final turn for the model to answer instead of stopping immediately. Must
        # match training so eval trajectories match what the policy was trained on.
        self.tool_budget_feedback = tool_budget_feedback

    def run(
        self,
        agent_input: AgentInput,
        *,
        gold_doc_ids: list[str] | None = None,
        prompt_version: str | None = None,
    ) -> AgentOutput:
        pv = prompt_version or self.prompt_version
        system_prompt = get_prompt(pv)

        # Serialization: when enabled, doc_ids are replaced with small integers
        # throughout the episode. This is a harness-level switch (config-driven),
        # independent of the prompt version.
        _use_id_map = self.use_id_map
        # id_map_inv[i] == real doc_id for serial int i; id_map is the reverse.
        # Populated lazily by assign_serial_id; only consulted when _use_id_map is True.
        id_map: dict[str, int] = {}
        id_map_inv: list[str] = []

        # The harness owns the system message; caller passes only the conversation.
        # Apply prompt-version-specific user content formatting (e.g. <client> wrap).
        messages: list[Message] = [Message(role="system", content=system_prompt)]
        for msg in agent_input.messages:
            if msg.role == "user":
                messages.append(
                    Message(
                        role="user",
                        content=format_user_content(
                            str(msg.content or ""), pv
                        ),
                    )
                )
            else:
                messages.append(msg)

        seen: dict[str, SeenPassage] = {}
        tool_call_traces: list[ToolCallTrace] = []
        prompt_tokens = 0
        completion_tokens = 0

        stopped_reason: StoppedReason = "max_turns"
        ranked_doc_ids: list[str] = []
        num_tool_calls = 0
        # tool_budget_feedback diagnostic: tool calls emitted after the budget was
        # spent (each gets a stub nudge). Mirrors agent_loop's counter.
        num_over_budget_calls = 0
        turn = 0
        # tool_budget_feedback: grant the post-budget "final answer" turn exactly
        # once so a model that keeps calling tools can't spin (mirrors agent_loop).
        final_answer_chance_used = False

        for turn in range(1, agent_input.max_turns + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_openai() for m in messages],
                tools=get_tool_schemas(pv, use_id_map=_use_id_map),
                tool_choice="auto",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            usage = response.usage
            if usage is not None:
                prompt_tokens += usage.prompt_tokens or 0
                completion_tokens += usage.completion_tokens or 0

            choice = response.choices[0]
            assistant_msg = choice.message

            assistant_record = Message(
                role="assistant",
                content=assistant_msg.content,
                tool_calls=(
                    [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (assistant_msg.tool_calls or [])
                    ]
                    if assistant_msg.tool_calls
                    else None
                ),
            )
            messages.append(assistant_record)

            if assistant_msg.tool_calls:
                budget_exhausted = False
                for tc in assistant_msg.tool_calls:
                    if num_tool_calls >= agent_input.max_tool_calls:
                        budget_exhausted = True
                        if not self.tool_budget_feedback:
                            break
                        num_over_budget_calls += 1
                        # Feedback mode: answer over-budget calls with an explicit
                        # stub that nudges the model to stop calling tools and emit
                        # its <answer> on the next turn (mirrors agent_loop).
                        messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tc.id,
                                content=json.dumps(
                                    {
                                        "error": "tool_budget_exhausted",
                                        "instruction": (
                                            "You have used all tool calls. Do not "
                                            "call any more tools. Return your final "
                                            "<answer> now."
                                        ),
                                    }
                                )
                                + f"\n[calls used: {num_tool_calls}/{agent_input.max_tool_calls}]",
                            )
                        )
                        continue
                    tool_msg, trace = self._execute_tool_call(
                        tc,
                        turn,
                        seen,
                        agent_input.top_k_default,
                        agent_input.source,
                        id_map=id_map if _use_id_map else None,
                        id_map_inv=id_map_inv if _use_id_map else None,
                    )
                    num_tool_calls += 1
                    if self.tool_budget_feedback:
                        tool_msg.content = (
                            f"{tool_msg.content}\n[calls used: "
                            f"{num_tool_calls}/{agent_input.max_tool_calls}]"
                        )
                    messages.append(tool_msg)
                    tool_call_traces.append(trace)

                # One-shot "final answer" turn once the budget is spent: the
                # over-budget stubs already tell the model it is out of tool
                # calls, so the loop just continues into one more assistant turn.
                grant_final_answer = (
                    budget_exhausted
                    and self.tool_budget_feedback
                    and not final_answer_chance_used
                )
                if grant_final_answer:
                    final_answer_chance_used = True

                if budget_exhausted and not grant_final_answer:
                    stopped_reason = "max_tool_calls"
                    break
                continue

            # No tool calls → expect <answer>...</answer>
            answer = parse_answer(assistant_msg.content)
            if answer is None:
                stopped_reason = "parse_error"
                break

            # Drop hallucinated ids that the agent never actually retrieved.
            # Deduplicate by first occurrence so the model cannot game NDCG/recall
            # by repeating the same doc_id multiple times.
            valid: list[str] = []
            seen_in_answer: set[str] = set()
            for token in answer:
                if _use_id_map:
                    # Model emits integer ids; translate back to real doc_ids.
                    try:
                        idx = int(token)
                        doc_id = id_map_inv[idx] if 0 <= idx < len(id_map_inv) else None
                    except (ValueError, IndexError):
                        doc_id = None
                    if doc_id is None:
                        log.warning("Unknown serial id in answer, dropping: %s", token)
                        continue
                else:
                    doc_id = token

                if doc_id not in seen:
                    log.warning("Hallucinated doc_id in answer, dropping: %s", doc_id)
                elif doc_id in seen_in_answer:
                    log.warning("Duplicate doc_id in answer, dropping: %s", doc_id)
                else:
                    valid.append(doc_id)
                    seen_in_answer.add(doc_id)
            ranked_doc_ids = valid
            stopped_reason = "answer"
            break

        ranked_passages = self._build_ranked_passages(ranked_doc_ids, seen)

        trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model=self.model,
            prompt_version=pv,
            input_messages=list(agent_input.messages),
            max_turns=agent_input.max_turns,
            max_tool_calls=agent_input.max_tool_calls,
            messages_full=messages,
            tool_calls=tool_call_traces,
            ranked_doc_ids=ranked_doc_ids,
            stopped_reason=stopped_reason,
            num_turns=turn,
            num_tool_calls=num_tool_calls,
            num_over_budget_calls=num_over_budget_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            gold_doc_ids=gold_doc_ids or [],
        )

        return AgentOutput(
            ranked_doc_ids=ranked_doc_ids,
            ranked_passages=ranked_passages,
            trajectory=trajectory,
            stopped_reason=stopped_reason,
        )

    def _execute_tool_call(
        self,
        tc: Any,
        turn: int,
        seen: dict[str, SeenPassage],
        top_k_default: int,
        pinned_source: str | None = None,
        id_map: dict[str, int] | None = None,
        id_map_inv: list[str] | None = None,
    ) -> tuple[Message, ToolCallTrace]:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}

        # A `source` emitted by the model overrides the per-row pinned source.
        source = str(args.get("source") or "").strip() or pinned_source

        _SEARCH_NAMES = {"local_search", "search"}

        if name not in _SEARCH_NAMES and name not in ("grep", "get_neighbours"):
            content = json.dumps({"error": f"unknown tool: {name}"})
            return (
                Message(role="tool", tool_call_id=tc.id, content=content),
                ToolCallTrace(
                    turn=turn,
                    tool=name,
                    arguments=args,
                    result_summary={"error": "unknown_tool"},
                    latency_ms=0,
                ),
            )

        t0 = time.perf_counter()

        if name in _SEARCH_NAMES:
            query = str(args.get("query", "")).strip()
            top_k = int(args.get("top_k", top_k_default))
            top_k = max(1, min(top_k, self.top_k_max))
            # Over-fetch so that after deduplication we still return ~top_k new docs.
            fetch_k = min(top_k + len(seen), self.top_k_max)
            try:
                response = self.tool_client.local_search(
                    query=query, top_k=fetch_k, source=source
                )
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log.exception("tool server error")
                err_content = json.dumps({"error": str(e), "results": []})
                return (
                    Message(role="tool", tool_call_id=tc.id, content=err_content),
                    ToolCallTrace(
                        turn=turn,
                        tool=name,
                        arguments=args,
                        result_summary={"error": str(e)},
                        latency_ms=latency_ms,
                    ),
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            results: list[dict[str, Any]] = response.get("results", [])
            # Render the ranked hits in order, preserving the position of every
            # passage but optimising context: NEW docs are shown as full
            # passages, already-seen docs as id-only placeholders
            # ({"doc_id": ..., "seen": true}). We over-fetched above; walk in
            # rank order until top_k NEW docs have been shown, then stop (any
            # trailing seen ids below the last new doc are dropped). Already-seen
            # docs still get their score metadata refreshed.
            display: list[dict[str, Any]] = []
            new_doc_ids: list[str] = []
            for r in results:
                doc_id = r["doc_id"]
                score = float(r["score"])
                if doc_id in seen:
                    seen[doc_id].update(score)
                    stub: dict[str, Any] = {"doc_id": doc_id}
                    if id_map is not None and id_map_inv is not None:
                        stub["doc_id"] = assign_serial_id(doc_id, id_map, id_map_inv)
                    display.append(stub)
                else:
                    seen[doc_id] = SeenPassage(
                        doc_id=doc_id,
                        title=r["title"],
                        text=r["text"],
                        best_score=score,
                        first_seen_turn=turn,
                    )
                    if id_map is not None and id_map_inv is not None:
                        serial = assign_serial_id(doc_id, id_map, id_map_inv)
                        display.append({**r, "doc_id": serial})
                    else:
                        display.append(r)
                    new_doc_ids.append(doc_id)
                    if len(new_doc_ids) >= top_k:
                        break
            rendered = render_search_results_json(display)
            summary: dict[str, Any] = {
                "num_results": len(display),
                "num_new": len(new_doc_ids),
                "num_seen_refs": len(display) - len(new_doc_ids),
                "top_doc_ids": new_doc_ids,
            }

        elif name == "grep":
            pattern = str(args.get("pattern", "")).strip()
            top_k = int(args.get("top_k", top_k_default))
            top_k = max(1, min(top_k, self.top_k_max))
            try:
                response = self.tool_client.grep(
                    pattern=pattern, top_k=top_k, source=source
                )
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log.exception("tool server error (grep)")
                err_content = json.dumps({"error": str(e), "results": []})
                return (
                    Message(role="tool", tool_call_id=tc.id, content=err_content),
                    ToolCallTrace(
                        turn=turn,
                        tool=name,
                        arguments=args,
                        result_summary={"error": str(e)},
                        latency_ms=latency_ms,
                    ),
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            results = response.get("results", [])
            total_matches: int = response.get("total_matches", len(results))
            # Grep hits have no score; use 1.0 as a sentinel for exact matches.
            for r in results:
                doc_id = r["doc_id"]
                if doc_id not in seen:
                    seen[doc_id] = SeenPassage(
                        doc_id=doc_id,
                        title=r["title"],
                        text=r["text"],
                        best_score=1.0,
                        first_seen_turn=turn,
                    )
            if id_map is not None and id_map_inv is not None:
                results_display = [
                    {**r, "doc_id": assign_serial_id(r["doc_id"], id_map, id_map_inv)}
                    for r in results
                ]
            else:
                results_display = results
            rendered = render_grep_results(results_display, total_matches)
            summary = {
                "num_results": len(results),
                "total_matches": total_matches,
                "top_doc_ids": [r["doc_id"] for r in results],
            }

        else:  # get_neighbours
            doc_id_arg_raw = str(args.get("doc_id", "")).strip()
            window = int(args.get("window", 1))
            window = max(1, min(window, 10))

            # When id_map is active the model passes a serial integer; decode to
            # the real doc_id before hitting the tool server.
            if id_map is not None and id_map_inv is not None:
                try:
                    idx = int(doc_id_arg_raw)
                    doc_id_arg = (
                        id_map_inv[idx] if 0 <= idx < len(id_map_inv) else doc_id_arg_raw
                    )
                except ValueError:
                    doc_id_arg = doc_id_arg_raw
                # Keep the original int string for error messages shown to the model.
                anchor_display: Any = doc_id_arg_raw
            else:
                doc_id_arg = doc_id_arg_raw
                anchor_display = doc_id_arg_raw

            try:
                response = self.tool_client.get_neighbours(
                    doc_id=doc_id_arg, window=window, source=source
                )
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log.exception("tool server error (get_neighbours)")
                err_content = json.dumps({"error": str(e), "results": []})
                return (
                    Message(role="tool", tool_call_id=tc.id, content=err_content),
                    ToolCallTrace(
                        turn=turn,
                        tool=name,
                        arguments=args,
                        result_summary={"error": str(e)},
                        latency_ms=latency_ms,
                    ),
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            results = response.get("results", [])
            status = str(response.get("status", "ok"))
            # Neighbours are structural (no relevance score). Register so the
            # model may cite them; use 0.0 since there is no semantic score.
            for r in results:
                doc_id = r["doc_id"]
                if doc_id not in seen:
                    seen[doc_id] = SeenPassage(
                        doc_id=doc_id,
                        title=r["title"],
                        text=r["text"],
                        best_score=0.0,
                        first_seen_turn=turn,
                    )
            if id_map is not None and id_map_inv is not None:
                results_for_render = [
                    {**r, "doc_id": assign_serial_id(r["doc_id"], id_map, id_map_inv)}
                    for r in results
                ]
            else:
                results_for_render = results
            rendered = render_neighbours_results(
                results_for_render, status, anchor_display, window
            )
            summary = {
                "num_results": len(results),
                "status": status,
                "top_doc_ids": [r["doc_id"] for r in results],
            }

        tool_msg = Message(role="tool", tool_call_id=tc.id, content=rendered)
        trace = ToolCallTrace(
            turn=turn,
            tool=name,
            arguments=args,
            result_summary=summary,
            latency_ms=latency_ms,
        )
        return tool_msg, trace

    @staticmethod
    def _build_ranked_passages(
        ranked_doc_ids: list[str], seen: dict[str, SeenPassage]
    ) -> list[RankedPassage]:
        out: list[RankedPassage] = []
        for rank, doc_id in enumerate(ranked_doc_ids):
            sp = seen.get(doc_id)
            if sp is None:
                continue
            out.append(
                RankedPassage(
                    doc_id=doc_id,
                    title=sp.title,
                    text=sp.text,
                    rank=rank,
                    best_score=sp.best_score,
                    first_seen_turn=sp.first_seen_turn,
                    num_times_retrieved=sp.num_times_retrieved,
                )
            )
        return out
