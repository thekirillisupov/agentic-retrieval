"""veRL AgentLoop for the agentic retrieval ReAct rollout.

Multi-turn GRPO needs three things from a rollout:

1. A trajectory of token ids that the optimiser can attribute credit to.
2. A *response mask* with 1 for tokens the policy generated and 0 for tokens
   that came from the prompt, the tool server, or the chat template framing.
   Without this mask the agent gets penalised for tool-server tokens it never
   produced — the classic multi-turn RL bug.
3. A scalar reward (delivered separately by `reward.py` reading `ranked_doc_ids`
   out of `extra_fields`).

This file implements (1) and (2). It registers as an AgentLoop named
``retrieval_react`` and is selected per-row via the ``agent_name`` column of
the GRPO parquet (see `data_prep.py`).

Token-bookkeeping (TI/TO consistency)
-------------------------------------
The single source of truth is ``apply_chat_template(messages, ..., tokenize=True)``.
We track a running ``trajectory_ids`` (= prompt + responses + tool wraps).
Each turn:

    * ``generate(trajectory_ids)``       -> ``model_ids``  (mask=1)
    * append assistant message
    * execute any tool calls, append tool messages
    * re-template the full message list with ``add_generation_prompt=True``;
      everything past ``len(trajectory_ids) + len(model_ids)`` is the framing
      delta for tool replies + the next assistant header (mask=0).

Doing it this way means the trainer's tokenisation of the trajectory is
*identical* to what we logged here — the property `ti_to_check.py` verifies
on a sample of trajectories.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

import httpx

try:
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
except Exception:  # pragma: no cover - verl optional at import time
    AgentLoopBase = object  # type: ignore[assignment]
    AgentLoopOutput = dict  # type: ignore[assignment]

    def register(name: str):  # type: ignore[no-redef]
        def _wrap(cls):
            cls._registered_name = name
            return cls
        return _wrap


from agent.parser import parse_answer
from agent.prompts import LOCAL_SEARCH_TOOL_SCHEMA, render_search_results

log = logging.getLogger(__name__)


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL
)


def _parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract Qwen3 XML tool calls from raw assistant text.

    vLLM's `qwen3_xml` parser does this server-side when streaming through the
    OpenAI API. We may receive raw decoded tokens here, so we re-implement the
    same parse. Tolerates both XML (``<function=...><parameter=...>``) and a
    JSON body (older Qwen format).
    """
    out: list[dict[str, Any]] = []
    for m in _QWEN_TOOL_CALL_RE.finditer(text):
        body = m.group("body").strip()
        fn_match = re.search(
            r"<function=(?P<name>[^>]+)>(?P<args>.*?)</function>", body, re.DOTALL
        )
        if fn_match:
            name = fn_match.group("name").strip()
            args_block = fn_match.group("args")
            args: dict[str, Any] = {}
            for pm in re.finditer(
                r"<parameter=(?P<k>[^>]+)>(?P<v>.*?)</parameter>",
                args_block,
                re.DOTALL,
            ):
                args[pm.group("k").strip()] = pm.group("v").strip()
            out.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )
            continue
        try:
            j = json.loads(body)
            out.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": j.get("name", ""),
                        "arguments": json.dumps(j.get("arguments", {})),
                    },
                }
            )
        except json.JSONDecodeError:
            continue
    return out


def _strip_internal_keys(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop keys we added for bookkeeping before chat-templating."""
    out: list[dict[str, Any]] = []
    for m in messages:
        clean = {k: v for k, v in m.items() if not k.startswith("_")}
        out.append(clean)
    return out


@register("retrieval_react")
class RetrievalReActAgentLoop(AgentLoopBase):
    """ReAct rollout for agentic retrieval.

    Trainer-side configuration (read from ``cls.config.agent_loop.retrieval_react``):
        tool_server_url       FastAPI server URL.
        tool_timeout_s        HTTP timeout for /local_search.
        max_turns             hard cap on assistant turns.
        max_tool_calls        hard cap on tool calls.
        top_k_default         fallback for top_k.
        top_k_max             clamp.
        max_response_tokens   ceiling on cumulative response_ids length.
    """

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):  # type: ignore[override]
        if getattr(cls, "_initialised", False):
            return
        cls.config = config
        cls.tokenizer = tokenizer
        cfg = {}
        try:
            cfg = dict(config["agent_loop"]["retrieval_react"])
        except (KeyError, TypeError):
            try:
                cfg = dict(getattr(getattr(config, "agent_loop"), "retrieval_react"))
            except Exception:
                cfg = {}
        cls.tool_server_url = str(cfg.get("tool_server_url", "http://localhost:8100")).rstrip("/")
        cls.tool_timeout_s = float(cfg.get("tool_timeout_s", 30.0))
        cls.max_turns = int(cfg.get("max_turns", 8))
        cls.max_tool_calls = int(cfg.get("max_tool_calls", 10))
        cls.top_k_default = int(cfg.get("top_k_default", 10))
        cls.top_k_max = int(cfg.get("top_k_max", 50))
        cls.max_response_tokens = int(cfg.get("max_response_tokens", 4096))
        cls._http = httpx.AsyncClient(timeout=cls.tool_timeout_s)
        cls._initialised = True

    # ---------------------------------------------------------------- helpers

    def _template(self, messages: list[dict[str, Any]], *, generation_prompt: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(
            _strip_internal_keys(messages),
            add_generation_prompt=generation_prompt,
            tools=[LOCAL_SEARCH_TOOL_SCHEMA],
            tokenize=True,
        )

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if name != "local_search":
            return json.dumps({"error": f"unknown tool: {name}"}), {"error": "unknown_tool"}
        query = str(arguments.get("query", "")).strip()
        try:
            top_k = int(arguments.get("top_k", self.top_k_default))
        except (TypeError, ValueError):
            top_k = self.top_k_default
        top_k = max(1, min(top_k, self.top_k_max))

        try:
            r = await self._http.post(
                f"{self.tool_server_url}/local_search",
                json={"query": query, "top_k": top_k},
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:  # pragma: no cover - net errors
            log.warning("tool_server error: %s", e)
            return json.dumps({"error": str(e), "results": []}), {"error": str(e)}

        results = payload.get("results", [])
        rendered = render_search_results(results)
        summary = {
            "num_results": len(results),
            "top_doc_ids": [r["doc_id"] for r in results],
            "doc_ids_with_scores": [
                {"doc_id": r["doc_id"], "score": float(r["score"])} for r in results
            ],
        }
        return rendered, summary

    # ------------------------------------------------------------------- run

    async def run(  # type: ignore[override]
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Roll out one trajectory and return an AgentLoopOutput."""
        messages: list[dict[str, Any]] = list(
            kwargs.get("messages") or kwargs.get("raw_prompt") or []
        )
        extra_info: dict[str, Any] = dict(kwargs.get("extra_info") or {})

        prompt_ids: list[int] = self._template(messages, generation_prompt=True)
        response_ids: list[int] = []
        response_mask: list[int] = []

        seen: set[str] = set()
        ranked_doc_ids: list[str] = []
        stopped_reason = "max_turns"
        num_tool_calls = 0
        num_turns = 0
        tool_call_traces: list[dict[str, Any]] = []

        sp = dict(sampling_params)
        sp.setdefault("max_tokens", self.max_response_tokens)

        for turn in range(1, self.max_turns + 1):
            num_turns = turn
            remaining = self.max_response_tokens - len(response_ids)
            if remaining <= 0:
                stopped_reason = "max_response_tokens"
                break
            sp_turn = dict(sp)
            sp_turn["max_tokens"] = min(int(sp_turn.get("max_tokens", remaining)), remaining)

            gen = await self.server_manager.generate(  # type: ignore[attr-defined]
                request_id=f"{uuid.uuid4().hex}-{turn}",
                prompt_ids=prompt_ids + response_ids,
                sampling_params=sp_turn,
            )
            new_ids: list[int] = list(gen)
            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)

            response_ids.extend(new_ids)
            response_mask.extend([1] * len(new_ids))

            tool_calls = _parse_qwen_tool_calls(text)
            if tool_calls and num_tool_calls < self.max_tool_calls:
                # Single assistant message carrying every tool call from this turn.
                messages.append(
                    {"role": "assistant", "content": text, "tool_calls": tool_calls}
                )

                budget_exhausted = False
                for tc in tool_calls:
                    if num_tool_calls >= self.max_tool_calls:
                        budget_exhausted = True
                        break
                    try:
                        args = json.loads(tc["function"]["arguments"]) or {}
                    except json.JSONDecodeError:
                        args = {}
                    rendered, summary = await self._call_tool(tc["function"]["name"], args)
                    num_tool_calls += 1

                    for did in summary.get("top_doc_ids", []):
                        seen.add(did)

                    tool_call_traces.append(
                        {
                            "turn": turn,
                            "tool": tc["function"]["name"],
                            "arguments": args,
                            "result_summary": summary,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": rendered,
                        }
                    )

                # Re-template the conversation; the tail past what we already
                # have is the framing (tool wraps + next assistant header).
                next_full_ids = self._template(messages, generation_prompt=True)
                already = len(prompt_ids) + len(response_ids)
                if next_full_ids[:already] != (prompt_ids + response_ids):
                    # The chat template should be a prefix-extending function.
                    # If it isn't (rare: tokeniser merges across boundaries),
                    # we fall back to extending the mask conservatively with
                    # mask=0 for the entire delta. This keeps training safe at
                    # the cost of slightly under-attributing credit.
                    log.warning(
                        "chat template not prefix-stable; falling back to "
                        "conservative re-tokenisation"
                    )
                    full = next_full_ids
                    response_ids = full[len(prompt_ids):]
                    # Mark only `new_ids` as policy-generated; everything else
                    # is framing.
                    response_mask = [0] * len(response_ids)
                    # Best-effort: find the first occurrence of new_ids inside
                    # response_ids and mark those positions.
                    for pos in range(0, len(response_ids) - len(new_ids) + 1):
                        if response_ids[pos:pos + len(new_ids)] == new_ids:
                            for i in range(len(new_ids)):
                                response_mask[pos + i] = 1
                            break
                else:
                    delta = next_full_ids[already:]
                    response_ids.extend(delta)
                    response_mask.extend([0] * len(delta))

                if budget_exhausted:
                    stopped_reason = "max_tool_calls"
                    break
                if len(response_ids) >= self.max_response_tokens:
                    stopped_reason = "max_response_tokens"
                    break
                continue

            # No tool calls — must contain <answer>.
            messages.append({"role": "assistant", "content": text})
            answer = parse_answer(text)
            if answer is None:
                stopped_reason = "parse_error"
                break
            ranked_doc_ids = [d for d in answer if d in seen]
            stopped_reason = "answer"
            break

        if len(response_ids) > self.max_response_tokens:
            response_ids = response_ids[: self.max_response_tokens]
            response_mask = response_mask[: self.max_response_tokens]

        metrics = {
            "num_turns": num_turns,
            "num_tool_calls": num_tool_calls,
            "stopped_reason": stopped_reason,
            "response_len": len(response_ids),
            "active_response_len": int(sum(response_mask)),
        }

        return AgentLoopOutput(  # type: ignore[call-arg]
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "ranked_doc_ids": ranked_doc_ids,
                "gold_doc_ids": list(extra_info.get("gold_doc_ids", [])),
                "stopped_reason": stopped_reason,
                "num_tool_calls": num_tool_calls,
                "tool_call_traces": tool_call_traces,
                "messages_full": _strip_internal_keys(messages),
            },
        )
