"""veRL AgentLoop for the agentic retrieval ReAct rollout.

Multi-turn GRPO needs three things from a rollout:

1. A trajectory of token ids that the optimiser can attribute credit to.
2. A *response mask* with 1 for tokens the policy generated and 0 for tokens
   that came from the prompt, the tool server, or the chat template framing.
   Without this mask the agent gets penalised for tool-server tokens it never
   produced — the classic multi-turn RL bug.
3. A scalar reward (delivered separately by `reward.py` reading the
   ``ranked_doc_ids`` field out of ``AgentLoopOutput.extra_fields``).

This file implements (1) and (2). It registers as an AgentLoop named
``retrieval_react`` and is selected per-row via the ``agent_name`` column of
the GRPO parquet (see `data_prep.py`).

Token-bookkeeping (TI/TO consistency)
-------------------------------------
The single source of truth is veRL's ``apply_chat_template`` helper. It always
appends ``add_generation_prompt=True``, so each call returns a sequence whose
suffix is the next assistant header — exactly what we want for the next
``server_manager.generate`` call.

Per turn:

    * generate(running_ids)       -> TokenOutput.token_ids   (mask=1)
    * append assistant message
    * if tool calls: execute, append tool messages, then
        apply_chat_template(tool_messages_only, remove_system_prompt=True)
            -> framing tokens for the tool wraps + next assistant header (mask=0)

Constructed identically to ``tool_agent_loop.ToolAgentLoop`` so the trainer's
tokenisation of the trajectory matches what we emit here. The ``ti_to_check``
module audits this property on a sample of trajectories per training step.
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
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    from verl.utils.profiler import simple_timer  # type: ignore
    from verl.utils.rollout_trace import rollout_trace_op  # type: ignore
    from verl.workers.rollout.replica import TokenOutput  # type: ignore

    _VERL_AVAILABLE = True
except Exception:  # pragma: no cover - verl optional at import time
    _VERL_AVAILABLE = False

    class AgentLoopBase:  # type: ignore[no-redef]
        """Stub base used when verl is not installed (lets tests import the module)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.config = kwargs.get("trainer_config", None)
            self.server_manager = kwargs.get("server_manager", None)
            self.tokenizer = kwargs.get("tokenizer", None)
            self.processor = None
            self.rollout_config = None
            self.system_prompt: list[int] = []

        async def apply_chat_template(self, *args, **kwargs):
            raise NotImplementedError("verl is required to actually run the AgentLoop")

    class AgentLoopMetrics:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class AgentLoopOutput(dict):  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(kwargs)

    class TokenOutput:  # type: ignore[no-redef]
        token_ids: list[int]
        extra_fields: dict[str, Any]
        num_preempted: int | None

    def register(name: str):  # type: ignore[no-redef]
        def _wrap(cls):
            cls._registered_name = name
            return cls
        return _wrap

    def rollout_trace_op(fn):  # type: ignore[no-redef]
        return fn

    def simple_timer(name: str, metrics: dict[str, Any]):  # type: ignore[no-redef]
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


from agent.parser import parse_answer
from agent.prompts import LOCAL_SEARCH_TOOL_SCHEMA, render_search_results

log = logging.getLogger(__name__)


_QWEN_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL
)


def _parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract Qwen3 XML tool calls from raw assistant text.

    vLLM's `qwen3_xml` parser does this server-side when streaming through the
    OpenAI API. veRL's rollout server returns raw token ids decoded by the
    tokenizer, so we re-implement the same parse here. Tolerates both XML
    (``<function=...><parameter=...>``) and a JSON body (older Qwen format).
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
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


@register("retrieval_react")
class RetrievalReActAgentLoop(AgentLoopBase):
    """ReAct rollout for agentic retrieval.

    Reads its config from two places:
      * ``self.rollout_config.multi_turn`` for length/turn caps (set by the
        standard veRL multi-turn block).
      * ``self.config.agent_loop.retrieval_react`` for retrieval-specific
        knobs (tool server URL, top_k clamps, …).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        rc = self.rollout_config
        self.max_assistant_turns = int(getattr(rc.multi_turn, "max_assistant_turns", 8) or 8)
        self.max_user_turns = int(getattr(rc.multi_turn, "max_user_turns", 0) or 0)
        self.max_tool_response_length = int(
            getattr(rc.multi_turn, "max_tool_response_length", 4096) or 4096
        )
        self.tool_response_truncate_side = str(
            getattr(rc.multi_turn, "tool_response_truncate_side", "right") or "right"
        )
        self.prompt_length = int(rc.prompt_length)
        self.response_length = int(rc.response_length)

        retr_cfg: Any = {}
        try:
            retr_cfg = self.config.agent_loop.retrieval_react
        except Exception:
            try:
                retr_cfg = self.config["agent_loop"]["retrieval_react"]
            except Exception:
                retr_cfg = {}

        def _g(key: str, default: Any) -> Any:
            try:
                v = retr_cfg.get(key, default) if hasattr(retr_cfg, "get") else getattr(
                    retr_cfg, key, default
                )
            except Exception:
                v = default
            return v if v is not None else default

        self.tool_server_url = str(_g("tool_server_url", "http://localhost:8100")).rstrip("/")
        self.tool_timeout_s = float(_g("tool_timeout_s", 30.0))
        self.max_tool_calls = int(_g("max_tool_calls", 10))
        self.top_k_default = int(_g("top_k_default", 10))
        self.top_k_max = int(_g("top_k_max", 50))

        self._http = httpx.AsyncClient(timeout=self.tool_timeout_s)

    # ------------------------------------------------------------------ tool

    async def _call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
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
        if len(rendered) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                rendered = rendered[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                rendered = "(truncated)..." + rendered[-self.max_tool_response_length :]
            else:
                half = self.max_tool_response_length // 2
                rendered = rendered[:half] + "...(truncated)..." + rendered[-half:]
        summary = {
            "num_results": len(results),
            "top_doc_ids": [r["doc_id"] for r in results],
            "doc_ids_with_scores": [
                {"doc_id": r["doc_id"], "score": float(r["score"])} for r in results
            ],
        }
        return rendered, summary

    # ------------------------------------------------------------------- run

    @rollout_trace_op
    async def run(  # type: ignore[override]
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> AgentLoopOutput:
        """Roll out one trajectory and return an AgentLoopOutput."""
        messages: list[dict[str, Any]] = list(
            kwargs.get("raw_prompt") or kwargs.get("messages") or []
        )
        extra_info: dict[str, Any] = dict(kwargs.get("extra_info") or {})
        request_id = uuid.uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids: list[int] = await self.apply_chat_template(
            messages, tools=[LOCAL_SEARCH_TOOL_SCHEMA]
        )
        running_ids: list[int] = list(prompt_ids)
        response_mask: list[int] = []

        seen: set[str] = set()
        ranked_doc_ids: list[str] = []
        stopped_reason = "max_turns"
        num_tool_calls = 0
        num_assistant_turns = 0
        tool_call_traces: list[dict[str, Any]] = []

        for _turn in range(self.max_assistant_turns):
            num_assistant_turns += 1

            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=running_ids,
                    sampling_params=sampling_params,
                )
            new_ids: list[int] = list(output.token_ids)
            running_ids = running_ids + new_ids
            response_mask.extend([1] * len(new_ids))

            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            tool_calls = _parse_qwen_tool_calls(text)

            if not tool_calls:
                messages.append({"role": "assistant", "content": text})
                answer = parse_answer(text)
                if answer is None:
                    stopped_reason = "parse_error"
                else:
                    ranked_doc_ids = [d for d in answer if d in seen]
                    stopped_reason = "answer"
                break

            messages.append(
                {"role": "assistant", "content": text, "tool_calls": tool_calls}
            )

            new_tool_messages: list[dict[str, Any]] = []
            budget_exhausted = False
            with simple_timer("tool_calls", metrics):
                for tc in tool_calls:
                    if num_tool_calls >= self.max_tool_calls:
                        budget_exhausted = True
                        break
                    try:
                        args = json.loads(tc["function"]["arguments"]) or {}
                    except json.JSONDecodeError:
                        args = {}
                    rendered, summary = await self._call_tool(
                        tc["function"]["name"], args
                    )
                    num_tool_calls += 1

                    for did in summary.get("top_doc_ids", []):
                        seen.add(did)
                    tool_call_traces.append(
                        {
                            "turn": num_assistant_turns,
                            "tool": tc["function"]["name"],
                            "arguments": args,
                            "result_summary": summary,
                        }
                    )
                    new_tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": rendered,
                        }
                    )

            messages.extend(new_tool_messages)

            if new_tool_messages:
                tool_ids: list[int] = await self.apply_chat_template(
                    new_tool_messages, remove_system_prompt=True
                )
                running_ids = running_ids + list(tool_ids)
                response_mask.extend([0] * len(tool_ids))

            if budget_exhausted:
                stopped_reason = "max_tool_calls"
                break
            if len(response_mask) >= self.response_length:
                stopped_reason = "max_response_tokens"
                break

        response_mask = response_mask[: self.response_length]
        response_ids = running_ids[len(prompt_ids) : len(prompt_ids) + len(response_mask)]

        agent_metrics = AgentLoopMetrics(
            generate_sequences=float(metrics.get("generate_sequences", 0.0)),
            tool_calls=float(metrics.get("tool_calls", 0.0)),
            num_preempted=int(metrics.get("num_preempted", -1)),
        )

        extra_fields = {
            "ranked_doc_ids": ranked_doc_ids,
            "gold_doc_ids": list(extra_info.get("gold_doc_ids", [])),
            "stopped_reason": stopped_reason,
            "num_tool_calls": num_tool_calls,
            "tool_call_traces": tool_call_traces,
            "messages_full": _strip_internal_keys(messages),
            "response_len": len(response_mask),
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=num_assistant_turns,
            metrics=agent_metrics,
            extra_fields=extra_fields,
        )
