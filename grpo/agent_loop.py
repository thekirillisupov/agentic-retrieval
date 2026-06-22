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
import random
import re
import time
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
from agent.prompts import (
    get_tool_schemas,
    render_grep_results,
    render_neighbours_results,
    render_search_results_json,
)

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
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
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
                        "arguments": json.dumps(j.get("arguments", {}), ensure_ascii=False),
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
        self.max_assistant_turns = int(
            getattr(rc.multi_turn, "max_assistant_turns", 8) or 8
        )
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
                v = (
                    retr_cfg.get(key, default)
                    if hasattr(retr_cfg, "get")
                    else getattr(retr_cfg, key, default)
                )
            except Exception:
                v = default
            return v if v is not None else default

        self.tool_server_url = str(
            _g("tool_server_url", "http://localhost:8100")
        ).rstrip("/")
        self.tool_timeout_s = float(_g("tool_timeout_s", 60.0))
        self.max_tool_calls = int(_g("max_tool_calls", 10))
        self.top_k_default = int(_g("top_k_default", 10))
        self.top_k_max = int(_g("top_k_max", 50))
        # Comma-separated list of tool names to expose, e.g. "local_search" or
        # "search,grep" for the extending prompt variant.
        _raw_tools = str(_g("tool_names", "local_search"))
        self.tool_names: set[str] = {
            t.strip() for t in _raw_tools.split(",") if t.strip()
        }
        # Fallback corpus when a row doesn't carry one. Per-row routing prefers
        # extra_info["source"] (written by data_prep). Empty -> tool server default.
        self.default_source: str = str(_g("default_source", "") or "")
        # Whether a model-emitted `source` tool arg may override the per-row
        # pinned corpus. Default False: every search/get_neighbours in a rollout
        # is pinned to the row's corpus, so one question == one base. Flip to
        # True later to train the model to choose sources itself.
        self.allow_model_source: bool = bool(_g("allow_model_source", False))
        # After this many consecutive tool-server failures the worker triggers
        # a /reload on the server. Set to 0 to disable the circuit breaker.
        # Sized for ~128 agent workers: a single brownout (e.g. weight-update
        # spike) can produce a short burst of timeouts across many workers
        # at once. The /reload endpoint is singleflight + cooldown on the
        # server side, so multiple workers tripping the breaker together is
        # cheap, but the threshold should still be lax enough to absorb a
        # transient blip without aborting the worker.
        self.max_consecutive_tool_errors = int(_g("max_consecutive_tool_errors", 20))
        # Timeout (seconds) for the /reload request. Index reload can be slow
        # on a large FAISS corpus, so allow generous headroom.
        self.recovery_wait_s = float(_g("recovery_wait_s", 120.0))

        self._http = httpx.AsyncClient(timeout=self.tool_timeout_s)
        self._consecutive_tool_errors: int = 0

        # Fail fast: verify the tool server is reachable before any rollout.
        self._check_tool_server_health()

    # ---------------------------------------------------------- health / circuit

    def _check_tool_server_health(self) -> None:
        """Synchronous startup ping — raises RuntimeError if the server is unreachable.

        128 rollout workers all call this at the same instant during training
        startup, which would otherwise stampede the server's anyio threadpool
        (default ~40) even though ``/healthz`` is now ``async def``. Retry on
        transient connect/read timeouts with jittered backoff; only treat a
        genuine 5xx/503 as a permanent failure.
        """
        import httpx as _httpx

        url = f"{self.tool_server_url}/healthz"
        max_attempts = 5
        per_attempt_timeout = 30.0
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            # Jittered backoff de-correlates the 128 workers so retries don't
            # re-stampede the server. attempt==1 has no extra wait.
            if attempt > 1:
                backoff = min(2.0 ** (attempt - 2), 8.0) + random.uniform(0.0, 1.0)
                time.sleep(backoff)
            try:
                resp = _httpx.get(url, timeout=per_attempt_timeout)
            except _httpx.TransportError as exc:
                last_exc = exc
                log.warning(
                    "tool server /healthz transient error " "(attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                continue

            # Deterministic failures: surface immediately, no point retrying.
            if resp.status_code == 503:
                raise RuntimeError(
                    f"Tool server at {url} is not ready (503). "
                    "Start the server before launching training."
                )
            if resp.status_code >= 500:
                raise RuntimeError(
                    f"Tool server at {url} returned {resp.status_code}: {resp.text}"
                )
            log.info("Tool server health check passed: %s (attempt %d)", url, attempt)
            return

        raise RuntimeError(
            f"Cannot reach tool server at {url} after {max_attempts} attempts: "
            f"{last_exc}. Start the server before launching training."
        ) from last_exc

    async def _trigger_server_reload(self) -> bool:
        """Ask the tool server to reload its FAISS index.

        Returns True if the server successfully reloaded, False otherwise.
        Uses a fresh HTTP client to avoid reusing a potentially broken connection.
        """
        url = f"{self.tool_server_url}/reload"
        try:
            async with httpx.AsyncClient(timeout=self.recovery_wait_s) as client:
                resp = await client.post(url)
                resp.raise_for_status()
        except Exception as exc:
            log.warning("tool server /reload failed: %s", exc)
            return False
        log.info("tool server reloaded successfully: %s", resp.text)
        # Also replace the shared client so subsequent tool calls get a clean connection.
        await self._http.aclose()
        self._http = httpx.AsyncClient(timeout=self.tool_timeout_s)
        return True

    async def _record_tool_error(self, err: Exception) -> None:
        """Track consecutive failures; on threshold, attempt recovery before raising."""
        self._consecutive_tool_errors += 1
        log.warning(
            "tool_server error (%d consecutive): %s",
            self._consecutive_tool_errors,
            err,
        )
        if (
            self.max_consecutive_tool_errors <= 0
            or self._consecutive_tool_errors < self.max_consecutive_tool_errors
        ):
            return

        recovered = await self._trigger_server_reload()
        if recovered:
            self._consecutive_tool_errors = 0
            return

        raise RuntimeError(
            f"Tool server failed {self._consecutive_tool_errors} times in a row "
            "and /reload did not succeed. "
            "Aborting rollout worker to prevent silent reward degradation."
        ) from err

    def _record_tool_success(self) -> None:
        self._consecutive_tool_errors = 0

    # ------------------------------------------------------------------ tool

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        pinned_source: str = "",
        seen: set[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        _SEARCH_NAMES = {"local_search", "search"}

        if name not in _SEARCH_NAMES and name not in ("grep", "get_neighbours"):
            return json.dumps({"error": f"unknown tool: {name}"}), {
                "error": "unknown_tool"
            }

        try:
            top_k = int(arguments.get("top_k", self.top_k_default))
        except (TypeError, ValueError):
            top_k = self.top_k_default
        top_k = max(1, min(top_k, self.top_k_max))

        # By default the per-row pinned corpus wins, so one question stays in one
        # base. Only when allow_model_source is set may a model-emitted `source`
        # override it (for training the model to choose sources).
        if self.allow_model_source:
            source = str(arguments.get("source") or "").strip() or pinned_source
        else:
            source = pinned_source

        if name in _SEARCH_NAMES:
            query = str(arguments.get("query", "")).strip()
            # Over-fetch so that after deduplication against `seen` we still
            # return ~top_k new docs (mirrors agent/harness.py). Only meaningful
            # when dedup is active (seen is not None).
            fetch_k = (
                min(top_k + len(seen), self.top_k_max)
                if seen is not None
                else top_k
            )
            body: dict[str, Any] = {"query": query, "top_k": fetch_k}
            if source:
                body["source"] = source
            try:
                r = await self._http.post(
                    f"{self.tool_server_url}/local_search",
                    json=body,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as e:  # pragma: no cover - net errors
                await self._record_tool_error(e)
                return json.dumps({"error": str(e), "results": []}), {"error": str(e)}

            self._record_tool_success()
            results = payload.get("results", [])
            # Render the ranked hits in order, preserving the position of every
            # passage but optimising context: NEW docs are shown as full
            # passages, already-seen docs as id-only placeholders
            # ({"doc_id": ..., "seen": true}). We over-fetched above; walk in
            # rank order until top_k NEW docs have been shown, then stop (any
            # trailing seen ids below the last new doc are dropped). Only NEW
            # docs are added to `seen`.
            if seen is not None:
                display: list[dict[str, Any]] = []
                new_doc_ids: list[str] = []
                for r in results:
                    did = r["doc_id"]
                    if did in seen:
                        display.append({"doc_id": did})
                    else:
                        display.append(r)
                        seen.add(did)
                        new_doc_ids.append(did)
                        if len(new_doc_ids) >= top_k:
                            break
                new_results = display
            else:
                new_results = results
                new_doc_ids = [r["doc_id"] for r in results]
            rendered = render_search_results_json(new_results)
            summary: dict[str, Any] = {
                "num_results": len(new_results),
                "num_new": len(new_doc_ids),
                "num_seen_refs": len(new_results) - len(new_doc_ids),
                # Only the newly-shown docs (full passages). Seen placeholders
                # were already reported when first surfaced.
                "top_doc_ids": new_doc_ids,
                "doc_ids_with_scores": [
                    {"doc_id": r["doc_id"], "score": float(r["score"])}
                    for r in new_results
                    if "score" in r
                ],
            }

        elif name == "grep":
            pattern = str(arguments.get("pattern", "")).strip()
            grep_body: dict[str, Any] = {"pattern": pattern, "top_k": top_k}
            if source:
                grep_body["source"] = source
            try:
                r = await self._http.post(
                    f"{self.tool_server_url}/grep",
                    json=grep_body,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as e:  # pragma: no cover - net errors
                await self._record_tool_error(e)
                return json.dumps({"error": str(e), "results": []}), {"error": str(e)}

            self._record_tool_success()
            results = payload.get("results", [])
            total_matches: int = payload.get("total_matches", len(results))
            rendered = render_grep_results(results, total_matches)
            summary = {
                "num_results": len(results),
                "total_matches": total_matches,
                "top_doc_ids": [r["doc_id"] for r in results],
            }

        else:  # get_neighbours
            doc_id_arg = str(arguments.get("doc_id", "")).strip()
            try:
                window = int(arguments.get("window", 1))
            except (TypeError, ValueError):
                window = 1
            window = max(1, min(window, 10))
            nb_body: dict[str, Any] = {"doc_id": doc_id_arg, "window": window}
            if source:
                nb_body["source"] = source
            try:
                r = await self._http.post(
                    f"{self.tool_server_url}/get_neighbours",
                    json=nb_body,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception as e:  # pragma: no cover - net errors
                await self._record_tool_error(e)
                return json.dumps({"error": str(e), "results": []}), {"error": str(e)}

            self._record_tool_success()
            results = payload.get("results", [])
            status = str(payload.get("status", "ok"))
            rendered = render_neighbours_results(results, status, doc_id_arg, window)
            summary = {
                "num_results": len(results),
                "status": status,
                "top_doc_ids": [r["doc_id"] for r in results],
            }

        if len(rendered) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                rendered = rendered[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                rendered = "(truncated)..." + rendered[-self.max_tool_response_length :]
            else:
                half = self.max_tool_response_length // 2
                rendered = rendered[:half] + "...(truncated)..." + rendered[-half:]

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

        prompt_version = str(extra_info.get("prompt_version", "v1"))
        # Per-row corpus: prefer an explicit `source`, else derive from
        # `data_source` (e.g. "musique_retrieval" -> "musique"), else the
        # configured default. Pins every search to the matching index.
        row_source = str(extra_info.get("source") or "").strip()
        if not row_source:
            ds = str(extra_info.get("data_source") or "").strip()
            if ds.endswith("_retrieval"):
                ds = ds[: -len("_retrieval")]
            row_source = ds
        if not row_source:
            row_source = self.default_source
        _tool_schemas = get_tool_schemas(prompt_version)
        if self.tool_names:
            filtered = [
                s
                for s in _tool_schemas
                if s.get("function", {}).get("name") in self.tool_names
            ]
            if filtered:
                _tool_schemas = filtered
        prompt_ids: list[int] = await self.apply_chat_template(
            messages, tools=_tool_schemas
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
                    # Deduplicate by first occurrence so the model cannot game
                    # NDCG/recall by repeating the same doc_id multiple times.
                    _seen_ans: set[str] = set()
                    ranked_doc_ids = []
                    for _d in answer:
                        if _d in seen and _d not in _seen_ans:
                            ranked_doc_ids.append(_d)
                            _seen_ans.add(_d)
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
                        tc["function"]["name"], args, pinned_source=row_source, seen=seen
                    )
                    num_tool_calls += 1

                    # For non-search tools (grep, get_neighbours) `seen` is not
                    # updated inside _call_tool, so we still track them here.
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
        response_ids = running_ids[
            len(prompt_ids) : len(prompt_ids) + len(response_mask)
        ]

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
            # Text-only rollout. Must be {} (not the default None) — verl's
            # _compute_multi_modal_inputs unconditionally calls
            # output.multi_modal_data.get("images") when self.processor is
            # not None (true for any Qwen-VL/Qwen3.5-VL checkpoint, even
            # when the inputs are text-only).
            multi_modal_data={},
        )
