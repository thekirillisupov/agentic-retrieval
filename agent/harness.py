"""ReAct loop. Talks to vLLM (OpenAI-compatible) and the tool server (FastAPI).

Episode semantics (seen/citability, id_map serialization, tool-response
budgets, tool-call budget feedback) live in agent/episode.py, shared with the
training rollout (grpo/agent_loop.py); this file owns only the transport: the
OpenAI-shaped model client, the tool-server HTTP client, and trajectory
logging.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.episode import (
    KNOWN_TOOL_NAMES,
    SEARCH_TOOL_NAMES,
    Episode,
    EpisodeConfig,
)
from agent.parser import parse_answer
from agent.prompts import (
    format_user_content,
    get_prompt,
    get_tool_schemas,
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
)

log = logging.getLogger(__name__)


def _default_result_fields() -> dict[str, str]:
    # Canonical field the harness needs -> field name in the external response
    # item. Identity by default (matches the in-repo tool server).
    return {
        "doc_id": "doc_id",
        "title": "title",
        "text": "text",
        "score": "score",
        "file_name": "file_name",
        "index": "index",
    }


@dataclass
class ResponseSchema:
    """Maps an external search/get_neighbours JSON response into the canonical
    shape the harness consumes.

    The harness expects, from every retrieval response, a top-level list of
    result items each carrying ``doc_id``/``title``/``text``/``score`` (plus
    optional ``file_name``/``index``), and — for grep/get_neighbours — top-level
    ``total_matches``/``status``. When an external service names those keys
    differently, configure the mapping under ``search.response`` in the config;
    the defaults reproduce the in-repo tool server exactly, so unset config = no
    change.
    """

    results_key: str = "results"
    status_key: str = "status"
    total_matches_key: str = "total_matches"
    fields: dict[str, str] = field(default_factory=_default_result_fields)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "ResponseSchema":
        if not cfg:
            return cls()
        merged = _default_result_fields()
        merged.update(cfg.get("fields") or {})
        base = cls()
        return cls(
            results_key=cfg.get("results_key", base.results_key),
            status_key=cfg.get("status_key", base.status_key),
            total_matches_key=cfg.get("total_matches_key", base.total_matches_key),
            fields=merged,
        )

    def _item(self, raw: dict[str, Any]) -> dict[str, Any]:
        f = self.fields
        doc_id_key = f.get("doc_id", "doc_id")
        if doc_id_key not in raw:
            raise KeyError(
                f"result item missing doc_id field {doc_id_key!r}; keys={list(raw)}"
            )
        out: dict[str, Any] = {
            "doc_id": raw[doc_id_key],
            # Defaulted so rendering never KeyErrors on services that omit them;
            # score is unused for grep/neighbours, 0.0 is a harmless sentinel.
            "title": raw.get(f.get("title", "title"), ""),
            "text": raw.get(f.get("text", "text"), ""),
            "score": raw.get(f.get("score", "score"), 0.0),
        }
        file_name = raw.get(f.get("file_name", "file_name"))
        if file_name is not None:
            out["file_name"] = file_name
        index = raw.get(f.get("index", "index"))
        if index is not None:
            out["index"] = index
        return out

    def normalize(self, response: dict[str, Any]) -> dict[str, Any]:
        """Return ``response`` with its result list + status/total_matches
        rewritten under canonical keys (extra top-level keys are preserved)."""
        items = response.get(self.results_key) or []
        out = dict(response)
        results = [self._item(r) for r in items]
        out["results"] = results
        out["status"] = str(response.get(self.status_key, "ok"))
        out["total_matches"] = int(
            response.get(self.total_matches_key, len(results))
        )
        return out


def _default_gigasearch_fields() -> dict[str, str]:
    # Canonical field the harness needs -> dotted path into one GigaSearch
    # `faq_sources[i]` item (see GigaSearchResponseSchema). Matches the
    # `universal_search` skill's current response shape (scripts/DEMO.ipynb).
    return {
        "doc_id": "faq_id",
        "score": "relevance_score",
        "text": "metadata.raw_text",
        "title": "metadata.breadcrumbs",
        "file_name": "metadata.file_name",
        "index": "metadata.passage_index",
    }


def _get_path(d: dict[str, Any], path: str) -> Any:
    """Dotted-path lookup, e.g. ``"metadata.raw_text"`` -> ``d["metadata"]["raw_text"]``.

    Returns None (instead of raising) if any segment is missing/not a dict, so
    a hit with a slightly different shape is skipped rather than crashing the
    whole search response.
    """
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


@dataclass
class GigaSearchResponseSchema:
    """Maps a GigaSearch ``universal_search`` skill response into the
    canonical shape the harness consumes.

    The skill nests its hits several levels deep —
    ``result.response.messages[-1].faq_sources[]``, each a
    ``{faq_id, relevance_score, metadata: {raw_text, file_name, ...}}`` (see
    scripts/DEMO.ipynb for a full example response) — whereas the rest of the
    harness (Episode, prompt rendering) only knows the flat
    ``{doc_id, title, text, score, file_name, index}`` shape ``ResponseSchema``
    produces for the in-repo tool server. This flattens one into the other so
    no GigaSearch-specific branches leak past ``GigaSearchClient``. Field
    paths are configurable (dotted, for nested lookups) under
    ``search.gigasearch.response.fields``; unset = the defaults above.
    """

    fields: dict[str, str] = field(default_factory=_default_gigasearch_fields)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "GigaSearchResponseSchema":
        merged = _default_gigasearch_fields()
        if cfg:
            merged.update(cfg.get("fields") or {})
        return cls(fields=merged)

    def _item(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        f = self.fields
        doc_id = _get_path(raw, f.get("doc_id", "faq_id"))
        if doc_id is None:
            log.warning("gigasearch hit missing doc_id field; keys=%s", list(raw))
            return None
        out: dict[str, Any] = {
            "doc_id": str(doc_id),
            "title": _get_path(raw, f.get("title", "metadata.breadcrumbs")) or "",
            "text": _get_path(raw, f.get("text", "metadata.raw_text")) or "",
            "score": float(_get_path(raw, f.get("score", "relevance_score")) or 0.0),
        }
        file_name = _get_path(raw, f.get("file_name", "metadata.file_name"))
        if file_name is not None:
            out["file_name"] = file_name
        index = _get_path(raw, f.get("index", "metadata.passage_index"))
        if index is not None:
            out["index"] = index
        return out

    def normalize(
        self, raw_response: dict[str, Any], top_k: int | None = None
    ) -> dict[str, Any]:
        """Pull ``faq_sources`` out of the last assistant message and flatten
        them into ``{"results": [...], "status": "ok", "total_matches": N}``."""
        try:
            messages = raw_response["result"]["response"]["messages"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                "unexpected universal_search response shape (expected "
                f"result.response.messages); top-level keys={list(raw_response)}"
            ) from e

        faq_sources: list[dict[str, Any]] = []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                faq_sources = msg.get("faq_sources") or []
                break

        results = [item for r in faq_sources if (item := self._item(r)) is not None]
        if top_k is not None:
            results = results[:top_k]
        return {"results": results, "status": "ok", "total_matches": len(results)}


class GigaSearchClient:
    """Talks to an external GigaSearch ``universal_search`` skill instead of
    the in-repo tool server — mirrors the reference call in
    scripts/DEMO.ipynb: an mTLS-authenticated POST to
    ``{url}/sync/skill/universal_search`` whose body wraps the chat request
    in ``meta``/``message``/``path``/``timeout``/``configuration``, and whose
    response nests hits under ``result.response.messages[-1].faq_sources``.

    Only ``search`` is a native GigaSearch capability. ``grep`` has no
    equivalent there and degrades to empty hits. When
    ``search.gigasearch.opensearch`` is configured, ``get_neighbours`` is served
    by direct OpenSearch queries (scripts/DEMO.ipynb); otherwise it degrades to
  ``no_metadata`` and Episode tells the model to rely on search instead.
    """

    def __init__(
        self,
        url: str,
        *,
        source_uuid: str,
        configuration: dict[str, Any],
        skill: str = "universal_search",
        predict_path: str = "/predict",
        timeout_s: float = 30.0,
        request_timeout_s: float = 300.0,
        tls: dict[str, Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        opensearch_cfg: dict[str, Any] | None = None,
    ) -> None:
        if not source_uuid:
            raise ValueError(
                "search.gigasearch.source_uuid is required for backend=gigasearch"
            )
        self.url = url.rstrip("/")
        self.source_uuid = source_uuid
        self.configuration = configuration
        self.skill_path = "/sync/skill/" + skill.strip("/")
        self.predict_path = predict_path
        # Inner `timeout` field of the skill request body (how long the
        # GigaSearch pipeline itself may run) — distinct from `timeout_s`,
        # which bounds the harness's own HTTP wait for the response.
        self.request_timeout_s = request_timeout_s
        self.schema = GigaSearchResponseSchema.from_config(response_schema)

        tls = tls or {}
        cert: Any = None
        if tls.get("cert_file") and tls.get("key_file"):
            cert = (tls["cert_file"], tls["key_file"])
        elif tls.get("cert_file"):
            cert = tls["cert_file"]
        self.client = httpx.Client(
            timeout=timeout_s,
            cert=cert,
            verify=tls.get("verify", True),
            headers={"accept": "application/json", "Content-Type": "application/json"},
        )

        self._opensearch_client: Any | None = None
        self._opensearch_index_id: str | None = None
        if opensearch_cfg:
            from agent.opensearch_neighbours import (
                build_opensearch_client,
                index_id_from_configuration,
            )

            self._opensearch_client = build_opensearch_client(opensearch_cfg)
            index_id = opensearch_cfg.get("index_id")
            self._opensearch_index_id = (
                str(index_id)
                if index_id
                else index_id_from_configuration(configuration)
            )

    def _predict(
        self,
        request: dict[str, Any],
        *,
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "meta": {
                "external_uuid": str(uuid.uuid4()),
                "source_uuid": self.source_uuid,
            },
            "message": {"request_type": "chat", "request": request},
            "path": self.predict_path,
            "timeout": int(self.request_timeout_s),
            "configuration": configuration
            if configuration is not None
            else self.configuration,
        }
        r = self.client.post(f"{self.url}{self.skill_path}", json=payload)
        r.raise_for_status()
        return r.json()

    def local_search(
        self,
        query: str,
        top_k: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # `source` (per-episode corpus routing) has no GigaSearch equivalent —
        # routing lives in `configuration` (index_id), not in this call.
        # `extra_params` (e.g. {"filters": {...}}, see scripts/DEMO.ipynb) is
        # merged into the request verbatim; `messages` is applied last so the
        # model-controlled query always wins.  Callers may also pass
        # ``extra_params["configuration"]`` to override the deployment-default
        # GigaSearch pipeline for a single search (see scripts/DEMO_agentic_retrieval.ipynb).
        request: dict[str, Any] = dict(extra_params or {})
        configuration = request.pop("configuration", None)
        request.pop("messages", None)
        request["messages"] = [{"role": "user", "content": query}]
        raw = self._predict(
            request,
            configuration=configuration if configuration is not None else None,
        )
        return self.schema.normalize(raw, top_k=top_k)

    def grep(
        self,
        pattern: str,
        top_k: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # No exact-match equivalent in GigaSearch; degrade to "no matches"
        # instead of erroring, in case a model still calls it.
        return {"results": [], "status": "ok", "total_matches": 0}

    def get_neighbours(
        self,
        doc_id: str,
        window: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._opensearch_client is None:
            return {"results": [], "status": "no_metadata", "total_matches": 0}
        if not self._opensearch_index_id:
            return {"results": [], "status": "no_metadata", "total_matches": 0}

        from agent.opensearch_neighbours import get_neighbours as os_get_neighbours

        resp = os_get_neighbours(
            self._opensearch_client,
            self._opensearch_index_id,
            doc_id,
            window,
        )
        results = resp.get("results", [])
        return {
            "results": results,
            "status": resp.get("status", "ok"),
            "total_matches": len(results),
        }

    def close(self) -> None:
        self.client.close()
        if self._opensearch_client is not None:
            self._opensearch_client.close()


class ToolServerClient:
    """Thin httpx wrapper around the tool server.

    Kept here (not in tool_server/) so the harness has zero runtime dependency on
    the server's Python module — it only needs the URL.
    """

    def __init__(
        self,
        url: str,
        timeout_s: float = 30.0,
        *,
        search_path: str = "/local_search",
        grep_path: str = "/grep",
        neighbours_path: str = "/get_neighbours",
        response_schema: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        # Endpoint paths are configurable so the same harness can drive an
        # external retrieval service whose route names differ (e.g. `/search`
        # instead of `/local_search`). Defaults preserve the in-repo tool server.
        self.search_path = "/" + search_path.lstrip("/")
        self.grep_path = "/" + grep_path.lstrip("/")
        self.neighbours_path = "/" + neighbours_path.lstrip("/")
        # Response field mapping; defaults reproduce the in-repo tool server.
        self.schema = ResponseSchema.from_config(response_schema)
        self.client = httpx.Client(timeout=timeout_s)

    @staticmethod
    def _merge(
        base: dict[str, Any], extra_params: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge caller-supplied passthrough params under the harness-owned keys.

        ``extra_params`` are extra retrieval knobs the *caller* pins for the
        whole episode (filters, corpus routing, top_k caps, …). The harness-owned
        keys (query/pattern/doc_id/top_k/window/source) are applied last so the
        model-controlled query always wins — mirroring how the agent was trained:
        the model decides *what* to search, the caller decides *how*.
        """
        payload: dict[str, Any] = dict(extra_params or {})
        payload.update(base)
        return payload

    def local_search(
        self,
        query: str,
        top_k: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._merge({"query": query, "top_k": top_k}, extra_params)
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}{self.search_path}", json=payload)
        r.raise_for_status()
        return self.schema.normalize(r.json())

    def grep(
        self,
        pattern: str,
        top_k: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._merge({"pattern": pattern, "top_k": top_k}, extra_params)
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}{self.grep_path}", json=payload)
        r.raise_for_status()
        return self.schema.normalize(r.json())

    def get_neighbours(
        self,
        doc_id: str,
        window: int,
        source: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._merge({"doc_id": doc_id, "window": window}, extra_params)
        if source:
            payload["source"] = source
        r = self.client.post(f"{self.url}{self.neighbours_path}", json=payload)
        r.raise_for_status()
        return self.schema.normalize(r.json())

    def close(self) -> None:
        self.client.close()


class AgentHarness:
    def __init__(
        self,
        *,
        model: str,
        vllm_url: str | None = None,
        api_key: str = "EMPTY",
        tool_client: ToolServerClient,
        prompt_version: str = "v1",
        max_tokens: int = 4096,
        temperature: float = 0.6,
        top_k_default: int = 10,
        top_k_max: int = 50,
        use_id_map: bool = False,
        tool_budget_feedback: bool = False,
        model_client: Any | None = None,
        allow_model_source: bool = False,
        max_tool_response_length: int = 0,
        tool_response_truncate_side: str = "left",
        max_passage_tokens: int = 0,
        tokenizer: Any = None,
    ) -> None:
        self.model = model
        # A prebuilt client (e.g. an http backend behind mTLS) wins; otherwise
        # default to a vLLM / OpenAI-compatible client from vllm_url. Any client
        # exposing ``.chat.completions.create`` in the OpenAI shape works.
        if model_client is not None:
            self.client = model_client
        else:
            # Imported lazily so environments that always inject model_client
            # (training image, tests) don't need the openai package.
            from openai import OpenAI

            self.client = OpenAI(base_url=vllm_url, api_key=api_key)
        self.tool_client = tool_client
        self.prompt_version = prompt_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k_default = top_k_default
        self.top_k_max = top_k_max
        self.use_id_map = use_id_map
        # These knobs mirror the training agent_loop config
        # (agent_loop.retrieval_react.* and rollout.multi_turn.*) and MUST match
        # the run that produced the weights: the policy only ever saw episodes
        # shaped by them. The episode semantics themselves live in
        # agent/episode.py, shared with grpo/agent_loop.py.
        self.tool_budget_feedback = tool_budget_feedback
        self.allow_model_source = allow_model_source
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncate_side = tool_response_truncate_side
        self.max_passage_tokens = max_passage_tokens
        # Needed only when max_passage_tokens > 0 (the harness talks to the
        # model over an API, so the tokenizer must be supplied explicitly).
        self.tokenizer = tokenizer

    def run(
        self,
        agent_input: AgentInput,
        *,
        gold_doc_ids: list[str] | None = None,
        prompt_version: str | None = None,
    ) -> AgentOutput:
        pv = prompt_version or self.prompt_version
        system_prompt = get_prompt(pv)

        # All episode semantics (seen/citability, id_map serialization,
        # response budgets, tool-call budget feedback) live in the shared
        # Episode — the same code the training rollout runs.
        episode = Episode(
            EpisodeConfig(
                max_tool_calls=agent_input.max_tool_calls,
                top_k_default=agent_input.top_k_default,
                top_k_max=self.top_k_max,
                tool_budget_feedback=self.tool_budget_feedback,
                use_id_map=self.use_id_map,
                allow_model_source=self.allow_model_source,
                max_tool_response_length=self.max_tool_response_length,
                tool_response_truncate_side=self.tool_response_truncate_side,
                max_passage_tokens=self.max_passage_tokens,
            ),
            tokenizer=self.tokenizer,
        )

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

        tool_call_traces: list[ToolCallTrace] = []
        prompt_tokens = 0
        completion_tokens = 0

        stopped_reason: StoppedReason = "max_turns"
        ranked_doc_ids: list[str] = []
        turn = 0

        for turn in range(1, agent_input.max_turns + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_openai() for m in messages],
                tools=get_tool_schemas(pv, use_id_map=self.use_id_map),
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
                    if episode.budget_spent:
                        budget_exhausted = True
                        if not self.tool_budget_feedback:
                            break
                        messages.append(
                            Message(
                                role="tool",
                                tool_call_id=tc.id,
                                content=episode.over_budget_stub(),
                            )
                        )
                        continue
                    tool_msg, trace = self._execute_tool_call(
                        tc,
                        turn,
                        episode,
                        agent_input.source,
                        search_params=agent_input.search_params,
                    )
                    tool_msg.content = episode.register_tool_call(
                        tool_msg.content or ""
                    )
                    messages.append(tool_msg)
                    tool_call_traces.append(trace)

                grant_final_answer = episode.grant_final_answer_turn(
                    budget_exhausted
                )
                if budget_exhausted and not grant_final_answer:
                    stopped_reason = "max_tool_calls"
                    break
                continue

            # No tool calls → expect <answer>...</answer>
            answer = parse_answer(assistant_msg.content)
            if answer is None:
                stopped_reason = "parse_error"
                break

            ranked_doc_ids = episode.validate_answer(answer)
            stopped_reason = "answer"
            break

        ranked_passages = self._build_ranked_passages(ranked_doc_ids, episode.seen)

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
            num_tool_calls=episode.num_tool_calls,
            num_over_budget_calls=episode.num_over_budget_calls,
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
        episode: Episode,
        pinned_source: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> tuple[Message, ToolCallTrace]:
        """Transport half of one tool call: parse args, hit the tool server,
        and feed the raw response into the shared Episode pipeline."""
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}

        if name not in KNOWN_TOOL_NAMES:
            content, summary = episode.unknown_tool_response(name)
            return (
                Message(role="tool", tool_call_id=tc.id, content=content),
                ToolCallTrace(
                    turn=turn,
                    tool=name,
                    arguments=args,
                    result_summary=summary,
                    latency_ms=0,
                ),
            )

        source = episode.resolve_source(args, pinned_source)
        t0 = time.perf_counter()

        def _error(e: Exception) -> tuple[Message, ToolCallTrace]:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.exception("tool server error (%s)", name)
            content, summary = episode.tool_error_response(e)
            return (
                Message(role="tool", tool_call_id=tc.id, content=content),
                ToolCallTrace(
                    turn=turn,
                    tool=name,
                    arguments=args,
                    result_summary=summary,
                    latency_ms=latency_ms,
                ),
            )

        if name in SEARCH_TOOL_NAMES:
            query = str(args.get("query", "")).strip()
            top_k = episode.parse_top_k(args)
            try:
                response = self.tool_client.local_search(
                    query=query,
                    top_k=episode.search_fetch_k(top_k),
                    source=source,
                    extra_params=search_params,
                )
            except Exception as e:
                return _error(e)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            rendered, summary = episode.process_search_results(
                response.get("results", []), top_k, turn
            )

        elif name == "grep":
            pattern = str(args.get("pattern", "")).strip()
            top_k = episode.parse_top_k(args)
            try:
                response = self.tool_client.grep(
                    pattern=pattern, top_k=top_k, source=source,
                    extra_params=search_params,
                )
            except Exception as e:
                return _error(e)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            results = response.get("results", [])
            rendered, summary = episode.process_grep_results(
                results, response.get("total_matches", len(results)), turn
            )

        else:  # get_neighbours
            doc_id_arg, anchor_display = episode.decode_doc_id_arg(
                str(args.get("doc_id", "")).strip()
            )
            window = episode.parse_window(args)
            try:
                response = self.tool_client.get_neighbours(
                    doc_id=doc_id_arg, window=window, source=source,
                    extra_params=search_params,
                )
            except Exception as e:
                return _error(e)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            rendered, summary = episode.process_neighbours_results(
                response.get("results", []),
                str(response.get("status", "ok")),
                anchor_display,
                window,
                turn,
            )

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
