"""HTTP wrapper around the ReAct harness — the inference entrypoint.

Exposes a single retrieval endpoint that the outside world calls with a
conversation plus (optional) retrieval parameters. Internally it runs the same
``AgentHarness`` used in eval: the model talks to a local vLLM (OpenAI-compatible)
and, for every ``search`` / ``get_neighbours`` the model decides to make, the
harness calls an *external* retrieval service (embedder + reranker + index live
there, not in this image).

Contract, mirroring how the agent was trained:
  * the *caller* supplies the dialogue and pins retrieval params (``source``,
    ``search_params`` — filters/top_k caps/routing);
  * the *model* decides only the query text for each search;
  * the response is the ranked list of ``doc_id`` for the user's last turn.

Config is loaded from ``AGENT_CONFIG`` (default ``configs/inference.yaml``);
a few env vars (``VLLM_URL``, ``SEARCH_URL``, ``MODEL_NAME``) override it so the
container entrypoint can wire things up without editing the file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.harness import AgentHarness, ToolServerClient
from agent.prompts import PROFILES
from agent.schemas import AgentInput, Message

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/inference.yaml"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the inference config, then apply env-var overrides.

    Env overrides let the entrypoint point the harness at the co-located vLLM and
    the external search service without rewriting the yaml:
      * ``VLLM_URL``   -> model.vllm_url
      * ``MODEL_NAME`` -> model.name (must match vLLM's --served-model-name)
      * ``SEARCH_URL`` -> search.url (base URL of the external retrieval service)
    """
    cfg_path = Path(path or os.environ.get("AGENT_CONFIG", DEFAULT_CONFIG))
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    cfg.setdefault("model", {})
    cfg.setdefault("search", {})
    cfg.setdefault("agent", {})
    cfg.setdefault("service", {})

    if os.environ.get("VLLM_URL"):
        cfg["model"]["vllm_url"] = os.environ["VLLM_URL"]
    if os.environ.get("MODEL_NAME"):
        cfg["model"]["name"] = os.environ["MODEL_NAME"]
    if os.environ.get("SEARCH_URL"):
        cfg["search"]["url"] = os.environ["SEARCH_URL"]

    return cfg


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None


class RetrieveRequest(BaseModel):
    # The conversation, oldest-first, ending on the user's latest turn. The
    # harness prepends its own system prompt and wraps user turns per the prompt
    # version, so callers pass raw conversation content.
    messages: list[ChatMessage]
    # Extra retrieval knobs forwarded verbatim to the external search service:
    # index routing, filters, top_k caps, … Everything except the query text.
    # (Per-row `source` routing is a training-only concern; at inference the
    # target index is selected here.) The model still owns the query.
    search_params: dict[str, Any] = Field(default_factory=dict)
    # Per-request overrides of the config defaults (all optional).
    max_turns: int | None = None
    max_tool_calls: int | None = None
    top_k_default: int | None = None
    # Prompt profile to run with (e.g. "v2" = search + get_neighbours,
    # "v2_search_only" = search only). None -> config default. Each profile fixes
    # its own tool set / user-turn wrapping, so this also switches which tools the
    # model is offered. See GET /prompts for the available versions.
    prompt_version: str | None = None
    # Include the full ReAct trajectory (messages, tool calls, tokens) in the
    # response. Off by default to keep responses small.
    include_trajectory: bool = False


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    app = FastAPI(title="agentic-retrieval inference")

    model_cfg = cfg["model"]
    search_cfg = cfg["search"]
    agent_cfg = cfg["agent"]
    endpoints = search_cfg.get("endpoints", {}) or {}

    def _make_tool_client() -> ToolServerClient:
        return ToolServerClient(
            search_cfg["url"],
            timeout_s=float(search_cfg.get("timeout_s", 30)),
            search_path=endpoints.get("search", "/local_search"),
            grep_path=endpoints.get("grep", "/grep"),
            neighbours_path=endpoints.get("get_neighbours", "/get_neighbours"),
            response_schema=search_cfg.get("response"),
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/prompts")
    def prompts() -> dict[str, Any]:
        """Prompt versions the harness can run, and the current default."""
        return {
            "default": agent_cfg.get("prompt_version", "v2"),
            "available": sorted(PROFILES),
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict[str, Any]:
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")

        pv = req.prompt_version or agent_cfg.get("prompt_version", "v2")
        if pv not in PROFILES:
            raise HTTPException(
                400,
                f"unknown prompt_version {pv!r}; available={sorted(PROFILES)}",
            )
        tool = _make_tool_client()
        harness = AgentHarness(
            model=model_cfg["name"],
            vllm_url=model_cfg["vllm_url"],
            api_key=model_cfg.get("api_key", "EMPTY"),
            tool_client=tool,
            prompt_version=pv,
            max_tokens=int(model_cfg.get("max_tokens", 4096)),
            temperature=float(model_cfg.get("temperature", 0.6)),
            top_k_default=int(agent_cfg.get("top_k_default", 10)),
            top_k_max=int(agent_cfg.get("top_k_max", 50)),
            use_id_map=bool(agent_cfg.get("use_id_map", False)),
            tool_budget_feedback=bool(agent_cfg.get("tool_budget_feedback", False)),
        )

        agent_input = AgentInput(
            messages=[Message(role=m.role, content=m.content) for m in req.messages],
            max_turns=req.max_turns or int(agent_cfg.get("max_turns", 8)),
            max_tool_calls=req.max_tool_calls
            or int(agent_cfg.get("max_tool_calls", 3)),
            top_k_default=req.top_k_default
            or int(agent_cfg.get("top_k_default", 10)),
            # No `source`: at inference the target index is carried in
            # search_params, not pinned per-row (that was a training concern).
            search_params=req.search_params,
        )

        try:
            output = harness.run(agent_input, prompt_version=pv)
        except Exception as exc:  # pragma: no cover - surface upstream failures
            log.exception("harness run failed")
            raise HTTPException(502, f"harness run failed: {exc}") from exc
        finally:
            tool.close()

        traj = output.trajectory
        resp: dict[str, Any] = {
            "ranked_doc_ids": output.ranked_doc_ids,
            "ranked_passages": [asdict(p) for p in output.ranked_passages],
            "stopped_reason": output.stopped_reason,
            "num_turns": traj.num_turns,
            "num_tool_calls": traj.num_tool_calls,
            "prompt_tokens": traj.prompt_tokens,
            "completion_tokens": traj.completion_tokens,
            "total_tokens": traj.total_tokens,
        }
        if req.include_trajectory:
            resp["trajectory"] = asdict(traj)
        return resp

    return app


# Module-level app so `uvicorn agent.service:app` works.
app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    svc = cfg.get("service", {})
    host = os.environ.get("HOST", svc.get("host", "0.0.0.0"))
    port = int(os.environ.get("PORT", svc.get("port", 8080)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
