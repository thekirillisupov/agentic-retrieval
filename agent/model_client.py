"""Pluggable model backend for the harness.

The harness drives a tool-calling ReAct loop and consumes an OpenAI-shaped chat
completion (``response.choices[0].message.{content,tool_calls}`` + ``usage``).
This module lets the *backend* be swapped via config without touching the loop:

  * ``openai`` — the default vLLM / OpenAI-compatible server (optionally behind
    mutual TLS).
  * ``http``   — a raw JSON endpoint like::

        curl -X POST https://.../qwen_35 --insecure \
          --cert published.pem --key user.key \
          -d '{"messages": [...], "max_tokens": 256, "temperature": 0.1,
               "top_p": 0.95, "reasoning": false,
               "chat_template_kwargs": {"enable_thinking": false}}'

    ``HttpChatClient`` duck-types ``.chat.completions.create(...)`` so it drops
    straight into ``AgentHarness``: it POSTs that body (with mTLS) and parses an
    OpenAI-shaped response, tolerating minor shape differences.

NOTE: the agent is a tool-using loop, so the ``http`` endpoint must accept
``tools``/``tool_choice`` and return OpenAI-style ``tool_calls`` for search to
work. Set ``send_tools: false`` only for a plain (single-shot) chat endpoint.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
from openai import OpenAI


def _make_httpx_client(tls: dict[str, Any], timeout_s: float) -> httpx.Client:
    """Build an httpx client honouring mutual-TLS / insecure settings.

    tls keys:
      * ``cert_file`` / ``key_file`` -> client certificate (curl --cert/--key)
      * ``verify`` -> True (default), False (curl --insecure), or a CA bundle path
    """
    cert: Any = None
    if tls.get("cert_file") and tls.get("key_file"):
        cert = (tls["cert_file"], tls["key_file"])
    elif tls.get("cert_file"):
        cert = tls["cert_file"]
    verify = tls.get("verify", True)
    return httpx.Client(cert=cert, verify=verify, timeout=timeout_s)


class _Completions:
    def __init__(self, owner: "HttpChatClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> SimpleNamespace:
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner: "HttpChatClient") -> None:
        self.completions = _Completions(owner)


class HttpChatClient:
    """OpenAI-client-shaped wrapper over a raw JSON chat endpoint."""

    def __init__(
        self,
        *,
        url: str,
        tls: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
        include_model: bool = False,
        send_tools: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        # Whether to put "model" in the body (the sample curl omits it).
        self.include_model = include_model
        # Whether to forward tools/tool_choice (required for the agent loop).
        self.send_tools = send_tools
        # Static fields merged into every request body: top_p, reasoning,
        # chat_template_kwargs, … Mirrors the sample curl.
        self.extra_body = dict(extra_body or {})
        self._client = _make_httpx_client(tls or {}, timeout_s)
        self.chat = _Chat(self)

    def _create(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **_ignore: Any,
    ) -> SimpleNamespace:
        body: dict[str, Any] = dict(self.extra_body)
        body["messages"] = messages
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if self.include_model and model is not None:
            body["model"] = model
        if self.send_tools and tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        resp = self._client.post(self.url, json=body)
        resp.raise_for_status()
        return self._parse(resp.json())

    @staticmethod
    def _parse(data: dict[str, Any]) -> SimpleNamespace:
        """Normalize an OpenAI-shaped (or near) response into the object the
        harness reads: ``.choices[0].message.{content,tool_calls}`` + ``.usage``."""
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
        else:
            # Fall back to flatter shapes: {"message": {...}} or {"content": ...}.
            message = data.get("message") or {"content": data.get("content")}

        content = message.get("content")

        tool_calls = None
        raw_tcs = message.get("tool_calls")
        if raw_tcs:
            tool_calls = []
            for i, tc in enumerate(raw_tcs):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                # OpenAI sends arguments as a JSON *string*; accept dicts too.
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                tool_calls.append(
                    SimpleNamespace(
                        id=tc.get("id") or f"call_{i}",
                        type=tc.get("type", "function"),
                        function=SimpleNamespace(name=fn.get("name"), arguments=args),
                    )
                )

        usage_raw = data.get("usage") or {}
        usage = SimpleNamespace(
            prompt_tokens=usage_raw.get("prompt_tokens", 0) or 0,
            completion_tokens=usage_raw.get("completion_tokens", 0) or 0,
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=tool_calls)
                )
            ],
            usage=usage,
        )

    def close(self) -> None:
        self._client.close()


def build_model_client(model_cfg: dict[str, Any]) -> Any:
    """Return a chat client (OpenAI or HttpChatClient) per ``model.backend``."""
    backend = str(model_cfg.get("backend", "openai")).lower()
    tls = model_cfg.get("tls") or {}
    timeout_s = float(model_cfg.get("timeout_s", 120))

    if backend == "openai":
        kwargs: dict[str, Any] = {
            "base_url": model_cfg["vllm_url"],
            "api_key": model_cfg.get("api_key", "EMPTY"),
        }
        if tls:
            # OpenAI-compatible API behind mutual TLS.
            kwargs["http_client"] = _make_httpx_client(tls, timeout_s)
        return OpenAI(**kwargs)

    if backend == "http":
        return HttpChatClient(
            url=model_cfg["url"],
            tls=tls,
            timeout_s=timeout_s,
            include_model=bool(model_cfg.get("include_model", False)),
            send_tools=bool(model_cfg.get("send_tools", True)),
            extra_body=model_cfg.get("extra_body"),
        )

    raise ValueError(f"unknown model.backend: {backend!r} (expected 'openai' or 'http')")
