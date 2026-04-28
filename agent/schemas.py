"""Shared agent-side schemas.

Kept independent of OpenAI / pydantic so harness code, eval, and trajectory writer
can share types without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Message:
    """A single chat message, OpenAI-style.

    Stored as a plain dict-shaped record so we can round-trip through JSON
    (trajectory logging) without losing information.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.name is not None:
            out["name"] = self.name
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            out["tool_calls"] = self.tool_calls
        return out

    @classmethod
    def from_openai(cls, msg: dict[str, Any]) -> "Message":
        return cls(
            role=msg["role"],
            content=msg.get("content"),
            name=msg.get("name"),
            tool_call_id=msg.get("tool_call_id"),
            tool_calls=msg.get("tool_calls"),
        )


@dataclass
class AgentInput:
    messages: list[Message]
    max_turns: int = 8
    max_tool_calls: int = 10
    top_k_default: int = 10


@dataclass
class RankedPassage:
    doc_id: str
    title: str
    text: str
    rank: int
    best_score: float | None = None
    first_seen_turn: int = -1
    num_times_retrieved: int = 0


@dataclass
class ToolCallTrace:
    turn: int
    tool: str
    arguments: dict[str, Any]
    result_summary: dict[str, Any]
    latency_ms: int


@dataclass
class Trajectory:
    trajectory_id: str
    timestamp: str
    model: str
    prompt_version: str
    input_messages: list[Message]
    max_turns: int
    max_tool_calls: int
    messages_full: list[Message]
    tool_calls: list[ToolCallTrace]
    ranked_doc_ids: list[str]
    stopped_reason: str
    num_turns: int
    num_tool_calls: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    gold_doc_ids: list[str] = field(default_factory=list)


StoppedReason = Literal["answer", "max_turns", "max_tool_calls", "parse_error"]


@dataclass
class AgentOutput:
    ranked_doc_ids: list[str]
    ranked_passages: list[RankedPassage]
    trajectory: Trajectory
    stopped_reason: StoppedReason


@dataclass
class SearchResult:
    """One row from /local_search response, harness-side mirror."""

    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class SeenPassage:
    """Mutable accumulator for a doc_id seen across multiple tool calls."""

    doc_id: str
    title: str
    text: str
    best_score: float
    first_seen_turn: int
    num_times_retrieved: int = 1

    def update(self, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
        self.num_times_retrieved += 1
