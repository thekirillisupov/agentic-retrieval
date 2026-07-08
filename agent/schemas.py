"""Shared agent-side schemas.

Kept independent of OpenAI / pydantic so harness code, eval, and trajectory writer
can share types without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


def assign_serial_id(
    doc_id: str, id_map: dict[str, int], id_map_inv: list[str]
) -> int:
    """Map a doc_id to a small per-episode integer, assigning on first sight.

    Integers are handed out by order of first appearance (0, 1, 2, …) and are
    stable within an episode: one doc_id always maps to the same int across
    turns and stubs. The serialization layer uses this so the model exchanges
    compact ints instead of long doc_id strings; the harness translates the
    model's int answers back to doc_ids for scoring. ``id_map_inv[i]`` is the
    doc_id for int ``i``; ``id_map`` is the reverse lookup.
    """
    idx = id_map.get(doc_id)
    if idx is None:
        idx = len(id_map_inv)
        id_map[doc_id] = idx
        id_map_inv.append(doc_id)
    return idx


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
    # Corpus this conversation belongs to. Pins every search to the matching
    # index (per-row routing). None -> tool server's default_source. A `source`
    # explicitly emitted by the model in a tool call overrides this.
    source: str | None = None
    # Extra retrieval knobs the caller pins for the whole episode (filters, top_k
    # caps, corpus routing, …). Forwarded verbatim to every search/grep/
    # get_neighbours request; the harness-owned keys (query, top_k, source, …)
    # override them so the model still controls *what* to search. Empty by
    # default -> no behaviour change for existing callers.
    search_params: dict[str, Any] = field(default_factory=dict)


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
    # tool_budget_feedback diagnostic: tool calls emitted after the budget was
    # already spent (each one ignored the "return <answer>" nudge).
    num_over_budget_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    gold_doc_ids: list[str] = field(default_factory=list)
    llm_latencies_ms: list[int] = field(default_factory=list)


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
