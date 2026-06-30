"""Versioned prompts, their tool schemas, user-turn formatting, and the canonical
tool-result rendering — all keyed off a single registry.

A "prompt" is not just a system string: it implies which tools the model is told
about, how user turns are wrapped, and how tool output is rendered back. Those are
trained/evaluated together and must never drift apart, so each one is bundled into a
single ``PromptProfile`` and registered in ``PROFILES``.

Adding a prompt = add one ``PromptProfile`` to ``PROFILES``.
Pointing a source at a prompt = add one line to ``SOURCE_DEFAULT_PROFILE``.
Editing a prompt = edit its string; its tools/formatting travel with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Prompt strings
# --------------------------------------------------------------------------- #

PROMPT_V2_search_only = """You are a retrieval agent. Given a conversation, find ALL documents in the corpus that could help answer the user's latest message, using the full conversation as context.

Conversation format:
<client>message</client>
<bot>message</bot>
...until the user's latest message.
<client>message</client>

Available Tools:
- search(query: str) — returns a list of documents with doc_id, title, text, and relevance score.

Instructions:
1. Analyze the conversation history to understand the exact intent. 
2. Break down the query into its key concepts and information needs.
3. Execute searches using the available tool. You may call search at most 3 times, so make each query count.
4. When search does not find the relevant documents, try different approaches and strategies.

You have a budget of 3 search calls. Once you have enough information — or have used all 3 searches — return ONLY the ranked list of doc_ids from most to least relevant. Do not include any reasoning or explanation.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""

PROMPT_V2 = """You are a retrieval agent. Given a conversation, find ALL documents in the corpus that could help answer the user's latest message, using the full conversation as context.

Conversation format:
<client>message</client>
<bot>message</bot>
...until the user's latest message.
<client>message</client>

Corpus structure:
The corpus contains two kinds of documents:
- Independent chunks: self-contained passages with no parent document.
- Sliced chunks: passages produced by splitting a larger source document. These carry two extra fields: file_name (the source document they were cut from) and index (their position within that document). Within a single file_name, chunks are ordered by ascending index, so consecutive indices are adjacent pieces of the same document.

Available Tools:
- search(query: str) — returns a list of documents with doc_id, title, text, relevance score, and — for sliced chunks only — file_name and index.
- get_neighbours(doc_id: str) — for a sliced chunk, returns the neighbouring chunks from the same file_name. Has no effect on independent chunks (those without file_name/index).

Instructions:
1. Analyze the conversation history to understand the exact intent.
2. Break down the query into its key concepts and information needs.
3. Execute searches using the available tool.
4. When a relevant sliced chunk looks cut off, or the evidence likely continues into adjacent passages of the same document, use get_neighbours to read the surrounding chunks before judging relevance.
5. When search does not find the relevant documents, try different approaches and strategies.

You have a budget of 3 tool calls in total (search and get_neighbours combined), so use them deliberately. Once you have enough information — or have used all 3 tool calls — return ONLY the ranked list of doc_ids from most to least relevant. Do not include any reasoning or explanation.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>
"""


PROMPT_V1_extending_web = """You are a retrieval subagent in a multi-agent system. Given a conversation, \
    identify and retrieve the most relevant documents that could help answer the user's latest message.

Available Tools:
- search(source: str, query: str, top_k: int = 8) — Hybrid semantic and keyword search across the corpus. Returns a list of documents with [doc_id, relevance score, text].
- grep(source: str, pattern: str, top_k: int = 8) — Exact string search across the corpus. Returns a list of documents that match the pattern [doc_id, text].

Sources:
- local - internal knowledge base
- web - web search

Search Strategy Guidelines:
1. Context Analysis: Analyze the conversation history to understand the exact intent. Break down the query into its key concepts and information needs.
2. Source Selection: Use `local_search` for internal/proprietary knowledge. Use `web_search` if the query requires real-time information or external facts.
3. Multi-turn Search: Plan several distinct, non-overlapping search strategies that approach the question from different angles.

After each round of searches, consider:
- What do I know? List the key topics, themes, or aspects of the question that your currently retrieved documents address.
- What should I search for next? Consider what search approaches, keywords that might yield valuable information.
- Do I have enough information? Given the question's complexity and requirements, do you have sufficient information to help answer it, or are there critical gaps?

Tactics to consider:
- When queries fail, try different approaches or keywords
- Avoid duplicate or redundant searches
- Execute multiple tool calls in parallel when possible
- Focus on gathering as much relevant information as possible; multiple perspectives on the same topic help confirm findings
- Follow explicit textual evidence rather than speculation

Final answer format (must be exact):
<answer>
[local]doc_id_1, doc_id_2, doc_id_3, ...
[web]chunk_id_1, chunk_id_2, chunk_id_3, ...
</answer>
"""


PROMPT_V1_extending = PROMPT_V1_extending_web


# --------------------------------------------------------------------------- #
# Tool schemas
# --------------------------------------------------------------------------- #

# Optional `source` selects which corpus/index to search. Currently NOT exposed
# in any tool schema: routing is pinned per-row by the rollout loop so one
# question stays in one base. Kept here (plus agent_loop's `allow_model_source`)
# so source-selection can be re-enabled later by adding it back to a schema.
_SOURCE_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": (
        "Corpus to search (e.g. 'musique', 'sbol'). Leave empty to search the "
        "corpus this conversation belongs to."
    ),
}

SEARCH_TOOL_SCHEMA_QUERY_ONLY: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search local corpus for documents relevant to the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

GREP_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Exact string search across the corpus. "
            "Scans every document for a literal substring or regex pattern (case-insensitive). "
            "Use for names, identifiers, or phrases that must appear verbatim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Literal string or regex pattern to search for",
                },
                "top_k": {"type": "integer", "default": 8, "maximum": 200},
            },
            "required": ["pattern"],
        },
    },
}

GET_NEIGHBOURS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_neighbours",
        "description": (
            "For a sliced chunk (one with file_name/index), return the neighbouring "
            "chunks from the same file_name within +/-1 of its index. Has no "
            "effect on independent chunks (those without file_name/index)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "doc_id of a previously seen sliced chunk",
                },
            },
            "required": ["doc_id"],
        },
    },
}


# Variant advertised when the harness serializes doc_ids to small integers
# (see ``get_tool_schemas(..., use_id_map=True)``). Identical to
# ``GET_NEIGHBOURS_TOOL_SCHEMA`` except ``doc_id`` is typed integer so
# constrained decoding emits the serial id the model was shown.
GET_NEIGHBOURS_TOOL_SCHEMA_SERIAL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_neighbours",
        "description": (
            "For a sliced chunk (one with file_name/index), return the neighbouring "
            "chunks from the same file_name within +/-1 of its index. Has no "
            "effect on independent chunks (those without file_name/index)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "integer",
                    "description": "Integer id of a previously seen sliced chunk",
                },
            },
            "required": ["doc_id"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Prompt registry — the single source of truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptProfile:
    """Everything tied to one prompt, co-located so it cannot drift apart.

    name          version key referenced by configs (`agent.prompt_version`) and
                  per-row `extra_info["prompt_version"]`.
    system_prompt the system message text.
    tool_schemas  the OpenAI tool-schema list this prompt advertises to the model.
    user_tag      if set (e.g. "client"), user turns are wrapped as
                  <tag>...</tag>; ``None`` passes user content through unchanged.
    """

    name: str
    system_prompt: str
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    user_tag: str | None = None


_PROFILE_LIST: list[PromptProfile] = [
    PromptProfile(
        "v2",
        PROMPT_V2,
        [SEARCH_TOOL_SCHEMA_QUERY_ONLY, GET_NEIGHBOURS_TOOL_SCHEMA],
        user_tag="client",
    ),
    PromptProfile(
        "v2_search_only",
        PROMPT_V2_search_only,
        [SEARCH_TOOL_SCHEMA_QUERY_ONLY],
        user_tag="client",
    ),
]

PROFILES: dict[str, PromptProfile] = {p.name: p for p in _PROFILE_LIST}

# Default prompt used when a source/version is unknown.
DEFAULT_PROFILE = "v2"

# Which prompt each source uses by default. A config's `agent.prompt_version` or a
# per-row `extra_info["prompt_version"]` still overrides this (kept for experiments).
# Add a new source + its prompt here in one line.
SOURCE_DEFAULT_PROFILE: dict[str, str] = {
    "sbol": "v2_search_only",
    "rnd": "v2_search_only",
    "musique": "v2_search_only",
    "ckr": "v2",
}

# Backwards-compatible derived views (some callers/docs read these directly).
PROMPTS: dict[str, str] = {name: p.system_prompt for name, p in PROFILES.items()}


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #


def get_prompt(version: str) -> str:
    """Return the system prompt for a version (raises on unknown version)."""
    profile = PROFILES.get(version)
    if profile is None:
        raise ValueError(f"Unknown prompt version: {version}")
    return profile.system_prompt


def get_tool_schemas(
    prompt_version: str, use_id_map: bool = False
) -> list[dict[str, Any]]:
    """Return the OpenAI tool-schema list for a given prompt version.

    When ``use_id_map`` is set the harness has replaced doc_id strings with
    small integers, so any ``get_neighbours`` schema is swapped for its
    integer-typed variant to keep the advertised arg type in sync with what the
    model actually sees.
    """
    profile = PROFILES.get(prompt_version)
    schemas = profile.tool_schemas if profile else [SEARCH_TOOL_SCHEMA_QUERY_ONLY]
    if use_id_map:
        schemas = [
            (
                GET_NEIGHBOURS_TOOL_SCHEMA_SERIAL
                if s.get("function", {}).get("name") == "get_neighbours"
                else s
            )
            for s in schemas
        ]
    return schemas


def profile_for_source(source: str) -> str:
    """Return the default prompt version for a corpus source (e.g. 'sbol')."""
    return SOURCE_DEFAULT_PROFILE.get(source, DEFAULT_PROFILE)


def format_user_content(content: str, prompt_version: str) -> str:
    """Wrap a user turn in the tag the prompt version expects.

    If the profile sets ``user_tag`` (e.g. "client"), wrap as <tag>...</tag>,
    migrating legacy <user>...</user> wrapping. Otherwise pass through unchanged.
    """
    profile = PROFILES.get(prompt_version)
    tag = profile.user_tag if profile else None
    if not tag:
        return content

    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    stripped = content.strip()
    if stripped.startswith(open_tag) and stripped.endswith(close_tag):
        return content
    if stripped.startswith("<user>") and stripped.endswith("</user>"):
        inner = stripped[len("<user>") : -len("</user>")]
        return f"{open_tag}{inner}{close_tag}"
    return f"{open_tag}{content}{close_tag}"


# --------------------------------------------------------------------------- #
# Tool-result rendering (canonical, shared across prompts)
# --------------------------------------------------------------------------- #


def _chunk_locator(file_name: Any, index: Any) -> str:
    """' | file: foo.md #3' for sliced chunks, '' for independent/legacy docs."""
    if file_name is None or index is None:
        return ""
    return f" | file: {file_name} #{index}"


def render_search_result(
    doc_id: str,
    title: str,
    text: str,
    score: float,
    file_name: Any = None,
    index: Any = None,
) -> str:
    """Canonical per-result rendering. Versioned with the prompt."""
    locator = _chunk_locator(file_name, index)
    return f"[doc_id: {doc_id} | score: {score:.2f}{locator}]\nTitle: {title}\n\n{text}"


def render_search_results(results: list[dict[str, Any]]) -> str:
    """Render the full /local_search response into a single tool-message string."""
    if not results:
        return "No results."
    blocks = [
        render_search_result(
            r["doc_id"],
            r["title"],
            r["text"],
            r["score"],
            r.get("file_name"),
            r.get("index"),
        )
        for r in results
    ]
    return "\n\n---\n\n".join(blocks)


def render_search_results_json(results: list[dict[str, Any]]) -> str:
    """Render search results as a JSON array so the model keeps structured context.

    Entries without a ``text`` field are rendered compactly as
    ``{"doc_id": ..., "seen": true}``. This is used for documents already shown
    in full earlier in the trajectory: their rank position is preserved (so the
    model still sees where they placed for this query) without repeating the
    passage text, which keeps the context small.
    """
    if not results:
        return json.dumps([])

    def _entry(r: dict[str, Any]) -> dict[str, Any]:
        if "text" not in r:
            return {"doc_id": r["doc_id"], "seen": True}
        entry: dict[str, Any] = {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "text": r["text"],
            "score": round(float(r["score"]), 4),
        }
        # Surface adjacency only when present so the model knows get_neighbours
        # is meaningful for this hit.
        if r.get("file_name") is not None and r.get("index") is not None:
            entry["file_name"] = r["file_name"]
            entry["index"] = r["index"]
        return entry

    return json.dumps([_entry(r) for r in results], ensure_ascii=False)


def apply_tool_response_truncation(
    rendered: str,
    max_len: int,
    truncate_side: str = "left",
) -> str:
    """Truncate a tool response string to *max_len* characters."""
    if len(rendered) <= max_len:
        return rendered
    side = (truncate_side or "right").lower()
    if side == "left":
        return rendered[:max_len] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + rendered[-max_len:]
    half = max_len // 2
    return rendered[:half] + "...(truncated)..." + rendered[-half:]


def fit_search_display_to_budget(
    results: list[dict[str, Any]],
    max_len: int,
    truncate_side: str = "left",
) -> tuple[list[dict[str, Any]], str]:
    """Drop leading/trailing hits until the JSON render fits *max_len*.

    Returns ``(fitted_results, rendered)``. Entries that survive are fully
    visible in *rendered*. If even a single hit overflows the budget, the
    fitted list is empty and *rendered* is truncated mid-entry — callers must
    not treat those documents as fully seen.
    """
    if not results:
        return [], json.dumps([])

    side = (truncate_side or "right").lower()
    working = list(results)

    while working:
        rendered = render_search_results_json(working)
        if len(rendered) <= max_len:
            return working, rendered
        if side == "right":
            working.pop(0)
        else:
            # ``left`` and ``middle`` both drop lowest-priority trailing hits.
            working.pop()

    rendered = apply_tool_response_truncation(
        render_search_results_json(results), max_len, truncate_side
    )
    return [], rendered


def render_grep_result(doc_id: str, title: str, text: str) -> str:
    """Canonical per-result rendering for grep hits (no score)."""
    return f"[doc_id: {doc_id}]\nTitle: {title}\n\n{text}"


def render_grep_results(results: list[dict[str, Any]], total_matches: int) -> str:
    """Render the full /grep response into a single tool-message string."""
    if not results:
        return "No matches."
    header = f"Showing {len(results)} of {total_matches} match(es)."
    blocks = [render_grep_result(r["doc_id"], r["title"], r["text"]) for r in results]
    return header + "\n\n" + "\n\n---\n\n".join(blocks)


def fit_grep_results_to_budget(
    results: list[dict[str, Any]],
    total_matches: int,
    max_len: int,
    truncate_side: str = "left",
) -> tuple[list[dict[str, Any]], str]:
    """Drop leading/trailing grep hits until the render fits *max_len*.

    Mirrors :func:`fit_search_display_to_budget`: returns ``(fitted, rendered)``
    where every surviving hit is fully visible in *rendered*. If even a single
    hit overflows the budget the fitted list is empty and *rendered* is
    truncated mid-block — callers must not treat those docs as fully seen.
    """
    if not results:
        return [], render_grep_results(results, total_matches)

    side = (truncate_side or "right").lower()
    working = list(results)

    while working:
        rendered = render_grep_results(working, total_matches)
        if len(rendered) <= max_len:
            return working, rendered
        if side == "right":
            working.pop(0)
        else:
            working.pop()

    rendered = apply_tool_response_truncation(
        render_grep_results(results, total_matches), max_len, truncate_side
    )
    return [], rendered


def render_neighbours_results(
    results: list[dict[str, Any]],
    status: str,
    anchor_doc_id: str,
    window: int,
) -> str:
    """Render a /get_neighbours response, explaining no-op cases to the model.

    ``status`` comes from the tool server (see FaissIndex.get_neighbours). The
    "no_metadata"/"independent_chunk" cases are normal, not errors: the model is
    told plainly so it relies on search instead of retrying get_neighbours.
    Data results are returned as a JSON array for consistency with search.
    """
    if status == "unknown_doc":
        return json.dumps(
            {
                "error": "unknown_doc",
                "message": (
                    f"doc_id '{anchor_doc_id}' was not found in this corpus, so it has no "
                    f"neighbours. Only call get_neighbours on a doc_id returned by search."
                ),
            }
        )
    if status in ("no_metadata", "independent_chunk"):
        return json.dumps(
            {
                "error": status,
                "message": (
                    f"No neighbouring chunks for '{anchor_doc_id}': this document is not part "
                    f"of a sliced source (no file_name/index), so adjacency is unavailable. "
                    f"Rely on search results."
                ),
            }
        )
    if not results:
        return json.dumps([])
    return json.dumps(
        [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "text": r["text"],
                **({"file_name": r["file_name"]} if r.get("file_name") else {}),
                **({"index": r["index"]} if r.get("index") is not None else {}),
            }
            for r in results
        ],
        ensure_ascii=False,
    )
