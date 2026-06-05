"""Versioned prompts and the canonical tool-result rendering.

Both system prompt and tool-result format are bumped together — the model is trained
(implicitly, through few-shot or future SFT) on a specific rendering of tool output,
so they cannot drift independently.
"""

from __future__ import annotations

from typing import Any

PROMPT_V1 = """You are a retrieval agent. Given a conversation, find \
    ALL documents in the corpus that could help answer the user's latest message, \
    using the full conversation as context.

You have one tool:
- local_search(query: str, top_k: int = 8) — returns a list of documents with \
doc_id, title, text, and relevance score.

You may call local_search multiple times with different queries as needed. When you \
have enough information, return a ranked list of doc_ids from most to least relevant.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""

PROMPT_V1_edit = """You are a retrieval agent. Given a conversation, find \
    ALL documents in the corpus that could help answer the user's latest message, \
    using the full conversation as context.

Available Tools:
- local_search(query: str) — returns a list of documents with doc_id, title, text, and relevance score.

Analyze the conversation history to understand the exact intent. Break down the query into its key concepts and information needs.
Plan several distinct, non-overlapping search strategies that approach the question from different angles.
When you have enough information, return a ranked list of doc_ids from most to least relevant.

After each round of searches, consider:
- What do I know? List the key topics, themes, or aspects of the question that your currently retrieved documents address.
- What should I search for next? Consider what search approaches, keywords that might yield valuable information.
- Do I have enough information? Given the question's complexity and requirements, do you have sufficient information to help answer it, or are there critical gaps?

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""

PROMPT_V1_with_grep = """You are a retrieval agent. Given a conversation, find \
    ALL documents in the corpus that could help answer the user's latest message, \
    using the full conversation as context.

Available Tools:
- search(query: str, top_k: int = 8) — Hybrid semantic and keyword search across the corpus. Returns a list of documents with [doc_id, relevance score, text].
- grep(pattern: str, top_k: int = 8) — Exact string search across the corpus. Returns a list of documents that match the pattern [doc_id, text].

You may call tools multiple times with different parameters as needed. When you \
have enough information, return a ranked list of doc_ids from most to least relevant.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""


PROMPT_V1_extending = """You are a retrieval subagent in a multi-agent system. Given a conversation, \
    identify and retrieve the most relevant documents that could help answer the user's latest message.

Available Tools:
- search(query: str, top_k: int = 8) — Hybrid semantic and keyword search across the corpus. Returns a list of documents with [doc_id, relevance score, text].
- grep(pattern: str, top_k: int = 8) — Exact string search across the corpus. Returns a list of documents that match the pattern [doc_id, text].

Search Strategy Guidelines:
- Context Analysis: analyze the conversation history to understand the exact intent. Break down the query into its key concepts and information needs.
- Multi-turn Search: plan several distinct, non-overlapping search strategies that approach the question from different angles.

After each round of searches, consider:
- What do I know? List the key topics, themes, or aspects of the question that your currently retrieved documents address.
- What should I search for next? Consider what search approaches, keywords that might yield valuable information.
- Do I have enough information? Given the question's complexity and requirements, do you have sufficient information to help answer it, or are there critical gaps?

Final answer format - list of doc_ids from most to least relevant (must be exact):
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


PROMPTS: dict[str, str] = {
    "v1": PROMPT_V1,
    "v1_edit": PROMPT_V1_edit,
    "v1_extending": PROMPT_V1_extending,
    "v1_extending_web": PROMPT_V1_extending_web,
    "v1_with_grep": PROMPT_V1_with_grep,
}


LOCAL_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "local_search",
        "description": "Search local corpus for documents relevant to the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10, "maximum": 50},
            },
            "required": ["query"],
        },
    },
}

LOCAL_SEARCH_TOOL_SCHEMA_QUERY_ONLY: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "local_search",
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

SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Hybrid semantic and keyword search across the corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 8, "maximum": 50},
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

_TOOL_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "v1": [LOCAL_SEARCH_TOOL_SCHEMA],
    "v1_edit": [LOCAL_SEARCH_TOOL_SCHEMA_QUERY_ONLY],
    "v1_with_grep": [SEARCH_TOOL_SCHEMA, GREP_TOOL_SCHEMA],
    "v1_extending": [SEARCH_TOOL_SCHEMA, GREP_TOOL_SCHEMA],
}


def get_tool_schemas(prompt_version: str) -> list[dict[str, Any]]:
    """Return the OpenAI tool-schema list for a given prompt version."""
    return _TOOL_SCHEMAS.get(prompt_version, [LOCAL_SEARCH_TOOL_SCHEMA])


def render_search_result(doc_id: str, title: str, text: str, score: float) -> str:
    """Canonical per-result rendering. Versioned with the prompt."""
    return f"[doc_id: {doc_id} | score: {score:.2f}]\n" f"Title: {title}\n\n" f"{text}"


def render_search_results(results: list[dict[str, Any]]) -> str:
    """Render the full /local_search response into a single tool-message string."""
    if not results:
        return "No results."
    blocks = [
        render_search_result(r["doc_id"], r["title"], r["text"], r["score"])
        for r in results
    ]
    return "\n\n---\n\n".join(blocks)


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


def get_prompt(version: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}")
    return PROMPTS[version]
