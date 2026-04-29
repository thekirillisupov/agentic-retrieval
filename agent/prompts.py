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
- local_search(query: str, top_k: int = 10) — returns a list of documents with \
doc_id, title, text, and relevance score.

You may call local_search multiple times with different queries as needed. When you \
have enough information, return a ranked list of doc_ids from most to least relevant.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""

PROMPT_V1_ru = """You are a retrieval agent. Given a conversation, find \
    ALL documents in the corpus that could help answer the user's latest message, \
    using the full conversation as context.

You have one tool:
- local_search(query: str, top_k: int = 10) — returns a list of documents with \
doc_id, title, text, and relevance score.

You may call local_search multiple times with different queries as needed. When you \
have enough information, return a ranked list of doc_ids from most to least relevant.

The corpus and user queries are in Russian. \
Generate local_search queries in Russian to match the corpus language.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

"""


PROMPTS: dict[str, str] = {
    "v1": PROMPT_V1,
    "v1_ru": PROMPT_V1_ru,
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


def get_prompt(version: str) -> str:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}")
    return PROMPTS[version]
