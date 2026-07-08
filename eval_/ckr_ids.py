"""CKR eval id helpers.

Gold labels in ``data/ckr_eval/*.jsonl`` use chunk ids
``{document_id}_{passage_index}``, matching the FAISS index built in
``indexing/compose_ckr_index.py`` (``doc_id = f"{document_id}_{index}"``).

IMPORTANT: ``document_id`` and ``file_name`` (the source HTML filename) are
*independent* UUIDs in ``data/ckr_eval/ckr_index.jsonl`` -- the stem of
``file_name`` is never equal to ``document_id``. GigaSearch hits only carry
``file_name``/``index`` (plus its own ``faq_id`` as ``doc_id``), so
reconstructing the gold-compatible chunk id requires looking up the real
``document_id`` for a given ``file_name`` via the CKR corpus, not just
stripping ``.html`` off ``file_name``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CKR_CHUNK_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+$",
    re.IGNORECASE,
)
_CLIENT_TAG_RE = re.compile(r"^<client>\s*(.*?)\s*</client>$", re.DOTALL)


def strip_client_tags(text: str) -> str:
    """Remove ``<client>...</client>`` wrapper from eval questions."""
    text = str(text).strip()
    match = _CLIENT_TAG_RE.match(text)
    if match:
        return match.group(1).strip()
    return text

DEFAULT_CKR_INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ckr_eval" / "ckr_index.jsonl"
)

_file_name_to_document_id_cache: dict[str, str] | None = None


def file_stem(file_name: str | None) -> str | None:
    if not file_name:
        return None
    name = str(file_name).strip()
    for suffix in (".html", ".htm"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_file_name_to_document_id(
    path: str | Path = DEFAULT_CKR_INDEX_PATH,
) -> dict[str, str]:
    """Build ``file_name -> document_id`` map from the CKR corpus.

    ``file_name`` and ``document_id`` are unrelated UUIDs for the same row in
    ``data/ckr_eval/ckr_index.jsonl``, so this lookup is the only reliable way
    to recover the gold-compatible ``document_id`` from a GigaSearch hit's
    ``file_name``.
    """
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            file_name = row.get("file_name")
            document_id = row.get("document_id")
            if file_name and document_id:
                mapping[str(file_name)] = str(document_id)
    return mapping


def get_file_name_to_document_id_map() -> dict[str, str]:
    """Lazily load and cache the ``file_name -> document_id`` map."""
    global _file_name_to_document_id_cache
    if _file_name_to_document_id_cache is None:
        _file_name_to_document_id_cache = load_file_name_to_document_id()
    return _file_name_to_document_id_cache


def ckr_chunk_id(
    file_name: str,
    index: int | str,
    *,
    file_name_to_document_id: dict[str, str] | None = None,
) -> str | None:
    """Build gold-compatible chunk id ``{document_id}_{index}`` for a CKR hit.

    ``document_id`` is looked up from ``file_name`` via
    ``file_name_to_document_id`` (defaults to the cached mapping loaded from
    ``data/ckr_eval/ckr_index.jsonl``). Returns ``None`` if ``file_name`` is
    not found in the mapping (e.g. corpus mismatch) rather than silently
    producing a wrong id.
    """
    name = str(file_name).strip() if file_name else ""
    if not name:
        return None
    mapping = (
        file_name_to_document_id
        if file_name_to_document_id is not None
        else get_file_name_to_document_id_map()
    )
    document_id = mapping.get(name)
    if not document_id:
        return None
    return f"{document_id}_{int(index)}"


def is_ckr_chunk_id(doc_id: str) -> bool:
    return bool(_CKR_CHUNK_RE.match(doc_id))


def parse_tool_hits(content: str) -> list[dict[str, Any]]:
    if not content:
        return []
    json_part = content.split("\n[calls used:")[0].strip()
    if not json_part:
        return []
    parsed = json.loads(json_part)
    if not isinstance(parsed, list):
        return []
    return parsed


def _iter_resolved_tool_hits(
    trajectory: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(real_doc_id, hit_dict)`` from search/grep/get_neighbours tools."""
    resolved: list[tuple[str, dict[str, Any]]] = []
    tool_calls = trajectory.get("tool_calls") or []
    tool_messages = [
        m for m in (trajectory.get("messages_full") or []) if m.get("role") == "tool"
    ]

    for tc, msg in zip(tool_calls, tool_messages):
        if tc.get("tool") not in ("search", "grep", "get_neighbours"):
            continue
        try:
            hits = parse_tool_hits(msg.get("content") or "")
        except json.JSONDecodeError:
            continue

        top_doc_ids = (tc.get("result_summary") or {}).get("top_doc_ids") or []
        new_i = 0
        for hit in hits:
            if hit.get("seen"):
                continue

            raw_doc = hit.get("doc_id")
            if isinstance(raw_doc, int):
                doc_id = top_doc_ids[new_i] if new_i < len(top_doc_ids) else None
            elif raw_doc is not None:
                doc_id = str(raw_doc).strip()
            else:
                doc_id = top_doc_ids[new_i] if new_i < len(top_doc_ids) else None
            new_i += 1

            if not doc_id:
                continue
            resolved.append((doc_id, hit))

    return resolved


def build_doc_id_metadata_map(
    trajectory: dict[str, Any],
    *,
    file_name_to_document_id: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map harness ``doc_id`` (faq_id) -> ``file_name``, ``index``, ``chunk_id``.

    ``chunk_id`` matches gold labels in ``data/ckr_eval/*.jsonl``
    (``{document_id}_{passage_index}``, with ``document_id`` looked up from
    ``file_name`` -- see ``ckr_chunk_id``).
    """
    mapping: dict[str, dict[str, Any]] = {}
    for doc_id, hit in _iter_resolved_tool_hits(trajectory):
        file_name = hit.get("file_name")
        index = hit.get("index")
        meta: dict[str, Any] = {}
        if file_name is not None:
            meta["file_name"] = str(file_name)
        if index is not None:
            meta["index"] = int(index)
        chunk_id = None
        if file_name is not None and index is not None:
            chunk_id = ckr_chunk_id(
                str(file_name),
                index,
                file_name_to_document_id=file_name_to_document_id,
            )
        if chunk_id:
            meta["chunk_id"] = chunk_id
        elif is_ckr_chunk_id(doc_id):
            meta["chunk_id"] = doc_id
        if meta:
            mapping[doc_id] = meta
    return mapping


def build_doc_id_to_chunk_map(trajectory: dict[str, Any]) -> dict[str, str]:
    """Map harness ``doc_id`` (faq_id or chunk id) -> CKR gold chunk id."""
    return {
        doc_id: str(meta["chunk_id"])
        for doc_id, meta in build_doc_id_metadata_map(trajectory).items()
        if meta.get("chunk_id")
    }


def ranked_chunk_ids(
    response: dict[str, Any],
    *,
    metadata_map: dict[str, dict[str, Any]] | None = None,
) -> list[str | None]:
    """CKR chunk ids for each ranked position (``None`` when metadata missing)."""
    meta = metadata_map or build_doc_id_metadata_map(response.get("trajectory") or {})
    chunk_map = build_doc_id_to_chunk_map(response.get("trajectory") or {})

    ranked_doc_ids = list(response.get("ranked_doc_ids") or [])
    if not ranked_doc_ids:
        ranked_doc_ids = [
            str(p["doc_id"])
            for p in (response.get("ranked_passages") or [])
            if p.get("doc_id") is not None
        ]

    out: list[str | None] = []
    for doc_id in ranked_doc_ids:
        doc_id = str(doc_id).strip()
        chunk_id = meta.get(doc_id, {}).get("chunk_id") or chunk_map.get(doc_id)
        if chunk_id:
            out.append(str(chunk_id))
        elif is_ckr_chunk_id(doc_id):
            out.append(doc_id)
        else:
            out.append(None)
    return out


def ranked_match_sets(response: dict[str, Any]) -> list[set[str]]:
    """One set of CKR-matchable ids per ranked position."""
    chunk_map = build_doc_id_to_chunk_map(response.get("trajectory") or {})
    items: list[set[str]] = []

    ranked_doc_ids = list(response.get("ranked_doc_ids") or [])
    if not ranked_doc_ids:
        ranked_doc_ids = [
            str(p["doc_id"])
            for p in (response.get("ranked_passages") or [])
            if p.get("doc_id") is not None
        ]

    for doc_id in ranked_doc_ids:
        doc_id = str(doc_id).strip()
        ids: set[str] = {doc_id}
        chunk_id = chunk_map.get(doc_id)
        if chunk_id:
            ids.add(chunk_id)
        elif is_ckr_chunk_id(doc_id):
            ids.add(doc_id)
        items.append(ids)

    return items
