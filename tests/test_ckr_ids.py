"""Tests for CKR faq_id -> chunk_id mapping helpers."""

from __future__ import annotations

import json

from eval_.ckr_ids import (
    build_doc_id_metadata_map,
    build_doc_id_to_chunk_map,
    ckr_chunk_id,
    ranked_chunk_ids,
)


def _search_trajectory() -> dict:
    hits = [
        {
            "doc_id": 0,
            "file_name": "2060c239-c4e2-463d-8524-047a90338dea.html",
            "index": 2,
        },
        {
            "doc_id": 2,
            "file_name": "2060c239-c4e2-463d-8524-047a90338dea.html",
            "index": 1,
        },
    ]
    top_doc_ids = [
        "0a72d8c6-90ee-4e9b-8382-45da7f933dff",
        "b5fee775-ae03-4947-b3ee-68594202d917",
    ]
    return {
        "tool_calls": [
            {
                "tool": "search",
                "result_summary": {"top_doc_ids": top_doc_ids},
            }
        ],
        "messages_full": [
            {
                "role": "tool",
                "content": json.dumps(hits) + "\n[calls used: 1/3]",
            }
        ],
    }


# ``file_name`` (source HTML filename) and ``document_id`` are independent
# UUIDs in data/ckr_eval/ckr_index.jsonl -- never equal to each other. Tests
# use an explicit mapping so they don't depend on loading the real (89MB)
# corpus file.
_FILE_NAME_TO_DOCUMENT_ID = {
    "2060c239-c4e2-463d-8524-047a90338dea.html": "7d88c5ef-3fad-4c2a-acc9-d949377a7023",
}


def test_ckr_chunk_id_looks_up_document_id_from_file_name():
    assert ckr_chunk_id(
        "2060c239-c4e2-463d-8524-047a90338dea.html",
        1,
        file_name_to_document_id=_FILE_NAME_TO_DOCUMENT_ID,
    ) == "7d88c5ef-3fad-4c2a-acc9-d949377a7023_1"


def test_ckr_chunk_id_returns_none_for_unknown_file_name():
    assert ckr_chunk_id("unknown-file.html", 1, file_name_to_document_id={}) is None


def test_build_doc_id_metadata_map_resolves_serial_ids():
    meta = build_doc_id_metadata_map(
        _search_trajectory(), file_name_to_document_id=_FILE_NAME_TO_DOCUMENT_ID
    )
    assert meta["0a72d8c6-90ee-4e9b-8382-45da7f933dff"] == {
        "file_name": "2060c239-c4e2-463d-8524-047a90338dea.html",
        "index": 2,
        "chunk_id": "7d88c5ef-3fad-4c2a-acc9-d949377a7023_2",
    }
    assert meta["b5fee775-ae03-4947-b3ee-68594202d917"]["chunk_id"] == (
        "7d88c5ef-3fad-4c2a-acc9-d949377a7023_1"
    )


def test_build_doc_id_to_chunk_map():
    # build_doc_id_to_chunk_map has no override param, so patch the module
    # cache instead of hitting the real corpus file.
    import eval_.ckr_ids as ckr_ids_module

    ckr_ids_module._file_name_to_document_id_cache = _FILE_NAME_TO_DOCUMENT_ID
    try:
        chunk_map = build_doc_id_to_chunk_map(_search_trajectory())
    finally:
        ckr_ids_module._file_name_to_document_id_cache = None
    assert chunk_map["0a72d8c6-90ee-4e9b-8382-45da7f933dff"] == (
        "7d88c5ef-3fad-4c2a-acc9-d949377a7023_2"
    )


def test_ranked_chunk_ids():
    import eval_.ckr_ids as ckr_ids_module

    response = {
        "ranked_doc_ids": [
            "0a72d8c6-90ee-4e9b-8382-45da7f933dff",
            "b5fee775-ae03-4947-b3ee-68594202d917",
            "missing-faq-id",
        ],
        "trajectory": _search_trajectory(),
    }
    ckr_ids_module._file_name_to_document_id_cache = _FILE_NAME_TO_DOCUMENT_ID
    try:
        result = ranked_chunk_ids(response)
    finally:
        ckr_ids_module._file_name_to_document_id_cache = None
    assert result == [
        "7d88c5ef-3fad-4c2a-acc9-d949377a7023_2",
        "7d88c5ef-3fad-4c2a-acc9-d949377a7023_1",
        None,
    ]
