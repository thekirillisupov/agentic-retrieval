"""Tests for OpenSearch-backed get_neighbours (scripts/DEMO.ipynb parity)."""

from __future__ import annotations

from typing import Any

from agent.opensearch_neighbours import (
    get_neighbours,
    index_id_from_configuration,
)


class _FakeOSClient:
    def __init__(self, docs: dict[str, dict[str, Any]], by_file: dict[str, list[dict]]):
        self.docs = docs
        self.by_file = by_file

    def get(self, *, index: str, id: str) -> dict[str, Any]:
        return {"_source": self.docs[id]}

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        file_name = None
        for clause in body["query"]["bool"]["should"]:
            term = clause.get("term", {})
            for key in ("metadata.file_name", "file_name", "metadata.file_name.keyword", "file_name.keyword"):
                if key in term:
                    file_name = term[key]
                    break
            if file_name is not None:
                break
        hits = self.by_file.get(str(file_name), [])
        return {"hits": {"hits": hits}}


def test_index_id_from_configuration():
    cfg = {
        "agent_configuration": {
            "data_sources_faq": [{"index_id": "idx-abc"}],
        }
    }
    assert index_id_from_configuration(cfg) == "idx-abc"
    assert index_id_from_configuration({}) is None


def test_get_neighbours_returns_adjacent_chunks():
    anchor_id = "doc-2"
    docs = {
        anchor_id: {
            "metadata": {
                "file_name": "guide.pdf",
                "passage_index": 2,
                "breadcrumbs": "Guide",
                "raw_text": "middle",
            }
        },
    }
    by_file = {
        "guide.pdf": [
            {
                "_id": "doc-1",
                "_source": {
                    "metadata": {
                        "file_name": "guide.pdf",
                        "passage_index": 1,
                        "breadcrumbs": "Guide",
                        "raw_text": "before",
                    }
                },
            },
            {
                "_id": anchor_id,
                "_source": docs[anchor_id],
            },
            {
                "_id": "doc-3",
                "_source": {
                    "metadata": {
                        "file_name": "guide.pdf",
                        "passage_index": 3,
                        "breadcrumbs": "Guide",
                        "raw_text": "after",
                    }
                },
            },
        ],
    }
    client = _FakeOSClient(docs, by_file)

    resp = get_neighbours(client, "test-index", anchor_id, window=1)

    assert resp["status"] == "ok"
    assert [r["doc_id"] for r in resp["results"]] == ["doc-1", "doc-3"]
    assert resp["results"][0]["index"] == 1
    assert resp["results"][1]["text"] == "after"


def test_get_neighbours_independent_chunk():
    client = _FakeOSClient(
        {"solo": {"metadata": {"raw_text": "standalone"}}},
        {},
    )
    resp = get_neighbours(client, "test-index", "solo", window=1)
    assert resp["status"] == "independent_chunk"
    assert resp["results"] == []


def test_get_neighbours_unknown_doc():
    class _MissingClient:
        def get(self, *, index: str, id: str) -> dict[str, Any]:
            from opensearchpy.exceptions import NotFoundError

            raise NotFoundError(404, "not found", {})

    resp = get_neighbours(_MissingClient(), "test-index", "missing", window=1)
    assert resp["status"] == "unknown_doc"
