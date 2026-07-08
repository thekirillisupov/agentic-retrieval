"""OpenSearch-backed ``get_neighbours`` — mirrors scripts/DEMO.ipynb.

Used when ``search.gigasearch.opensearch`` is set in the inference config:
search still goes through GigaSearch ``universal_search``, but chunk adjacency is
resolved by direct mTLS queries against the OpenSearch index that backs the FAQ
corpus.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse


def _pick(src: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        cur: Any = src
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is not None:
            return cur
    return None


def index_id_from_configuration(configuration: dict[str, Any]) -> str | None:
    """``data_sources_faq[0].index_id`` from a GigaSearch agent_configuration."""
    sources = configuration.get("agent_configuration", {}).get("data_sources_faq")
    if not sources:
        return None
    index_id = sources[0].get("index_id")
    return str(index_id) if index_id else None


def build_opensearch_client(cfg: dict[str, Any]) -> Any:
    """Build an mTLS ``OpenSearch`` client from ``search.gigasearch.opensearch``."""
    try:
        from opensearchpy import OpenSearch, RequestsHttpConnection
    except ImportError as e:
        raise ImportError(
            "opensearch-py is required when search.gigasearch.opensearch is set; "
            "pip install opensearch-py"
        ) from e

    host = cfg.get("host")
    port = int(cfg.get("port", 443))
    if not host:
        url = cfg.get("url", "")
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = parsed.hostname
        if parsed.port:
            port = parsed.port
    if not host:
        raise ValueError(
            "search.gigasearch.opensearch requires host or url "
            "(or inherit search.url in service.py)"
        )

    tls = cfg.get("tls") or {}
    client_cert = tls.get("cert_file")
    client_key = tls.get("key_file")

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=False,
        use_ssl=True,
        verify_certs=bool(tls.get("verify", False)),
        ssl_show_warn=False,
        connection_class=RequestsHttpConnection,
        client_cert=client_cert,
        client_key=client_key,
        url_prefix=cfg.get("url_prefix", "/opensearch"),
    )


def _hit_from_source(doc_id: str, src: dict[str, Any]) -> dict[str, Any]:
    chunk_index = _pick(
        src, "metadata.passage_index", "metadata.index", "passage_index", "index"
    )
    return {
        "doc_id": doc_id,
        "title": _pick(src, "metadata.breadcrumbs", "breadcrumbs", "title") or "",
        "text": _pick(src, "metadata.raw_text", "content", "raw_text") or "",
        "file_name": _pick(src, "metadata.file_name", "file_name"),
        "index": int(chunk_index) if chunk_index is not None else None,
    }


def _term_queries(field: str, value: str) -> list[dict[str, Any]]:
    return [
        {"term": {f"{field}.keyword": value}},
        {"term": {field: value}},
    ]


def _search_same_file(
    client: Any,
    index: str,
    file_name: str,
    *,
    file_name_fields: tuple[str, ...] = ("file_name", "metadata.file_name"),
    size: int = 1000,
) -> list[dict[str, Any]]:
    should: list[dict[str, Any]] = []
    for field in file_name_fields:
        should.extend(_term_queries(field, file_name))

    body = {
        "query": {"bool": {"should": should, "minimum_should_match": 1}},
        "size": size,
        "_source": True,
    }
    resp = client.search(index=index, body=body)
    return resp.get("hits", {}).get("hits", [])


def get_neighbours(
    client: Any,
    index: str,
    doc_id: str,
    window: int = 1,
) -> dict[str, Any]:
    """Adjacent chunks for a sliced passage — mirrors tool_server ``get_neighbours``.

    ``doc_id`` is the OpenSearch ``_id`` (same as GigaSearch ``faq_id``).
  ``index`` is the OpenSearch index name (e.g. from ``data_sources_faq[].index_id``).
    """
    from opensearchpy.exceptions import NotFoundError

    t0 = time.perf_counter()
    window = max(0, int(window))

    try:
        anchor = client.get(index=index, id=doc_id)
    except NotFoundError:
        return {
            "results": [],
            "status": "unknown_doc",
            "anchor_doc_id": doc_id,
            "window": window,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    anchor_src = anchor["_source"]
    file_name = _pick(anchor_src, "metadata.file_name", "file_name")
    anchor_index = _pick(
        anchor_src,
        "metadata.passage_index",
        "metadata.index",
        "passage_index",
        "index",
    )

    if file_name is None or anchor_index is None:
        return {
            "results": [],
            "status": "independent_chunk",
            "anchor_doc_id": doc_id,
            "window": window,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    anchor_index = int(anchor_index)
    lo, hi = anchor_index - window, anchor_index + window

    hits = _search_same_file(client, index, str(file_name))
    neighbours: list[dict[str, Any]] = []
    for hit in hits:
        item = _hit_from_source(hit["_id"], hit["_source"])
        idx = item.get("index")
        if idx is None or idx < lo or idx > hi or idx == anchor_index:
            continue
        neighbours.append(item)

    neighbours.sort(key=lambda x: x["index"])

    return {
        "results": neighbours,
        "status": "ok",
        "anchor_doc_id": doc_id,
        "window": window,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }
