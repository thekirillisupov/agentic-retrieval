"""Pydantic request/response models for the tool server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocalSearchRequest(BaseModel):
    query: str = Field(
        ..., description="Search query (no E5 prefix; the server adds it)"
    )
    top_k: int = Field(default=10, ge=1, le=50)
    source: str | None = Field(
        default=None,
        description=(
            "Which index to search. None -> server's default_source. "
            "Must match one of the loaded sources when several are configured."
        ),
    )


class SearchHit(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    # Present only for sliced chunks; None for independent chunks / legacy indexes.
    file_name: str | None = None
    index: int | None = None


class LocalSearchResponse(BaseModel):
    results: list[SearchHit]
    latency_ms: int


class LookupByIdRequest(BaseModel):
    doc_ids: list[str]


class Doc(BaseModel):
    doc_id: str
    title: str
    text: str


class LookupByIdResponse(BaseModel):
    docs: list[Doc]


class GrepRequest(BaseModel):
    pattern: str = Field(
        ...,
        description="Literal string or regex pattern to search for (case-insensitive)",
    )
    top_k: int = Field(default=8, ge=1, le=200)
    source: str | None = Field(
        default=None,
        description="Which index to scan. None -> server's default_source.",
    )


class GrepHit(BaseModel):
    doc_id: str
    title: str
    text: str


class GrepResponse(BaseModel):
    results: list[GrepHit]
    latency_ms: int
    total_matches: int


class GetNeighboursRequest(BaseModel):
    doc_id: str = Field(..., description="doc_id of a previously seen sliced chunk")
    window: int = Field(default=1, ge=1, le=10)
    source: str | None = Field(
        default=None,
        description="Which index the doc_id belongs to. None -> server's default_source.",
    )


class NeighbourHit(BaseModel):
    doc_id: str
    title: str
    text: str
    file_name: str | None = None
    index: int | None = None


class GetNeighboursResponse(BaseModel):
    results: list[NeighbourHit]
    # One of: "ok", "unknown_doc", "no_metadata", "independent_chunk". Lets the
    # caller explain to the model why neighbours are (un)available.
    status: str
    anchor_doc_id: str
    window: int
    latency_ms: int


class StatsResponse(BaseModel):
    num_docs: int
    embedder: str
    dim: int
    index_type: str
    default_source: str | None = None
    sources: dict[str, int] = Field(default_factory=dict)
