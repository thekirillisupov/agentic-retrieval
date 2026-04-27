"""Pydantic request/response models for the tool server."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocalSearchRequest(BaseModel):
    query: str = Field(..., description="Search query (no E5 prefix; the server adds it)")
    top_k: int = Field(default=10, ge=1, le=50)


class SearchHit(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


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


class StatsResponse(BaseModel):
    num_docs: int
    embedder: str
    dim: int
    index_type: str
