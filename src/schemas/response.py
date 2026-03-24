from __future__ import annotations

from pydantic import BaseModel


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    quote: str
    doc_title: str
    page: int | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    image_url: str | None = None
    query: str


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    doc_title: str
    status: str = "ok"


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted_chunks: int
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    weaviate: str
    version: str = "1.0.0"
