from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    VIDEO = "video"


class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    doc_id: str | None = Field(None, description="Restrict retrieval to this document ID")
    top_k: int = Field(default=5, ge=1, le=20, description="Max chunks to retrieve")


class IngestRequest(BaseModel):
    """Used when body metadata is sent alongside a file upload."""
    doc_id: str | None = None
    doc_title: str = ""


# Kept for pre-existing route compatibility (Step 2 contract)
class NormalizedQueryResponse(BaseModel):
    request_id: str
    query_text: str
    original_question: str
    modality: Modality
    storage_ref: str | None = None
    modality_metadata: dict = {}
    answer: str | None = None
    cited_chunk_ids: list[str] = []
