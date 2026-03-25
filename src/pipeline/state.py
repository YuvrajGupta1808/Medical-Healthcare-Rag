from __future__ import annotations

from typing import Any, TypedDict

from src.schemas.query import Modality


class Citation(TypedDict):
    chunk_id: str
    doc_id: str
    quote: str
    doc_title: str
    page: int | None
    section: str | None
    storage_ref: str | None
    modality_type: str | None


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    modality_type: str   # "text" | "image" | "audio" | "video"
    page: int | None
    section: str
    caption: str
    score: float
    storage_ref: str | None


class RAGState(TypedDict, total=False):
    # ---- input (Step 1) ----
    query_text: str
    modality_type: str        # "text" | "image" | "audio" | "video"
    image_bytes: bytes | None
    audio_ref: str | None
    doc_id: str | None        # optional: filter retrieval to one doc

    # ---- input router fields (Step 2) ----
    request_id: str
    original_question: str
    modality: Modality        # Modality enum set by InputRouter
    modality_metadata: dict[str, Any]
    storage_ref: str | None   # object-storage key for binary uploads

    # ---- retrieval ----
    retrieved_chunks: list[Chunk]

    # ---- generation ----
    answer: str
    citations: list[Citation]

    # ---- output routing signals ----
    prefer_retrieved_image: bool
    include_pdf_export: bool
    image_url: str | None
    pdf_url: str | None
    confidence_score: float | None
    low_confidence: bool | None

    # ---- misc ----
    metadata: dict[str, Any]
    error: str | None
