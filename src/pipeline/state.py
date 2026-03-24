from __future__ import annotations

from typing import Any, TypedDict


class Citation(TypedDict):
    chunk_id: str
    doc_id: str
    quote: str
    doc_title: str
    page: int | None
    section: str | None


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
    # ---- input ----
    query_text: str
    modality_type: str        # "text" | "image" | "audio" | "video"
    image_bytes: bytes | None
    audio_ref: str | None
    doc_id: str | None        # optional: filter retrieval to one doc

    # ---- retrieval ----
    retrieved_chunks: list[Chunk]

    # ---- generation ----
    answer: str
    citations: list[Citation]

    # ---- output routing signals ----
    prefer_retrieved_image: bool
    include_pdf_export: bool
    image_url: str | None

    # ---- misc ----
    metadata: dict[str, Any]
    error: str | None
