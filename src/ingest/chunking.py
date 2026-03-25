from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import tiktoken

from src.ingest.loaders import RawPage

logger = logging.getLogger(__name__)

_ENCODING = "cl100k_base"
DEFAULT_CHUNK_SIZE = 600   # target tokens per chunk (within 500–800 range)
DEFAULT_OVERLAP = 100      # overlap tokens between consecutive chunks
MIN_CHUNK_TOKENS = 50      # micro-chunks below this threshold are discarded


@dataclass
class IngestChunk:
    """A single chunk ready for embedding and Weaviate upsert."""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    modality_type: str  # "text" or "image"
    page: int | None
    section: str
    caption: str
    storage_ref: str | None
    attachment_id: str
    image_bytes: bytes | None = None


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Deterministic, short chunk ID based on doc_id + sequential index.
    Uses first 16 hex chars of SHA-256 — collision probability negligible at scale.
    """
    raw = f"{doc_id}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_pages(
    pages: list[RawPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[IngestChunk]:
    """
    Sliding-window token-aware chunker.

    Splits each page into overlapping windows of ``chunk_size`` tokens with
    ``overlap`` token stride.  Chunks below ``min_tokens`` are discarded to
    prevent micro-fragments from polluting the retrieval index.

    Args:
        pages:      RawPage objects from a loader.
        chunk_size: Target maximum tokens per chunk (default 600, within 500–800).
        overlap:    Token overlap between consecutive windows (default 100).
        min_tokens: Minimum tokens a chunk must have to be kept (default 50).

    Returns:
        Ordered list of IngestChunk objects with all metadata fields populated.
    """
    enc = tiktoken.get_encoding(_ENCODING)
    chunks: list[IngestChunk] = []
    chunk_index = 0   # global counter across all pages — keeps IDs deterministic
    discarded = 0

    for page in pages:
        if page.modality_type == "image" and page.image_bytes:
            chunks.append(
                IngestChunk(
                    chunk_id=make_chunk_id(page.doc_id, chunk_index),
                    doc_id=page.doc_id,
                    doc_title=page.doc_title,
                    text="",
                    modality_type="image",
                    page=page.page_number,
                    section=page.section,
                    caption=f"Image from {page.doc_title} (Page {page.page_number})",
                    storage_ref=None,
                    attachment_id="",
                    image_bytes=page.image_bytes,
                )
            )
            chunk_index += 1
            continue

        tokens = enc.encode(page.text)
        if not tokens:
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens).strip()

            if chunk_text and len(chunk_tokens) >= min_tokens:
                chunks.append(
                    IngestChunk(
                        chunk_id=make_chunk_id(page.doc_id, chunk_index),
                        doc_id=page.doc_id,
                        doc_title=page.doc_title,
                        text=chunk_text,
                        modality_type="text",
                        page=page.page_number,
                        section=page.section,
                        caption="",
                        storage_ref=None,
                        attachment_id="",
                    )
                )
                chunk_index += 1
            elif chunk_text:
                discarded += 1
                logger.debug(
                    "Discarded micro-chunk (%d tokens < min=%d) for doc_id=%s",
                    len(chunk_tokens), min_tokens, page.doc_id,
                )

            if end == len(tokens):
                break
            start += chunk_size - overlap

    if pages:
        logger.info(
            "Chunked %d page(s) → %d chunks, %d discarded as micro-chunks "
            "(doc_id=%s, size=%d, overlap=%d, min=%d)",
            len(pages), len(chunks), discarded,
            pages[0].doc_id, chunk_size, overlap, min_tokens,
        )
    return chunks


def validate_chunks(chunks: list[IngestChunk], chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Quality-check a list of IngestChunk objects.

    Verifies:
    - No empty text
    - All required metadata fields populated (doc_id, chunk_id, doc_title)
    - Token count does not exceed ``chunk_size``
    - No duplicate chunk_ids

    Returns:
        List of human-readable error strings.  Empty = all checks passed.
    """
    enc = tiktoken.get_encoding(_ENCODING)
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, chunk in enumerate(chunks):
        prefix = f"chunk[{i}] (id={chunk.chunk_id!r})"

        if chunk.modality_type != "image" and not chunk.text.strip():
            errors.append(f"{prefix}: empty text")
        if not chunk.doc_id:
            errors.append(f"{prefix}: missing doc_id")
        if not chunk.chunk_id:
            errors.append(f"{prefix}: missing chunk_id")
        if not chunk.doc_title:
            errors.append(f"{prefix}: missing doc_title")

        token_count = len(enc.encode(chunk.text))
        if token_count > chunk_size:
            errors.append(
                f"{prefix}: token count {token_count} exceeds chunk_size {chunk_size}"
            )

        if chunk.chunk_id in seen_ids:
            errors.append(f"{prefix}: duplicate chunk_id")
        seen_ids.add(chunk.chunk_id)

    return errors
