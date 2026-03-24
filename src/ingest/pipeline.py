from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from src.ingest.chunking import IngestChunk, chunk_pages, validate_chunks
from src.ingest.loaders import load_document
from src.services import gemini_embed, weaviate_client

logger = logging.getLogger(__name__)


async def ingest_document(
    path: str | Path,
    doc_id: str | None = None,
    doc_title: str = "",
) -> dict:
    """
    Full ingest pipeline for a single document:
      load → chunk → embed (concurrent) → batch upsert to Weaviate

    Args:
        path: path to the document file (.pdf, .txt, .md)
        doc_id: optional stable ID; generated as UUID4 if not provided
        doc_title: human-readable title stored with every chunk

    Returns:
        dict with keys: doc_id, chunk_count, doc_title
    """
    path = Path(path)
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    # ---- 1. Load ----
    pages = load_document(path, doc_id=doc_id, doc_title=doc_title)
    if not pages:
        raise ValueError(f"No content extracted from '{path}'. File may be empty or image-only PDF.")

    # ---- 2. Chunk ----
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"Chunking produced zero chunks for '{path}'.")

    # ---- 3. Quality gate ----
    errors = validate_chunks(chunks)
    if errors:
        raise ValueError(
            f"Chunk quality check failed for '{path}' ({len(errors)} error(s)): "
            + "; ".join(errors[:5])  # surface first 5 to keep message readable
        )
    logger.info(
        "Chunk quality OK: %d chunks, token range [%s–%s] for doc_id=%s",
        len(chunks),
        min(len(c.text.split()) for c in chunks),
        max(len(c.text.split()) for c in chunks),
        doc_id,
    )

    # ---- 4. Embed all chunks concurrently ----
    logger.info("Embedding %d chunks for doc_id=%s …", len(chunks), doc_id)
    embeddings: list[list[float]] = await asyncio.gather(
        *[gemini_embed.embed_text(chunk.text) for chunk in chunks]
    )

    # ---- 5. Idempotent upsert: delete stale chunks, then insert fresh ones ----
    loop = asyncio.get_running_loop()
    deleted = await loop.run_in_executor(None, weaviate_client.delete_document, doc_id)
    if deleted:
        logger.info("Removed %d stale chunk(s) for doc_id=%s before re-ingest", deleted, doc_id)

    upserted = await loop.run_in_executor(
        None,
        _batch_upsert_sync,
        chunks,
        embeddings,
    )

    logger.info(
        "Ingest complete: doc_id=%s, doc_title='%s', chunks=%d",
        doc_id, doc_title, upserted,
    )
    return {"doc_id": doc_id, "chunk_count": upserted, "doc_title": doc_title or path.stem}


# ---------------------------------------------------------------------------
# Synchronous Weaviate batch insert (must run in thread pool from async context)
# ---------------------------------------------------------------------------

def _batch_upsert_sync(chunks: list[IngestChunk], embeddings: list[list[float]]) -> int:
    """Insert chunks with their embeddings into Weaviate using dynamic batch."""
    collection = weaviate_client.get_client().collections.get(
        weaviate_client.COLLECTION_NAME
    )
    upserted = 0
    with collection.batch.dynamic() as batch:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            batch.add_object(
                properties={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "modality_type": chunk.modality_type,
                    "text": chunk.text,
                    "caption": chunk.caption,
                    "page": chunk.page,
                    "section": chunk.section,
                    "attachment_id": chunk.attachment_id,
                    "storage_ref": chunk.storage_ref,
                },
                vector=embedding,
            )
            upserted += 1
    return upserted
