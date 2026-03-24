from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.ingest.pipeline import ingest_document
from src.schemas.response import DeleteDocumentResponse, IngestResponse
from src.services import weaviate_client

router = APIRouter()
logger = logging.getLogger(__name__)

_SUPPORTED_TYPES = {".pdf", ".txt", ".md"}


@router.post("", response_model=IngestResponse, status_code=201)
async def ingest(
    file: UploadFile = File(..., description="Document to ingest (.pdf, .txt, .md)"),  # noqa: B008
    doc_id: str = Form(default="", description="Stable document ID (auto-generated if empty)"),
    doc_title: str = Form(default="", description="Human-readable document title"),
) -> IngestResponse:
    """
    Ingest a document into the vector store.

    Supported formats: PDF, plain text (.txt), Markdown (.md).
    Chunked into 500–800 token windows with 100 token overlap.
    Each chunk is embedded via Gemini multimodal and stored in Weaviate.
    """
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()

    if suffix not in _SUPPORTED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(_SUPPORTED_TYPES)}",
        )

    resolved_doc_id = doc_id.strip() or str(uuid.uuid4())
    resolved_title = doc_title.strip() or Path(filename).stem

    # Write upload to a temp file — ingest pipeline needs a real path
    content = await file.read()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        result = await ingest_document(
            path=tmp_path,
            doc_id=resolved_doc_id,
            doc_title=resolved_title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Ingest error for doc_id=%s: %s", resolved_doc_id, exc)
        raise HTTPException(status_code=500, detail="Ingest failed. Check server logs.") from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        doc_id=result["doc_id"],
        chunk_count=result["chunk_count"],
        doc_title=result["doc_title"],
    )


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_ingested_document(doc_id: str) -> DeleteDocumentResponse:
    """
    Delete all indexed chunks for a document ID from Weaviate.
    """
    resolved_doc_id = doc_id.strip()
    if not resolved_doc_id:
        raise HTTPException(status_code=422, detail="doc_id must be a non-empty string.")

    try:
        deleted = weaviate_client.delete_document(resolved_doc_id)
    except Exception as exc:
        logger.error("Delete error for doc_id=%s: %s", resolved_doc_id, exc)
        raise HTTPException(status_code=500, detail="Delete failed. Check server logs.") from exc

    return DeleteDocumentResponse(doc_id=resolved_doc_id, deleted_chunks=deleted)
