from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.pipeline.graph import rag_pipeline
from src.router.input_router import build_initial_state
from src.schemas.query import TextQueryRequest
from src.schemas.response import CitationOut, QueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", response_model=QueryResponse)
async def query(request: TextQueryRequest) -> QueryResponse:
    """
    Query the RAG pipeline with a text question.
    Runs the full graph: retrieve → generate → output_route.
    Returns a grounded answer with source citations.
    """
    initial_state = build_initial_state(
        query_text=request.query,
        doc_id=request.doc_id,
    )

    try:
        final_state = await rag_pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error for query='%s…': %s", request.query[:60], exc)
        raise HTTPException(status_code=500, detail="Pipeline execution failed.") from exc

    citations = [
        CitationOut(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            quote=c["quote"],
            doc_title=c["doc_title"],
            page=c.get("page"),
            section=c.get("section"),
        )
        for c in (final_state.get("citations") or [])
    ]

    return QueryResponse(
        answer=final_state.get("answer", ""),
        citations=citations,
        image_url=final_state.get("image_url"),
        query=request.query,
    )
