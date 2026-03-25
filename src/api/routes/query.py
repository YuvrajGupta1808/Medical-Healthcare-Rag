from __future__ import annotations

import logging

import json
import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.pipeline.graph import get_rag_pipeline
from src.router.input_router import InputRouter
from src.schemas.query import Modality, TextQueryRequest
from src.schemas.response import CitationOut, QueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _map_state_to_query_response(final_state: dict, query_text: str) -> QueryResponse:
    from src.services import storage
    
    citations = []
    for c in (final_state.get("citations") or []):
        cit_img_url = None
        if c.get("modality_type") == "image" and c.get("storage_ref"):
            try:
                cit_img_url = storage.generate_signed_url(c["storage_ref"])
            except Exception:
                pass
                
        citations.append(
            CitationOut(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                quote=c["quote"],
                doc_title=c["doc_title"],
                page=c.get("page"),
                section=c.get("section"),
                image_url=cit_img_url,
            )
        )

    return QueryResponse(
        answer=final_state.get("answer", ""),
        citations=citations,
        image_url=final_state.get("image_url"),
        pdf_url=final_state.get("pdf_url"),
        confidence_score=final_state.get("confidence_score"),
        low_confidence=final_state.get("low_confidence"),
        query=query_text,
    )


@router.post("", response_model=QueryResponse)
async def query(request: TextQueryRequest) -> QueryResponse:
    """
    Query the RAG pipeline with a text question.
    Runs the full graph: retrieve → generate → output_route.
    Returns a grounded answer with source citations.
    """
    query_text = request.query
    
    # Enrich query with patient name for better semantic alignment
    if request.patient_id:
        from src.api.routes.patients import load_patients_db
        patients = load_patients_db()
        patient = next((p for p in patients if p["id"] == request.patient_id), None)
        if patient:
            query_text = f"Patient: {patient['name']}. Question: {request.query}"

    try:
        initial_state = await InputRouter().route(
            modality=Modality.TEXT,
            question=query_text,
        )
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        initial_state["metadata"]["top_k"] = request.top_k
        if request.doc_id:
            initial_state["doc_id"] = request.doc_id
        if request.patient_id:
            initial_state["patient_id"] = request.patient_id

        pipeline = get_rag_pipeline()
        final_state = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error for query='%s…': %s", request.query[:60], exc)
        raise HTTPException(status_code=500, detail="Pipeline execution failed.") from exc

    return _map_state_to_query_response(final_state, request.query)

@router.post("/stream")
async def query_stream(request: TextQueryRequest):
    """
    Stream the RAG pipeline execution via Server-Sent Events (SSE).
    Yields intermediate trace steps and the final grounded response.
    """
    async def event_generator():
        try:
            query_text = request.query
            
            # Enrich query with patient name for better semantic alignment
            if request.patient_id:
                from src.api.routes.patients import load_patients_db
                patients = load_patients_db()
                patient = next((p for p in patients if p["id"] == request.patient_id), None)
                if patient:
                    query_text = f"Patient: {patient['name']}. Question: {request.query}"

            initial_state = await InputRouter().route(
                modality=Modality.TEXT,
                question=query_text,
            )
            if "metadata" not in initial_state:
                initial_state["metadata"] = {}
            initial_state["metadata"]["top_k"] = request.top_k
            if request.doc_id:
                initial_state["doc_id"] = request.doc_id
            if request.patient_id:
                initial_state["patient_id"] = request.patient_id

            pipeline = get_rag_pipeline()
            
            node_messages = {
                "input_router": {"step": "Routing", "status": "Analyzing intent and targeting vectors"},
                "retrieve": {"step": "Retrieving", "status": "Searching Weaviate for contextual chunks..."},
                "rerank": {"step": "Retrieving", "status": "Re-ranking extracted semantic chunks"},
                "generate": {"step": "Generation", "status": "Synthesizing grounded response using AI..."},
                "citation_gate": {"step": "Generation", "status": "Verifying grounded citations..."},
                "output_route": {"step": "Generation", "status": "Finalizing response"}
            }

            final_state = initial_state
            async for step_data in pipeline.astream(initial_state):
                if not step_data:
                    continue
                    
                for node_name, node_state in step_data.items():
                    # Handle end of graph signals or null node outputs safely
                    if node_state and isinstance(node_state, dict):
                         final_state = {**final_state, **node_state} # Merge state
                         
                    if node_name in node_messages:
                        msg = node_messages[node_name]
                        yield f"data: {json.dumps({'type': 'trace', 'trace': msg})}\n\n"
                        await asyncio.sleep(0.1)
            
            # Flush final generation
            resp = _map_state_to_query_response(final_state, request.query)
            yield f"data: {json.dumps({'type': 'result', 'data': resp.model_dump()})}\n\n"
                
        except Exception as exc:
            logger.error("Pipeline streaming error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/multimodal", response_model=QueryResponse)
async def query_multimodal(
    file: UploadFile | None = File(None),
    modality: Modality = Form(...),
    query: str = Form(""),
    doc_id: str | None = Form(None),
    patient_id: str | None = Form(None),
    top_k: int = Form(5),
) -> QueryResponse:
    """
    Query the RAG pipeline with multimodal input (Image, Audio, Video, PDF).
    Accepts files and parameters via multipart/form-data.
    """
    try:
        initial_state = await InputRouter().route(
            modality=modality,
            question=query,
            file=file,
        )
        # Propagate configuration hints to down-stream logic
        if "metadata" not in initial_state:
            initial_state["metadata"] = {}
        initial_state["metadata"]["top_k"] = top_k
        if doc_id:
            initial_state["doc_id"] = doc_id
        if patient_id:
            initial_state["patient_id"] = patient_id

        pipeline = get_rag_pipeline()
        final_state = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error for multimodal query: %s", exc)
        raise HTTPException(status_code=500, detail="Pipeline execution failed.") from exc

    return _map_state_to_query_response(final_state, final_state.get("query_text") or query)
