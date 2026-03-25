from __future__ import annotations

import logging

import httpx

from src.pipeline.state import Chunk, RAGState
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


async def rerank_node(state: RAGState) -> dict:
    """
    LangGraph node: Rerank retrieved chunks using Fireworks Rerank API.
    
    - Inputs: state["retrieved_chunks"] (top-M chunks)
    - If fireworks_api_key is set, calls Fireworks Rerank API
    - Otherwise, falls back to returning top-K chunks by their original retrieval score
    - Outputs: state["retrieved_chunks"] reduced to top-K
    """
    settings = get_settings()
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        logger.warning("rerank_node: No chunks to rerank")
        return {"retrieved_chunks": []}

    top_k = settings.rerank_top_k

    if not settings.fireworks_api_key:
        logger.warning("rerank_node: fireworks_api_key not set, falling back to top-k truncation")
        return {"retrieved_chunks": chunks[:top_k]}

    try:
        # For image chunks where text is empty, provide the caption to the text re-ranker
        docs = [chunk.get("text") or chunk.get("caption", "Image Snippet") for chunk in chunks]

        headers = {
            "Authorization": f"Bearer {settings.fireworks_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": state["query_text"],
            "documents": docs,
            "model": settings.fireworks_rerank_model,
            "top_n": top_k,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.fireworks_base_url}/rerank",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            response_json = response.json()

        reranked_chunks: list[Chunk] = []
        ranking_data = response_json.get("data", []) or response_json.get("results", [])
        
        for result in ranking_data:
            idx = result["index"]
            score = result["relevance_score"]
            orig_chunk = chunks[idx]
            
            # Since Chunk is a TypedDict, we can simply copy and update the score
            new_chunk = orig_chunk.copy()
            new_chunk["score"] = float(score)
            reranked_chunks.append(new_chunk)
            
        logger.info(
            "rerank_node: Reranked %d chunks to %d chunks using %s",
            len(chunks), len(reranked_chunks), settings.fireworks_rerank_model
        )
        return {"retrieved_chunks": reranked_chunks}

    except Exception as exc:
        logger.error("rerank_node: Fireworks rerank failed: %s. Falling back to original order.", exc)
        return {"retrieved_chunks": chunks[:top_k]}
