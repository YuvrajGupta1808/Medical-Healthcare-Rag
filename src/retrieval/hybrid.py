from __future__ import annotations

import asyncio
import logging

import weaviate.classes as wvc

from src.pipeline.state import Chunk, RAGState
from src.services import gemini_embed, weaviate_client
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


async def retrieve_node(state: RAGState) -> dict:
    """
    LangGraph node: embed the query and retrieve top-k chunks via hybrid search.

    - Embeds query text using Gemini multimodal embeddings (async)
    - Queries Weaviate hybrid in thread pool (sync client)
    - Applies optional doc_id filter

    Returns partial state update: {"retrieved_chunks": [...]}
    """
    settings = get_settings()

    # Embed query asynchronously (RETRIEVAL_QUERY task type for better recall)
    embedding = await gemini_embed.embed_query(state["query_text"])

    # Run Weaviate query in thread pool (client is synchronous)
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(
        None,
        _query_weaviate_sync,
        state["query_text"],
        embedding,
        state.get("patient_id"),
        state.get("doc_id"),
        state.get("modality_type"),
        settings.rerank_top_m,
        settings.hybrid_alpha,
    )

    logger.info(
        "retrieve_node (hybrid): %d chunks for query='%s…' doc_id=%s modality=%s alpha=%f",
        len(chunks),
        state["query_text"][:60],
        state.get("doc_id"),
        state.get("modality_type"),
        settings.hybrid_alpha,
    )
    return {"retrieved_chunks": chunks}


def _query_weaviate_sync(
    query_text: str,
    embedding: list[float],
    patient_id: str | None,
    doc_id: str | None,
    modality_type: str | None,
    top_k: int,
    alpha: float,
) -> list[Chunk]:
    """Execute hybrid query against Weaviate."""
    collection = weaviate_client.get_client().collections.get(
        weaviate_client.COLLECTION_NAME
    )

    filters = []
    if patient_id:
        filters.append(wvc.query.Filter.by_property("patient_id").equal(patient_id))
    if doc_id:
        filters.append(wvc.query.Filter.by_property("doc_id").equal(doc_id))
    if modality_type and modality_type != "text":
        filters.append(wvc.query.Filter.by_property("modality_type").equal(modality_type))

    combined_filter = None
    if filters:
        combined_filter = wvc.query.Filter.all_of(filters)

    response = collection.query.hybrid(
        query=query_text,
        vector=embedding,
        alpha=alpha,
        limit=top_k,
        filters=combined_filter,
        return_properties=[
            "chunk_id", "doc_id", "doc_title", "text",
            "modality_type", "page", "section", "storage_ref", "caption",
        ],
        return_metadata=wvc.query.MetadataQuery(score=True),
    )

    chunks: list[Chunk] = []
    for obj in response.objects:
        p = obj.properties
        score = 0.0
        if obj.metadata and obj.metadata.score is not None:
            score = float(obj.metadata.score)

        chunks.append(
            Chunk(
                chunk_id=str(p.get("chunk_id", "")),
                doc_id=str(p.get("doc_id", "")),
                doc_title=str(p.get("doc_title", "")),
                text=str(p.get("text", "")),
                modality_type=str(p.get("modality_type", "text")),
                page=p.get("page"),
                section=str(p.get("section", "")),
                caption=str(p.get("caption", "")),
                score=score,
                storage_ref=p.get("storage_ref"),
            )
        )

    return chunks
