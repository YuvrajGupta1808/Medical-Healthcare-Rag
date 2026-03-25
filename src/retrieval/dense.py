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
    LangGraph node: embed the query and retrieve top-k chunks via dense search.

    - Embeds query text using Gemini multimodal embeddings (async)
    - Queries Weaviate near_vector in thread pool (sync client)
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
        embedding,
        state.get("doc_id"),
        settings.rerank_top_m,
    )

    logger.info(
        "retrieve_node: %d chunks for query='%s…' doc_id=%s",
        len(chunks),
        state["query_text"][:60],
        state.get("doc_id"),
    )
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Synchronous Weaviate query (runs in executor)
# ---------------------------------------------------------------------------

def _query_weaviate_sync(
    embedding: list[float],
    doc_id: str | None,
    top_k: int,
) -> list[Chunk]:
    """Execute near_vector query against Weaviate."""
    collection = weaviate_client.get_client().collections.get(
        weaviate_client.COLLECTION_NAME
    )

    filters = None
    if doc_id:
        filters = wvc.query.Filter.by_property("doc_id").equal(doc_id)

    response = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k,
        filters=filters,
        return_properties=[
            "chunk_id", "doc_id", "doc_title", "text",
            "modality_type", "page", "section", "storage_ref", "caption",
        ],
        return_metadata=wvc.query.MetadataQuery(certainty=True),
    )

    chunks: list[Chunk] = []
    for obj in response.objects:
        p = obj.properties
        certainty = 0.0
        if obj.metadata and obj.metadata.certainty is not None:
            certainty = float(obj.metadata.certainty)

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
                score=certainty,
                storage_ref=p.get("storage_ref"),
            )
        )

    return chunks
