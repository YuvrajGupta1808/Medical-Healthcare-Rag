from __future__ import annotations

import logging

from src.pipeline.state import RAGState

logger = logging.getLogger(__name__)


async def output_route_node(state: RAGState) -> dict:
    """
    LangGraph node: deterministic output routing.

    Decides which response branches to populate based on:
    - modality_type of the top retrieved chunk
    - LLM-set signals: prefer_retrieved_image, include_pdf_export
    - Availability of a storage_ref on the top chunk

    This is pure rule logic — no second LLM call.

    Branches:
    - Text: always populated (answer + citations)
    - Image: populated if top chunk is an image with a storage_ref
    - PDF: reserved for Step 6+ (include_pdf_export flag)

    Returns partial state update: {"image_url": ...}
    """
    image_url: str | None = None

    # Image branch: activate when the LLM signals it AND the top chunk has a media ref
    if state.get("prefer_retrieved_image") and state["retrieved_chunks"]:
        top = state["retrieved_chunks"][0]
        if top.get("modality_type") == "image" and top.get("storage_ref"):
            image_url = top["storage_ref"]
            logger.info("output_route_node: image branch → %s", image_url)

    return {"image_url": image_url}
