from __future__ import annotations

import logging

from src.pipeline.state import RAGState
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# Hybrid search scores are BM25 + vector fusion values (Weaviate reciprocal-rank
# fusion), NOT cosine similarity. Values typically range 0.0–1.0 but their
# meaning is corpus-dependent. Tune LOW_CONFIDENCE_THRESHOLD via config.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.3


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

    Returns partial state update: {"image_url": ..., "pdf_url": ..., "confidence_score": ..., "low_confidence": ...}
    """
    image_url: str | None = None
    pdf_url: str | None = None
    confidence_score: float | None = None
    low_confidence: bool | None = None

    chunks = state.get("retrieved_chunks") or []

    # 1. Confidence Thresholds: calculate overall certainty from available score metadata.
    #    NOTE: Weaviate hybrid scores are reciprocal-rank fusion values, not cosine
    #    similarity. Threshold is configurable via CONFIDENCE_THRESHOLD in settings.
    if chunks:
        scores = [c.get("score") for c in chunks if c.get("score") is not None]
        if scores:
            confidence_score = sum(scores) / len(scores)
            threshold = getattr(get_settings(), "confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD)
            logger.info(
                "output_route_node: confidence_score=%.4f threshold=%.4f",
                confidence_score, threshold,
            )
            if confidence_score < threshold:
                low_confidence = True
                logger.warning(
                    "output_route_node: low confidence (%.4f < %.4f) — flagging response",
                    confidence_score, threshold,
                )

    score_meets_threshold = (confidence_score is not None and not low_confidence)

    # 2. Image branch: activate if an image is found in retrieved chunks with decent score or requested by LLM
    target_image_chunk = None
    
    if chunks:
        # Scan retrieved chunks for a high-confidence image
        for c in chunks:
             if c.get("modality_type") == "image" and c.get("storage_ref") and score_meets_threshold:
                 target_image_chunk = c
                 break

    # If LLM forced preference, fallback to top chunk if it is an image
    if not target_image_chunk and state.get("prefer_retrieved_image") and chunks:
        if chunks[0].get("modality_type") == "image" and chunks[0].get("storage_ref"):
             target_image_chunk = chunks[0]

    if target_image_chunk:
        from src.services import storage
        try:
            image_url = storage.generate_signed_url(target_image_chunk["storage_ref"])
            logger.info("output_route_node: image branch → signed URL generated")
        except Exception as e:
            logger.error("output_route_node: image branch URL generation failed: %s", e)

    # 3. PDF branch: activate when include_pdf_export is set by LLM
    if state.get("include_pdf_export") and chunks:
        top = chunks[0]
        if top.get("storage_ref"):
            # Mock or return retrieval PDF URL; in step 7 this resolves properly.
            pdf_url = top["storage_ref"] if top.get("modality_type") == "pdf" else "EXPORT_PENDING"
            logger.info("output_route_node: pdf branch → %s", pdf_url)

    return {
        "image_url": image_url,
        "pdf_url": pdf_url,
        "confidence_score": confidence_score,
        "low_confidence": low_confidence,
    }
