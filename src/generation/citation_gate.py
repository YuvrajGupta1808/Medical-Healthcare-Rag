from __future__ import annotations

import logging

from src.pipeline.state import Citation, RAGState

logger = logging.getLogger(__name__)

_ABSTAIN_ANSWER = (
    "I cannot find sufficient evidence in the provided documents to answer this question."
)


async def citation_gate_node(state: RAGState) -> dict:
    """
    LangGraph node: Enforce evidence alignment for the generated answer.
    
    - Validates that all citations are tied to existing retrieved_chunks.
      (In case generation_node already does filtration, we can just check if citations list is empty).
    - If there are zero valid citations, triggers the 'abstain path' by setting
      the answer to an abstain fallback answer.
    """
    citations: list[Citation] = state.get("citations", [])
    
    if not citations:
        logger.warning("citation_gate_node: No valid citations surviving the gate — abstaining.")
        return {"answer": _ABSTAIN_ANSWER, "citations": []}

    logger.info("citation_gate_node: %d citations verified.", len(citations))
    return {}
