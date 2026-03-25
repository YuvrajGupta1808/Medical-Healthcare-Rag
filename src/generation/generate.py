from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from src.pipeline.state import Citation, RAGState
from src.utils.config import get_settings
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_ABSTAIN_ANSWER = (
    "I cannot find sufficient evidence in the provided documents to answer this question."
)


def _build_context(chunks: list) -> str:
    """Format retrieved chunks as a numbered, labelled context block for the LLM."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta_parts = [f"CHUNK_ID={chunk['chunk_id']}", f"DOC={chunk['doc_id']}"]
        if chunk.get("page"):
            meta_parts.append(f"PAGE={chunk['page']}")
        if chunk.get("section"):
            meta_parts.append(f"SECTION={chunk['section']}")
        header = f"[{i}] " + " | ".join(meta_parts)
        
        chunk_text = chunk.get("text") or ""
        if chunk.get("modality_type") == "image":
            caption = chunk.get("caption") or f"Image on page {chunk.get('page', '?')}"
            chunk_text = f"[IMAGE SNIPPET: {caption}] - Please cite this chunk ID if the visual graph/chart is relevant to the user query."
            
        parts.append(f"{header}\n{chunk_text}")
    return "\n\n---\n\n".join(parts)


async def generate_node(state: RAGState) -> dict:
    """
    LangGraph node: generate a cited answer from retrieved chunks.

    - Uses Fireworks AI Llama 3.1 70B (OpenAI-compatible API)
    - Requires JSON output with answer + citations array
    - Validates that all cited chunk_ids are from the retrieved set
    - Returns abstain response if no chunks were retrieved

    Returns partial state update: {"answer": ..., "citations": [...]}
    """
    settings = get_settings()

    if not state["retrieved_chunks"]:
        logger.warning("generate_node: no chunks retrieved — returning abstain answer")
        return {"answer": _ABSTAIN_ANSWER, "citations": []}

    client = AsyncOpenAI(
        api_key=settings.fireworks_api_key,
        base_url=settings.fireworks_base_url,
    )

    system_prompt = load_prompt("system", version=settings.prompt_version)
    context_block = _build_context(state["retrieved_chunks"])

    user_message = (
        "Context from retrieved document chunks:\n\n"
        f"{context_block}\n\n"
        f"Question: {state['query_text']}"
    )

    try:
        response = await client.chat.completions.create(
            model=settings.fireworks_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)

    except json.JSONDecodeError as exc:
        logger.error("generate_node: failed to parse LLM JSON response: %s", exc)
        return {"answer": "Error: model returned unparseable output.", "citations": []}
    except Exception as exc:
        logger.error("generate_node: LLM call failed: %s", exc)
        raise

    answer: str = parsed.get("answer", _ABSTAIN_ANSWER)
    raw_citations: list[dict] = parsed.get("citations", [])

    # Citation gate: only keep citations whose chunk_id was actually retrieved
    chunks_map = {chunk["chunk_id"]: chunk for chunk in state["retrieved_chunks"]}
    citations: list[Citation] = []
    dropped = 0
    
    for cit in raw_citations:
        cid = cit.get("chunk_id", "")
        if cid in chunks_map:
            c_meta = chunks_map[cid]
            citations.append(
                Citation(
                    chunk_id=cid,
                    doc_id=str(cit.get("doc_id", "")),
                    quote=str(cit.get("quote", ""))[:250],  # cap quote length
                    doc_title=str(cit.get("doc_title", "")),
                    page=cit.get("page"),
                    section=cit.get("section"),
                    storage_ref=c_meta.get("storage_ref"),
                    modality_type=c_meta.get("modality_type"),
                )
            )
        else:
            dropped += 1

    if dropped:
        logger.warning(
            "generate_node: dropped %d citation(s) with unrecognised chunk_ids", dropped
        )

    logger.info(
        "generate_node: answer=%d chars, citations=%d (dropped=%d)",
        len(answer), len(citations), dropped,
    )
    return {"answer": answer, "citations": citations}
