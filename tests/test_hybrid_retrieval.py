"""Hybrid retrieve_node: no live Gemini or Weaviate (mocked)."""

from __future__ import annotations

import pytest

from src.pipeline.state import RAGState
from src.retrieval.hybrid import retrieve_node


@pytest.mark.asyncio
async def test_hybrid_retrieval_basic(monkeypatch):
    """Sanity check: retrieve_node passes query through embed + Weaviate and returns chunks."""
    fake_chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "Sample",
            "text": "Patient presents with acute appendicitis.",
            "modality_type": "text",
            "page": 1,
            "section": "",
            "caption": "",
            "score": 0.95,
            "storage_ref": None,
        }
    ]

    async def fake_embed(_q: str) -> list[float]:
        return [0.0] * 8

    def fake_query_weaviate(*_args, **_kwargs):
        return fake_chunks

    monkeypatch.setattr("src.retrieval.hybrid.gemini_embed.embed_query", fake_embed)
    monkeypatch.setattr("src.retrieval.hybrid._query_weaviate_sync", fake_query_weaviate)

    state: RAGState = {"query_text": "appendicitis"}
    result = await retrieve_node(state)
    chunks = result.get("retrieved_chunks", [])

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    required_keys = {"chunk_id", "doc_id", "doc_title", "text", "modality_type", "score"}
    chunk = chunks[0]
    assert required_keys.issubset(chunk.keys())
    assert isinstance(chunk["score"], float)
