"""
Unit tests: src/retrieval/rerank.py  (Fireworks Rerank API)
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.citation_gate import _ABSTAIN_ANSWER, citation_gate_node
from src.pipeline.state import Citation
from src.retrieval.rerank import rerank_node


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chunk(
    chunk_id: str,
    text: str,
    score: float,
    doc_id: str = "d1",
    doc_title: str = "T1",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "text": text,
        "modality_type": "text",
        "page": None,
        "section": "",
        "caption": "",
        "score": score,
        "storage_ref": None,
    }


def _fake_settings(*, api_key: str = "fw_test_key", top_k: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        fireworks_api_key=api_key,
        fireworks_rerank_model="accounts/fireworks/models/qwen3-reranker-0p6b",
        fireworks_base_url="https://api.fireworks.ai/inference/v1",
        rerank_top_k=top_k,
    )


# ---------------------------------------------------------------------------
# fallback — no api key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rerank_node_no_api_key_falls_back_to_top_k(monkeypatch):
    """Without a Fireworks API key rerank falls back to top-K truncation."""
    monkeypatch.setattr(
        "src.retrieval.rerank.get_settings",
        lambda: _fake_settings(api_key=""),
    )
    chunks = [_chunk("1", "t1", 0.9), _chunk("2", "t2", 0.8), _chunk("3", "t3", 0.7)]
    result = await rerank_node({"query_text": "test", "retrieved_chunks": chunks})
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["chunk_id"] == "1"


@pytest.mark.asyncio
async def test_rerank_node_empty_chunks(monkeypatch):
    """rerank_node returns empty list when there are no chunks."""
    monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: _fake_settings())
    result = await rerank_node({"query_text": "test", "retrieved_chunks": []})
    assert result["retrieved_chunks"] == []


# ---------------------------------------------------------------------------
# success path — mocked Fireworks HTTP call
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rerank_node_fireworks_success(monkeypatch):
    """rerank_node correctly reorders chunks using the Fireworks API response."""
    monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: _fake_settings())

    # Fireworks returns idx 1 (score 0.95) then idx 0 (score 0.5)  — reverse of input order
    fw_response = {
        "data": [
            {"index": 1, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.50},
        ]
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = fw_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    chunks = [_chunk("low", "low text", 0.5), _chunk("high", "high text", 0.9)]
    state = {"query_text": "medical query", "retrieved_chunks": chunks}

    with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_client):
        result = await rerank_node(state)

    reranked = result["retrieved_chunks"]
    assert len(reranked) == 2
    # chunk originally at idx=1 ("high") should now be first
    assert reranked[0]["chunk_id"] == "high"
    assert reranked[0]["score"] == pytest.approx(0.95)
    # chunk originally at idx=0 ("low") should now be second
    assert reranked[1]["chunk_id"] == "low"
    assert reranked[1]["score"] == pytest.approx(0.50)


@pytest.mark.asyncio
async def test_rerank_node_payload_structure(monkeypatch):
    """Verify rerank_node posts the correct payload to Fireworks."""
    settings = _fake_settings()
    monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: settings)

    fw_response = {"data": [{"index": 0, "relevance_score": 0.9}]}
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = fw_response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    chunks = [_chunk("c1", "some text", 0.9)]
    with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_client):
        await rerank_node({"query_text": "heart disease", "retrieved_chunks": chunks})

    call_args = mock_client.post.call_args
    posted_url = call_args[0][0]
    posted_payload = call_args[1]["json"]
    posted_headers = call_args[1]["headers"]

    assert "/rerank" in posted_url
    assert posted_payload["query"] == "heart disease"
    assert posted_payload["documents"] == ["some text"]
    assert posted_payload["model"] == settings.fireworks_rerank_model
    assert posted_payload["top_n"] == 2
    assert "Bearer fw_test_key" in posted_headers["Authorization"]


@pytest.mark.asyncio
async def test_rerank_node_http_error_falls_back(monkeypatch):
    """If Fireworks returns HTTP error, rerank gracefully falls back to top-K."""
    monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: _fake_settings())

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 429")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    chunks = [_chunk("1", "t1", 0.9), _chunk("2", "t2", 0.8), _chunk("3", "t3", 0.7)]
    with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_client):
        result = await rerank_node({"query_text": "test", "retrieved_chunks": chunks})

    # falls back to top-k=2 with original order
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["chunk_id"] == "1"


# ---------------------------------------------------------------------------
# citation gate  (step 5 — still colocated here)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_citation_gate_node_empty_abstains():
    """Citation gate triggers abstain when no citations are present."""
    state = {"answer": "Some random answer.", "citations": []}
    result = await citation_gate_node(state)
    assert result["answer"] == _ABSTAIN_ANSWER
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_citation_gate_node_valid_passes_through():
    """Citation gate passes state through unchanged when citations exist."""
    state = {
        "answer": "Hypertension is managed by...",
        "citations": [
            Citation(chunk_id="1", doc_id="d1", quote="test", doc_title="T1", page=None, section=None)
        ],
    }
    result = await citation_gate_node(state)
    assert result == {}
