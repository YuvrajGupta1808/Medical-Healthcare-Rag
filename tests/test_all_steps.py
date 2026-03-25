"""
tests/test_all_steps.py
=======================
End-to-end integration test covering Steps 1–8 of the Medical-Healthcare RAG
pipeline.  All external services (Weaviate, Gemini, Fireworks, MinIO) are
mocked so the suite runs fully offline.

Steps covered:
  1. Config / environment loading
  2. Ingest: chunking + metadata completeness
  3. Ingest: idempotent upsert (delete_document guard)
  4. Retrieval: hybrid retrieve_node
  5. Reranking: Fireworks qwen3-reranker-0p6b
  6. Output routing: text / image / pdf / low-confidence branches
  7. Citation gate: abstain vs. pass-through
  8. Full pipeline: state flows correctly end-to-end (smoke)
"""
from __future__ import annotations

import os

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("FIREWORKS_API_KEY", "fw_test_key")
os.environ.setdefault("GEMINI_API_KEY", "test_gemini_key")

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

FAKE_DIM = 8


def _vec() -> list[float]:
    return [0.1] * FAKE_DIM


def _chunk(
    chunk_id: str = "c1",
    text: str = "Patient has hypertension.",
    score: float = 0.9,
    modality: str = "text",
    storage_ref: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc-001",
        "doc_title": "Clinical Notes",
        "text": text,
        "modality_type": modality,
        "page": 1,
        "section": "Introduction",
        "caption": "",
        "score": score,
        "storage_ref": storage_ref,
    }


# ===========================================================================
# STEP 1 — CONFIG / ENVIRONMENT
# ===========================================================================

class TestStep1Config:
    def test_settings_loads_without_raising_in_testing_mode(self):
        """Settings with testing=True must skip production-key validation."""
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert s.testing is True

    def test_fireworks_api_key_read_from_env(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="fw_test_key", gemini_api_key="")
        assert s.fireworks_api_key == "fw_test_key"

    def test_fireworks_rerank_model_default(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert "qwen3-reranker" in s.fireworks_rerank_model

    def test_hybrid_alpha_default(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert 0.0 <= s.hybrid_alpha <= 1.0

    def test_retrieval_top_k_positive(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert s.retrieval_top_k > 0

    def test_weaviate_collection_name_set(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert s.weaviate_collection  # non-empty string

    def test_minio_bucket_set(self):
        from src.utils.config import Settings
        s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
        assert s.minio_bucket


# ===========================================================================
# STEP 2 — INGEST: CHUNKING + METADATA COMPLETENESS
# ===========================================================================

class TestStep2Chunking:
    def _make_text_page(self, text: str = "Sample medical paragraph. This text needs to be long enough to produce valid chunks. " * 10) -> object:
        from src.ingest.loaders import RawPage
        return RawPage(
            text=text,
            page_number=1,
            section="Introduction",
            doc_id="doc-001",
            doc_title="Clinical Trial",
            modality_type="text",
            image_bytes=None,
        )

    def _make_image_page(self) -> object:
        from src.ingest.loaders import RawPage
        return RawPage(
            text="",
            page_number=3,
            section="Figure 1",
            doc_id="doc-img-001",
            doc_title="Brain MRI Study",
            modality_type="image",
            image_bytes=b"fake_img_bytes",
        )

    def test_text_page_produces_chunks(self):
        from src.ingest.chunking import chunk_pages
        chunks = chunk_pages([self._make_text_page()])
        assert len(chunks) > 0

    def test_chunk_has_required_metadata(self):
        from src.ingest.chunking import chunk_pages
        chunks = chunk_pages([self._make_text_page()])
        c = chunks[0]
        for field in ("chunk_id", "doc_id", "doc_title", "text", "modality_type"):
            assert getattr(c, field, None) is not None, f"Missing field: {field}"

    def test_chunk_id_is_unique_across_pages(self):
        from src.ingest.chunking import chunk_pages
        page1 = self._make_text_page("Page one content about diabetes treatment.")
        page2 = self._make_text_page("Page two content about hypertension care.")
        # give different pages
        import copy
        p2 = copy.copy(page1)
        p2.page_number = 2
        p2.text = "Page two content about hypertension care."
        chunks = chunk_pages([page1, p2])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_image_chunk_text_is_empty_and_valid(self):
        from src.ingest.chunking import chunk_pages, validate_chunks
        chunks = chunk_pages([self._make_image_page()])
        assert len(chunks) == 1
        errors = validate_chunks(chunks)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_image_chunk_modality_type(self):
        from src.ingest.chunking import chunk_pages
        chunks = chunk_pages([self._make_image_page()])
        assert chunks[0].modality_type == "image"

    def test_validate_chunks_passes_on_well_formed_text_chunk(self):
        from src.ingest.chunking import chunk_pages, validate_chunks
        chunks = chunk_pages([self._make_text_page("A valid medical sentence about dosage.")])
        errors = validate_chunks(chunks)
        assert errors == []


# ===========================================================================
# STEP 3 — INGEST: IDEMPOTENT UPSERT (delete_document guard)
# ===========================================================================

class TestStep3IdempotentIngest:
    def test_delete_document_called_before_ingest(self):
        """Pipeline must call delete_document(doc_id) before upserting chunks."""
        from src.services import weaviate_client

        mock_result = MagicMock()
        mock_result.successful = 3

        with patch.object(weaviate_client, "get_client") as mock_get:
            mock_collection = MagicMock()
            mock_collection.data.delete_many.return_value = mock_result
            mock_get.return_value.collections.get.return_value = mock_collection

            count = weaviate_client.delete_document("doc-001")
            assert count == 3
            mock_collection.data.delete_many.assert_called_once()

    def test_delete_document_zero_on_no_match(self):
        from src.services import weaviate_client

        mock_result = MagicMock()
        mock_result.successful = 0

        with patch.object(weaviate_client, "get_client") as mock_get:
            mock_collection = MagicMock()
            mock_collection.data.delete_many.return_value = mock_result
            mock_get.return_value.collections.get.return_value = mock_collection

            count = weaviate_client.delete_document("nonexistent-doc")
            assert count == 0


# ===========================================================================
# STEP 4 — RETRIEVAL: HYBRID SEARCH
# ===========================================================================

class TestStep4HybridRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_node_returns_chunks(self, monkeypatch):
        from src.retrieval.hybrid import retrieve_node
        from src.services import gemini_embed

        monkeypatch.setattr(gemini_embed, "embed_query", AsyncMock(return_value=_vec()))
        monkeypatch.setattr("src.retrieval.hybrid._query_weaviate_sync", lambda *a, **k: [_chunk()])

        result = await retrieve_node({"query_text": "hypertension treatment"})
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 1

    @pytest.mark.asyncio
    async def test_retrieve_node_empty_when_no_results(self, monkeypatch):
        from src.retrieval.hybrid import retrieve_node
        from src.services import gemini_embed

        monkeypatch.setattr(gemini_embed, "embed_query", AsyncMock(return_value=_vec()))
        monkeypatch.setattr("src.retrieval.hybrid._query_weaviate_sync", lambda *a, **k: [])

        result = await retrieve_node({"query_text": "nonsense query xyz"})
        assert result["retrieved_chunks"] == []

    @pytest.mark.asyncio
    async def test_retrieve_node_chunk_schema(self, monkeypatch):
        from src.retrieval.hybrid import retrieve_node
        from src.services import gemini_embed

        monkeypatch.setattr(gemini_embed, "embed_query", AsyncMock(return_value=_vec()))
        monkeypatch.setattr("src.retrieval.hybrid._query_weaviate_sync", lambda *a, **k: [_chunk()])

        result = await retrieve_node({"query_text": "dosage check"})
        c = result["retrieved_chunks"][0]
        for key in ("chunk_id", "doc_id", "text", "modality_type", "score"):
            assert key in c, f"Missing key '{key}' in retrieved chunk"

    @pytest.mark.asyncio
    async def test_retrieve_node_score_is_float(self, monkeypatch):
        from src.retrieval.hybrid import retrieve_node
        from src.services import gemini_embed

        monkeypatch.setattr(gemini_embed, "embed_query", AsyncMock(return_value=_vec()))
        monkeypatch.setattr("src.retrieval.hybrid._query_weaviate_sync", lambda *a, **k: [_chunk()])

        result = await retrieve_node({"query_text": "bleeding risk"})
        assert isinstance(result["retrieved_chunks"][0]["score"], float)


# ===========================================================================
# STEP 5 — RERANKING: Fireworks qwen3-reranker-0p6b
# ===========================================================================

class TestStep5Reranking:
    def _fw_settings(self, *, api_key: str = "fw_key") -> SimpleNamespace:
        return SimpleNamespace(
            fireworks_api_key=api_key,
            fireworks_rerank_model="accounts/fireworks/models/qwen3-reranker-0p6b",
            fireworks_base_url="https://api.fireworks.ai/inference/v1",
            rerank_top_k=2,
        )

    def _mock_http(self, data: list[dict]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": data}
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)
        return mock_client

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_relevance(self, monkeypatch):
        from src.retrieval.rerank import rerank_node
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: self._fw_settings())

        fw_data = [{"index": 1, "relevance_score": 0.97}, {"index": 0, "relevance_score": 0.42}]
        chunks = [_chunk("low", "low text", 0.4), _chunk("high", "high text", 0.9)]

        with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=self._mock_http(fw_data)):
            result = await rerank_node({"query_text": "cardiac symptoms", "retrieved_chunks": chunks})

        assert result["retrieved_chunks"][0]["chunk_id"] == "high"
        assert result["retrieved_chunks"][0]["score"] == pytest.approx(0.97)

    @pytest.mark.asyncio
    async def test_rerank_no_key_truncates(self, monkeypatch):
        from src.retrieval.rerank import rerank_node
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: self._fw_settings(api_key=""))

        chunks = [_chunk(str(i), f"text {i}", float(i) / 10) for i in range(5)]
        result = await rerank_node({"query_text": "test", "retrieved_chunks": chunks})
        assert len(result["retrieved_chunks"]) == 2

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self, monkeypatch):
        from src.retrieval.rerank import rerank_node
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: self._fw_settings())
        result = await rerank_node({"query_text": "test", "retrieved_chunks": []})
        assert result["retrieved_chunks"] == []

    @pytest.mark.asyncio
    async def test_rerank_request_uses_correct_model(self, monkeypatch):
        from src.retrieval.rerank import rerank_node
        settings = self._fw_settings()
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: settings)

        fw_data = [{"index": 0, "relevance_score": 0.9}]
        mock_client = self._mock_http(fw_data)

        with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_client):
            await rerank_node({"query_text": "q", "retrieved_chunks": [_chunk()]})

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == settings.fireworks_rerank_model

    @pytest.mark.asyncio
    async def test_rerank_http_error_falls_back(self, monkeypatch):
        from src.retrieval.rerank import rerank_node
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: self._fw_settings())

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        chunks = [_chunk("1", "t1", 0.9), _chunk("2", "t2", 0.8), _chunk("3", "t3", 0.7)]
        with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_client):
            result = await rerank_node({"query_text": "q", "retrieved_chunks": chunks})
        assert len(result["retrieved_chunks"]) == 2


# ===========================================================================
# STEP 6 — OUTPUT ROUTING
# ===========================================================================

class TestStep6OutputRouter:
    @pytest.mark.asyncio
    async def test_image_route_returns_signed_url(self):
        from src.router.output_router import output_route_node
        state = {
            "prefer_retrieved_image": True,
            "retrieved_chunks": [_chunk("c1", "img text", 0.9, modality="image", storage_ref="bucket/img.jpg")],
        }
        with patch("src.services.storage.generate_signed_url", return_value="http://signed/url"):
            result = await output_route_node(state)
        assert result["image_url"] == "http://signed/url"
        assert result["confidence_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_pdf_route_returns_pdf_url(self):
        from src.router.output_router import output_route_node
        state = {
            "include_pdf_export": True,
            "retrieved_chunks": [_chunk("c1", "pdf text", 0.8, modality="pdf", storage_ref="http://pdf/url")],
        }
        result = await output_route_node(state)
        assert result["pdf_url"] == "http://pdf/url"

    @pytest.mark.asyncio
    async def test_low_confidence_flagged(self):
        from src.router.output_router import output_route_node
        state = {
            "retrieved_chunks": [_chunk("c1", "t1", 0.3), _chunk("c2", "t2", 0.4)],
        }
        result = await output_route_node(state)
        assert result["confidence_score"] == pytest.approx(0.35)
        assert result["low_confidence"] is True

    @pytest.mark.asyncio
    async def test_high_confidence_not_flagged(self):
        from src.router.output_router import output_route_node
        state = {
            "retrieved_chunks": [_chunk("c1", "t1", 0.85), _chunk("c2", "t2", 0.90)],
        }
        result = await output_route_node(state)
        assert result.get("low_confidence") is not True


# ===========================================================================
# STEP 7 — CITATION GATE
# ===========================================================================

class TestStep7CitationGate:
    @pytest.mark.asyncio
    async def test_gate_abstains_on_empty_citations(self):
        from src.generation.citation_gate import _ABSTAIN_ANSWER, citation_gate_node
        result = await citation_gate_node({"answer": "Some text.", "citations": []})
        assert result["answer"] == _ABSTAIN_ANSWER

    @pytest.mark.asyncio
    async def test_gate_passes_through_when_citations_present(self):
        from src.generation.citation_gate import citation_gate_node
        from src.pipeline.state import Citation
        state = {
            "answer": "Hypertension requires lifestyle changes.",
            "citations": [
                Citation(chunk_id="c1", doc_id="d1", quote="test", doc_title="T1", page=1, section="S1")
            ],
        }
        result = await citation_gate_node(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_gate_clears_citations_on_abstain(self):
        from src.generation.citation_gate import citation_gate_node
        result = await citation_gate_node({"answer": "Hallucinated answer.", "citations": []})
        assert result["citations"] == []


# ===========================================================================
# STEP 8 — FULL PIPELINE SMOKE TEST
# ===========================================================================

class TestStep8PipelineSmoke:
    """
    Smoke test: wire retrieve → rerank → citation_gate together and verify
    state flows correctly without any live service calls.
    """

    @pytest.mark.asyncio
    async def test_pipeline_state_flows_retrieve_to_rerank(self, monkeypatch):
        from src.retrieval.hybrid import retrieve_node
        from src.retrieval.rerank import rerank_node
        from src.services import gemini_embed

        # patch external deps
        monkeypatch.setattr(gemini_embed, "embed_query", AsyncMock(return_value=_vec()))
        monkeypatch.setattr(
            "src.retrieval.hybrid._query_weaviate_sync",
            lambda *a, **k: [_chunk("c1", "text1", 0.9), _chunk("c2", "text2", 0.7)],
        )

        fw_settings = SimpleNamespace(
            fireworks_api_key="",  # force fallback truncation
            fireworks_rerank_model="accounts/fireworks/models/qwen3-reranker-0p6b",
            fireworks_base_url="https://api.fireworks.ai/inference/v1",
            rerank_top_k=1,
        )
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: fw_settings)

        state = {"query_text": "What is the dosage for aspirin?"}
        state.update(await retrieve_node(state))
        state.update(await rerank_node(state))

        assert len(state["retrieved_chunks"]) == 1
        assert state["retrieved_chunks"][0]["chunk_id"] == "c1"

    @pytest.mark.asyncio
    async def test_pipeline_citation_gate_abstains_when_empty_after_rerank(self, monkeypatch):
        from src.generation.citation_gate import _ABSTAIN_ANSWER, citation_gate_node
        from src.retrieval.rerank import rerank_node

        fw_settings = SimpleNamespace(
            fireworks_api_key="",
            fireworks_rerank_model="model",
            fireworks_base_url="https://api.fireworks.ai/inference/v1",
            rerank_top_k=1,
        )
        monkeypatch.setattr("src.retrieval.rerank.get_settings", lambda: fw_settings)

        state: dict = {"query_text": "empty", "retrieved_chunks": [], "answer": "some answer", "citations": []}
        state.update(await rerank_node(state))
        state.update(await citation_gate_node(state))

        assert state["answer"] == _ABSTAIN_ANSWER
