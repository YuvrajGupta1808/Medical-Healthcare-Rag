#!/usr/bin/env python3
"""
tests/run_all_steps_manual.py
============================
Self-contained test runner for Steps 1–8 of the Medical-Healthcare RAG
pipeline. Requires NO pytest, NO external service connections.
All Weaviate / Gemini / Fireworks calls are stubbed with stdlib mocks.

Usage:
    PYTHONPATH=. python3 tests/run_all_steps_manual.py

Expected output: each step prints PASS / FAIL with a reason.
"""
from __future__ import annotations

import asyncio
import copy
import os
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# ── env must be set BEFORE any src import ──────────────────────────────────
os.environ["TESTING"] = "true"
os.environ.setdefault("FIREWORKS_API_KEY", "fw_test_key")
os.environ.setdefault("GEMINI_API_KEY", "test_gemini_key")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
BLUE  = "\033[94m"
RESET = "\033[0m"

passed = failed = 0


def ok(name: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}✓ PASS{RESET}  {name}")


def fail(name: str, reason: str) -> None:
    global failed
    failed += 1
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    print(f"          → {reason}")


def section(title: str) -> None:
    print(f"\n{BLUE}{'─'*60}{RESET}")
    print(f"{BLUE}  {title}{RESET}")
    print(f"{BLUE}{'─'*60}{RESET}")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── shared helpers ──────────────────────────────────────────────────────────
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
        "section": "Intro",
        "caption": "",
        "score": score,
        "storage_ref": storage_ref,
    }


def _fw_settings(*, api_key: str = "fw_key") -> SimpleNamespace:
    return SimpleNamespace(
        fireworks_api_key=api_key,
        fireworks_rerank_model="accounts/fireworks/models/qwen3-reranker-0p6b",
        fireworks_base_url="https://api.fireworks.ai/inference/v1",
        rerank_top_k=2,
    )
def _base_settings() -> SimpleNamespace:
    """Minimal settings that avoids any .env file read."""
    return SimpleNamespace(
        hybrid_alpha=0.5,
        rerank_top_m=20,
        rerank_top_k=5,
        confidence_threshold=0.6,
        fireworks_api_key="fw_test_key",
        fireworks_rerank_model="accounts/fireworks/models/qwen3-reranker-0p6b",
        fireworks_base_url="https://api.fireworks.ai/inference/v1",
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — CONFIG / ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 1 — Config / Environment")

try:
    from pydantic_settings import SettingsConfigDict
    from src.utils.config import Settings

    class _SettingsNoFile(Settings):
        """Settings subclass that reads from env vars only — no .env file."""
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    s = _SettingsNoFile(testing=True, fireworks_api_key="fw_key", gemini_api_key="")

    if s.testing is True:
        ok("Settings loads in testing mode without production validation")
    else:
        fail("testing flag", "Expected testing=True")

    if s.fireworks_api_key == "fw_key":
        ok("fireworks_api_key reads from constructor")
    else:
        fail("fireworks_api_key", f"Got '{s.fireworks_api_key}'")

    if "qwen3-reranker" in s.fireworks_rerank_model:
        ok(f"fireworks_rerank_model set to '{s.fireworks_rerank_model}'")
    else:
        fail("fireworks_rerank_model", f"Got '{s.fireworks_rerank_model}'")

    if 0.0 <= s.hybrid_alpha <= 1.0:
        ok(f"hybrid_alpha in valid range: {s.hybrid_alpha}")
    else:
        fail("hybrid_alpha", f"Got {s.hybrid_alpha}")

    if s.retrieval_top_k > 0:
        ok(f"retrieval_top_k is positive: {s.retrieval_top_k}")
    else:
        fail("retrieval_top_k", f"Got {s.retrieval_top_k}")

    if s.weaviate_collection:
        ok(f"weaviate_collection = '{s.weaviate_collection}'")
    else:
        fail("weaviate_collection", "Empty string")

except Exception as e:
    fail("Step 1 import", traceback.format_exc(limit=3))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — CHUNKING + METADATA COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 2 — Ingest: Chunking + Metadata")

try:
    from src.ingest.chunking import chunk_pages, validate_chunks
    from src.ingest.loaders import RawPage

    def _page(text: str = ("Clinical guideline for the management of hypertension. "
                           "Patients with stage 2 hypertension should be treated "
                           "with combination antihypertensive therapy. " * 4),
              page: int = 1,
               modality: str = "text", image_bytes: bytes | None = None) -> RawPage:
        return RawPage(
            text=text, page_number=page, section="Intro",
            doc_id="doc-001", doc_title="Clinical Trial",
            modality_type=modality, image_bytes=image_bytes,
        )

    LONG_TEXT = ("Clinical guideline for the management of hypertension. "
                 "Patients at risk should receive regular monitoring. " * 6)
    chunks = chunk_pages([_page(LONG_TEXT)])
    if chunks:
        ok(f"Text page produces {len(chunks)} chunk(s)")
    else:
        fail("text chunking", "No chunks produced")

    c = chunks[0] if chunks else None
    if c:
        missing = [f for f in ("chunk_id", "doc_id", "doc_title", "text", "modality_type") if not getattr(c, f, None)]
        if not missing:
            ok("Chunk has all required metadata fields")
        else:
            fail("chunk metadata", f"Missing: {missing}")

    # uniqueness across pages
    LONG1 = "Page one content about diabetes treatment protocols. " * 6
    LONG2 = "Page two content about hypertension care guidelines. " * 6
    p1 = _page(LONG1, page=1)
    p2 = _page(LONG2, page=2)
    all_chunks = chunk_pages([p1, p2])
    ids = [ch.chunk_id for ch in all_chunks]
    if len(ids) == len(set(ids)):
        ok("chunk_ids are unique across pages")
    else:
        fail("chunk uniqueness", f"Duplicates found: {ids}")

    # image chunk
    img_page = _page("", page=3, modality="image", image_bytes=b"fake_img")
    img_chunks = chunk_pages([img_page])
    if img_chunks:
        ok(f"Image page produces {len(img_chunks)} chunk(s)")
        errors = validate_chunks(img_chunks)
        if errors == []:
            ok("Image chunk passes validate_chunks (empty text is allowed)")
        else:
            fail("image chunk validation", str(errors))
        if img_chunks[0].modality_type == "image":
            ok("Image chunk modality_type = 'image'")
        else:
            fail("image modality_type", f"Got '{img_chunks[0].modality_type}'")
    else:
        fail("image chunking", "No chunks produced for image page")

except Exception:
    fail("Step 2", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — IDEMPOTENT INGEST (delete_document guard)
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 3 — Ingest: Idempotent Upsert")

try:
    from src.services import weaviate_client

    # Test delete_document returns correct count
    mock_result = MagicMock()
    mock_result.successful = 3
    with patch.object(weaviate_client, "get_client") as mock_get:
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = mock_result
        mock_get.return_value.collections.get.return_value = mock_col

        count = weaviate_client.delete_document("doc-001")
        if count == 3:
            ok("delete_document returns correct successful count")
        else:
            fail("delete_document count", f"Got {count}")

        if mock_col.data.delete_many.called:
            ok("delete_many was called during delete_document")
        else:
            fail("delete_many call", "delete_many was not invoked")

    # Test 0 count when no match
    mock_result.successful = 0
    with patch.object(weaviate_client, "get_client") as mock_get:
        mock_col = MagicMock()
        mock_col.data.delete_many.return_value = mock_result
        mock_get.return_value.collections.get.return_value = mock_col

        count = weaviate_client.delete_document("nonexistent")
        if count == 0:
            ok("delete_document returns 0 when no chunks matched")
        else:
            fail("delete_document 0 count", f"Got {count}")

except Exception:
    fail("Step 3", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — HYBRID RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 4 — Hybrid Retrieval (Weaviate)")

try:
    from src.retrieval.hybrid import retrieve_node
    from src.services import gemini_embed as _gem

    async def _step4():
        with patch("src.retrieval.hybrid.get_settings", return_value=_base_settings()):
         with patch.object(_gem, "embed_query", new=AsyncMock(return_value=_vec())):
            with patch("src.retrieval.hybrid._query_weaviate_sync", return_value=[_chunk()]):
                result = await retrieve_node({"query_text": "hypertension treatment"})
                chunks = result.get("retrieved_chunks", [])

                if isinstance(chunks, list) and len(chunks) == 1:
                    ok("retrieve_node returns list with 1 chunk")
                else:
                    fail("retrieve_node result", f"Got {chunks}")

                required = {"chunk_id", "doc_id", "text", "modality_type", "score"}
                c = chunks[0] if chunks else {}
                missing = required - set(c.keys())
                if not missing:
                    ok("Retrieved chunk has all required schema keys")
                else:
                    fail("chunk schema", f"Missing keys: {missing}")

                if isinstance(c.get("score"), float):
                    ok(f"Chunk score is float: {c['score']}")
                else:
                    fail("score type", f"Got {type(c.get('score'))}")

         # empty results
         with patch("src.retrieval.hybrid.get_settings", return_value=_base_settings()):
          with patch.object(_gem, "embed_query", new=AsyncMock(return_value=_vec())):
            with patch("src.retrieval.hybrid._query_weaviate_sync", return_value=[]):
                result = await retrieve_node({"query_text": "nonsense xyz"})
                if result["retrieved_chunks"] == []:
                    ok("retrieve_node returns empty list when no results")
                else:
                    fail("empty result", f"Got {result}")

    run(_step4())

except Exception:
    fail("Step 4", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — FIREWORKS RERANKING
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 5 — Reranking: Fireworks qwen3-reranker-0p6b")

try:
    from src.retrieval.rerank import rerank_node

    def _mock_http(data: list[dict]) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": data}
        cli = AsyncMock()
        cli.__aenter__ = AsyncMock(return_value=cli)
        cli.__aexit__ = AsyncMock(return_value=False)
        cli.post = AsyncMock(return_value=resp)
        return cli

    async def _step5():
        # no key → truncation
        with patch("src.retrieval.rerank.get_settings", return_value=_fw_settings(api_key="")):
            chunks = [_chunk(str(i), f"t{i}", float(i)/10) for i in range(5)]
            r = await rerank_node({"query_text": "q", "retrieved_chunks": chunks})
            if len(r["retrieved_chunks"]) == 2:
                ok("No-API-key fallback truncates to top_k=2")
            else:
                fail("truncation fallback", f"Got {len(r['retrieved_chunks'])} chunks")

        # empty chunks
        with patch("src.retrieval.rerank.get_settings", return_value=_fw_settings()):
            r = await rerank_node({"query_text": "q", "retrieved_chunks": []})
            if r["retrieved_chunks"] == []:
                ok("Empty chunks input returns empty list")
            else:
                fail("empty input", str(r))

        # reorder by relevance score
        fw_data = [{"index": 1, "relevance_score": 0.97}, {"index": 0, "relevance_score": 0.42}]
        chunks = [_chunk("low", "low text", 0.4), _chunk("high", "high text", 0.9)]
        mock_cli = _mock_http(fw_data)
        with patch("src.retrieval.rerank.get_settings", return_value=_fw_settings()):
            with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_cli):
                r = await rerank_node({"query_text": "cardiac", "retrieved_chunks": chunks})
                if r["retrieved_chunks"][0]["chunk_id"] == "high":
                    ok("Fireworks rerank reorders chunks by relevance score")
                else:
                    fail("reorder", f"First chunk_id={r['retrieved_chunks'][0]['chunk_id']}")
                if abs(r["retrieved_chunks"][0]["score"] - 0.97) < 0.001:
                    ok("Reranked score correctly overwritten to 0.97")
                else:
                    fail("score update", f"Got {r['retrieved_chunks'][0]['score']}")

        # payload structure
        fw_data = [{"index": 0, "relevance_score": 0.9}]
        mock_cli = _mock_http(fw_data)
        settings = _fw_settings()
        with patch("src.retrieval.rerank.get_settings", return_value=settings):
            with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=mock_cli):
                await rerank_node({"query_text": "q", "retrieved_chunks": [_chunk()]})
                payload = mock_cli.post.call_args[1]["json"]
                assert payload["model"] == settings.fireworks_rerank_model, f"model mismatch: {payload['model']}"
                assert "/rerank" in mock_cli.post.call_args[0][0], "URL doesn't contain /rerank"
                assert "Bearer fw_key" in mock_cli.post.call_args[1]["headers"]["Authorization"]
                ok("Fireworks POST uses correct model, URL, and Authorization header")

        # HTTP error → graceful fallback
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = Exception("HTTP 500")
        err_cli = AsyncMock()
        err_cli.__aenter__ = AsyncMock(return_value=err_cli)
        err_cli.__aexit__ = AsyncMock(return_value=False)
        err_cli.post = AsyncMock(return_value=err_resp)
        chunks = [_chunk("1", "t1", 0.9), _chunk("2", "t2", 0.8), _chunk("3", "t3", 0.7)]
        with patch("src.retrieval.rerank.get_settings", return_value=_fw_settings()):
            with patch("src.retrieval.rerank.httpx.AsyncClient", return_value=err_cli):
                r = await rerank_node({"query_text": "q", "retrieved_chunks": chunks})
                if len(r["retrieved_chunks"]) == 2:
                    ok("HTTP error triggers graceful fallback to top-k")
                else:
                    fail("HTTP error fallback", f"Got {len(r['retrieved_chunks'])} chunks")

    run(_step5())

except Exception:
    fail("Step 5", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — OUTPUT ROUTING
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 6 — Output Routing")

try:
    from src.router.output_router import output_route_node
    import src.services.storage as _storage  # ensure module is loaded before patching

    async def _step6():
        base = _base_settings()
        # image route
        state = {
            "prefer_retrieved_image": True,
            "retrieved_chunks": [_chunk("c1", "img", 0.9, modality="image", storage_ref="b/img.jpg")],
        }
        with patch("src.router.output_router.get_settings", return_value=base):
          with patch.object(_storage, "generate_signed_url", return_value="http://signed/url"):
            r = await output_route_node(state)
        if r.get("image_url") == "http://signed/url":
            ok("Image route returns signed URL")
        else:
            fail("image URL", f"Got {r.get('image_url')}")
        if abs(r.get("confidence_score", 0.0) - 0.9) < 0.01:
            ok(f"confidence_score = {r['confidence_score']}")
        else:
            fail("confidence_score", f"Got {r.get('confidence_score')}")

        # pdf route
        state = {
            "include_pdf_export": True,
            "retrieved_chunks": [_chunk("c1", "pdf", 0.8, modality="pdf", storage_ref="http://pdf/url")],
        }
        with patch("src.router.output_router.get_settings", return_value=base):
            r = await output_route_node(state)
        if r.get("pdf_url") == "http://pdf/url":
            ok("PDF route returns pdf_url")
        else:
            fail("pdf_url", f"Got {r.get('pdf_url')}")

        # low confidence
        state = {"retrieved_chunks": [_chunk("c1", "t1", 0.3), _chunk("c2", "t2", 0.4)]}
        with patch("src.router.output_router.get_settings", return_value=base):
            r = await output_route_node(state)
        avg = (0.3 + 0.4) / 2
        if abs(r.get("confidence_score", 0) - avg) < 0.01:
            ok(f"Low-confidence avg score = {r['confidence_score']:.2f}")
        else:
            fail("low-confidence score", f"Got {r.get('confidence_score')}")
        if r.get("low_confidence") is True:
            ok("low_confidence flag set to True when score < threshold")
        else:
            fail("low_confidence flag", f"Got {r.get('low_confidence')}")

    run(_step6())

except Exception:
    fail("Step 6", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — CITATION GATE
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 7 — Citation Gate")

try:
    from src.generation.citation_gate import _ABSTAIN_ANSWER, citation_gate_node
    from src.pipeline.state import Citation

    async def _step7():
        # empty → abstain
        r = await citation_gate_node({"answer": "Random text.", "citations": []})
        if r.get("answer") == _ABSTAIN_ANSWER:
            ok("Empty citations triggers abstain answer")
        else:
            fail("abstain answer", f"Got '{r.get('answer')}'")
        if r.get("citations") == []:
            ok("Abstain state sets citations to []")
        else:
            fail("abstain citations", f"Got {r.get('citations')}")

        # valid → pass-through
        state = {
            "answer": "Hypertension is managed by lifestyle changes.",
            "citations": [
                Citation(chunk_id="c1", doc_id="d1", quote="test", doc_title="T1", page=1, section="S1")
            ],
        }
        r = await citation_gate_node(state)
        if r == {}:
            ok("Valid citations → gate passes through (empty partial update)")
        else:
            fail("gate pass-through", f"Got {r}")

    run(_step7())

except Exception:
    fail("Step 7", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — PIPELINE SMOKE (retrieve → rerank → citation_gate)
# ═══════════════════════════════════════════════════════════════════════════
section("STEP 8 — Full Pipeline Smoke Test")

try:
    from src.retrieval.hybrid import retrieve_node as rn
    from src.retrieval.rerank import rerank_node as rk_node
    from src.generation.citation_gate import _ABSTAIN_ANSWER, citation_gate_node

    async def _step8():
        # Smoke: retrieve → rerank (fallback) → verify chunks flow
        with patch("src.retrieval.hybrid.get_settings", return_value=_base_settings()):
         with patch.object(_gem, "embed_query", new=AsyncMock(return_value=_vec())):
            with patch("src.retrieval.hybrid._query_weaviate_sync",
                       return_value=[_chunk("c1", "text1", 0.9), _chunk("c2", "text2", 0.7)]):
                state = {"query_text": "What is the dosage for aspirin?"}
                state.update(await rn(state))

        # rerank with top_k=1 fallback (no API key)
        settings_1 = SimpleNamespace(
            fireworks_api_key="",
            fireworks_rerank_model="model",
            fireworks_base_url="https://api.fireworks.ai/inference/v1",
            rerank_top_k=1,
        )
        with patch("src.retrieval.rerank.get_settings", return_value=settings_1):
            state.update(await rk_node(state))

        if len(state["retrieved_chunks"]) == 1:
            ok("Pipeline: retrieve(2) → rerank(top_k=1) = 1 chunk in state")
        else:
            fail("pipeline chunk count", f"Got {len(state['retrieved_chunks'])}")

        if state["retrieved_chunks"][0]["chunk_id"] == "c1":
            ok("Pipeline: highest-score chunk c1 survived rerank")
        else:
            fail("pipeline ordering", f"Got {state['retrieved_chunks'][0]['chunk_id']}")

        # Smoke: empty → abstain
        empty_state: dict = {
            "query_text": "empty", "retrieved_chunks": [],
            "answer": "some hallucinated text", "citations": []
        }
        with patch("src.retrieval.rerank.get_settings", return_value=settings_1):
            empty_state.update(await rk_node(empty_state))
        empty_state.update(await citation_gate_node(empty_state))

        if empty_state["answer"] == _ABSTAIN_ANSWER:
            ok("Pipeline: empty chunks → citation gate abstains correctly")
        else:
            fail("pipeline abstain", f"Got '{empty_state['answer']}'")

    run(_step8())

except Exception:
    fail("Step 8", traceback.format_exc(limit=5))


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'═'*60}")
print(f"  {GREEN}{passed}{RESET}/{total} passed   {RED}{failed}{RESET}/{total} failed")
print(f"{'═'*60}\n")

sys.exit(0 if failed == 0 else 1)
