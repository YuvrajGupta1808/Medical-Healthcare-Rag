"""
Unit tests for Step 2: Input Router Normalization.

Verifies that every modality produces a valid, consistent RAGState
with required fields populated.  Tests run without any live services.

Plan reference (Step 2 done criteria):
  - Every modality yields valid state
  - Normalized JSON state shape is identical except modality fields
  - Does not break downstream nodes (input_router_node validation passes)
"""
from __future__ import annotations

import pytest

from src.router.input_router import InputRouter, input_router_node
from src.schemas.query import Modality

QUESTION = "What is the recommended treatment for hypertension?"


@pytest.fixture
def router() -> InputRouter:
    return InputRouter()


# ---------------------------------------------------------------------------
# Shared invariant helper
# ---------------------------------------------------------------------------

def assert_base_invariants(state: dict, expected_modality: Modality) -> None:
    """Every modality must satisfy these fields after routing."""
    assert state["request_id"], "request_id must be a non-empty string"
    assert state["query_text"], "query_text must be a non-empty string"
    assert state["original_question"], "original_question must be a non-empty string"
    assert state["modality"] == expected_modality
    assert isinstance(state["modality_metadata"], dict)
    assert "storage_ref" in state  # key must be present (value may be None)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

async def test_text_query_text_equals_question(router):
    state = await router.route(modality=Modality.TEXT, question=QUESTION)
    assert_base_invariants(state, Modality.TEXT)
    assert state["query_text"] == QUESTION


async def test_text_original_question_equals_question(router):
    state = await router.route(modality=Modality.TEXT, question=QUESTION)
    assert state["original_question"] == QUESTION


async def test_text_storage_ref_is_none(router):
    state = await router.route(modality=Modality.TEXT, question=QUESTION)
    assert state["storage_ref"] is None, "TEXT modality must not produce a storage_ref"


async def test_text_metadata_is_empty_dict(router):
    state = await router.route(modality=Modality.TEXT, question=QUESTION)
    assert state["modality_metadata"] == {}


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

async def test_image_sets_storage_ref(router, mock_image_file):
    state = await router.route(
        modality=Modality.IMAGE, question=QUESTION, file=mock_image_file
    )
    assert_base_invariants(state, Modality.IMAGE)
    assert state["storage_ref"] is not None
    assert state["storage_ref"].startswith("images/")


async def test_image_storage_ref_contains_request_id(router, mock_image_file):
    state = await router.route(
        modality=Modality.IMAGE, question=QUESTION, file=mock_image_file
    )
    assert state["request_id"] in state["storage_ref"]


async def test_image_metadata_has_required_keys(router, mock_image_file):
    state = await router.route(
        modality=Modality.IMAGE, question=QUESTION, file=mock_image_file
    )
    meta = state["modality_metadata"]
    assert meta["filename"] == "chest_xray.jpg"
    assert meta["content_type"] == "image/jpeg"
    assert meta["size_bytes"] > 0
    assert "description" in meta


async def test_image_query_text_contains_original_question(router, mock_image_file):
    state = await router.route(
        modality=Modality.IMAGE, question=QUESTION, file=mock_image_file
    )
    assert QUESTION in state["query_text"]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

async def test_audio_sets_storage_ref(router, mock_audio_file):
    state = await router.route(
        modality=Modality.AUDIO, question=QUESTION, file=mock_audio_file
    )
    assert_base_invariants(state, Modality.AUDIO)
    assert state["storage_ref"] is not None
    assert state["storage_ref"].startswith("audio/")


async def test_audio_metadata_has_transcript(router, mock_audio_file):
    state = await router.route(
        modality=Modality.AUDIO, question=QUESTION, file=mock_audio_file
    )
    assert "transcript" in state["modality_metadata"]
    assert state["modality_metadata"]["transcript"]  # non-empty


async def test_audio_with_question_combines_question_and_transcript(router, mock_audio_file):
    state = await router.route(
        modality=Modality.AUDIO, question=QUESTION, file=mock_audio_file
    )
    assert QUESTION in state["query_text"]
    assert "Transcript" in state["query_text"]


async def test_audio_without_question_uses_transcript_as_query_text(router, mock_audio_file):
    """When no text question is provided, the transcript IS the query."""
    state = await router.route(
        modality=Modality.AUDIO, question="", file=mock_audio_file
    )
    transcript = state["modality_metadata"]["transcript"]
    assert state["query_text"] == transcript
    assert state["original_question"] == transcript


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

async def test_pdf_sets_storage_ref(router, mock_pdf_file):
    state = await router.route(
        modality=Modality.PDF, question=QUESTION, file=mock_pdf_file
    )
    assert_base_invariants(state, Modality.PDF)
    assert state["storage_ref"] is not None
    assert state["storage_ref"].startswith("pdfs/")


async def test_pdf_query_text_equals_question(router, mock_pdf_file):
    """PDF: question drives retrieval; PDF content is stored by reference."""
    state = await router.route(
        modality=Modality.PDF, question=QUESTION, file=mock_pdf_file
    )
    assert state["query_text"] == QUESTION


async def test_pdf_metadata_has_filename_and_size(router, mock_pdf_file):
    state = await router.route(
        modality=Modality.PDF, question=QUESTION, file=mock_pdf_file
    )
    meta = state["modality_metadata"]
    assert meta["filename"] == "clinical_trial_report.pdf"
    assert meta["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

async def test_video_sets_storage_ref(router, mock_video_file):
    state = await router.route(
        modality=Modality.VIDEO, question=QUESTION, file=mock_video_file
    )
    assert_base_invariants(state, Modality.VIDEO)
    assert state["storage_ref"] is not None
    assert state["storage_ref"].startswith("videos/")


async def test_video_query_text_contains_original_question(router, mock_video_file):
    state = await router.route(
        modality=Modality.VIDEO, question=QUESTION, file=mock_video_file
    )
    assert QUESTION in state["query_text"]


async def test_video_metadata_has_description(router, mock_video_file):
    state = await router.route(
        modality=Modality.VIDEO, question=QUESTION, file=mock_video_file
    )
    assert "description" in state["modality_metadata"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_missing_file_raises_value_error(router):
    with pytest.raises(ValueError, match="file upload is required"):
        await router.route(modality=Modality.IMAGE, question=QUESTION, file=None)


async def test_custom_request_id_is_preserved(router):
    custom_id = "custom-uuid-1234"
    state = await router.route(
        modality=Modality.TEXT, question=QUESTION, request_id=custom_id
    )
    assert state["request_id"] == custom_id


# ---------------------------------------------------------------------------
# State shape uniformity — the key Step 2 invariant
# ---------------------------------------------------------------------------

async def test_all_modalities_produce_same_required_keys(
    router, mock_image_file, mock_audio_file, mock_pdf_file, mock_video_file
):
    """
    Core invariant: required state keys are present for ALL modalities.
    This ensures downstream nodes can safely access these fields regardless
    of which modality initiated the request.
    """
    required_keys = {
        "request_id",
        "query_text",
        "original_question",
        "modality",
        "modality_metadata",
        "storage_ref",
    }
    test_cases = [
        (Modality.TEXT, QUESTION, None),
        (Modality.IMAGE, QUESTION, mock_image_file),
        (Modality.AUDIO, QUESTION, mock_audio_file),
        (Modality.PDF, QUESTION, mock_pdf_file),
        (Modality.VIDEO, QUESTION, mock_video_file),
    ]
    for modality, question, file in test_cases:
        state = await router.route(modality=modality, question=question, file=file)
        missing = required_keys - state.keys()
        assert not missing, (
            f"State for modality={modality.value!r} is missing keys: {missing}"
        )


async def test_all_modalities_query_text_is_non_empty(
    router, mock_image_file, mock_audio_file, mock_pdf_file, mock_video_file
):
    """query_text must always be retrieval-ready (non-empty) for every modality."""
    test_cases = [
        (Modality.TEXT, QUESTION, None),
        (Modality.IMAGE, QUESTION, mock_image_file),
        (Modality.AUDIO, QUESTION, mock_audio_file),
        (Modality.PDF, QUESTION, mock_pdf_file),
        (Modality.VIDEO, QUESTION, mock_video_file),
    ]
    for modality, question, file in test_cases:
        state = await router.route(modality=modality, question=question, file=file)
        assert state["query_text"], (
            f"query_text is empty for modality={modality.value!r}"
        )


# ---------------------------------------------------------------------------
# LangGraph node — input_router_node
# ---------------------------------------------------------------------------

async def test_input_router_node_passes_valid_state():
    state = {
        "request_id": "test-request-id-123",
        "query_text": "What are the side effects of metformin?",
        "original_question": "What are the side effects of metformin?",
        "modality": Modality.TEXT,
        "modality_metadata": {},
        "storage_ref": None,
    }
    result = await input_router_node(state)
    # No-op for valid state: returns empty dict (no changes needed)
    assert result == {} or not result.get("error")


async def test_input_router_node_sets_error_on_missing_query_text():
    state = {
        "request_id": "test-request-id-123",
        "modality": Modality.TEXT,
    }
    result = await input_router_node(state)
    assert "error" in result
    assert "query_text" in result["error"]


async def test_input_router_node_sets_error_on_missing_request_id():
    state = {
        "query_text": "some question",
        "modality": Modality.TEXT,
    }
    result = await input_router_node(state)
    assert "error" in result
    assert "request_id" in result["error"]
