"""
Unit tests for src/services/gemini_embed.py

Tests cover:
  - embed_text  (RETRIEVAL_DOCUMENT task type)
  - embed_query (RETRIEVAL_QUERY task type)
  - embed_image (multimodal, Part.from_bytes)
  - Config-driven model name and output dimension
  - RuntimeError when GEMINI_API_KEY is absent (non-test mode)

All tests run without any live Google AI Studio calls.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 3072
MODEL = "gemini-embedding-2-preview"


def _make_embed_result(dim: int = DIM) -> SimpleNamespace:
    """Build a fake EmbedContentResponse with the expected shape."""
    values = [0.1] * dim
    embedding = SimpleNamespace(values=values)
    return SimpleNamespace(embeddings=[embedding])


def _reset_module_cache() -> None:
    """
    Clear module-level singletons so each test starts with a clean slate.
    Necessary because _client and _settings_cache are cached at module level.
    """
    import src.services.gemini_embed as mod

    mod._client = None
    mod._settings_cache = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset module globals before and after every test."""
    _reset_module_cache()
    yield
    _reset_module_cache()


@pytest.fixture
def mock_genai_client():
    """
    Patch genai.Client so no real HTTP connections are made.
    Returns the mock async embed_content callable.
    """
    embed_mock = AsyncMock(return_value=_make_embed_result())

    client_instance = MagicMock()
    client_instance.aio.models.embed_content = embed_mock

    with patch("src.services.gemini_embed.genai.Client", return_value=client_instance):
        yield embed_mock


@pytest.fixture
def mock_settings(mock_genai_client):
    """Patch get_settings to return controlled config values."""
    settings = SimpleNamespace(
        gemini_api_key="fake-key",
        gemini_embedding_model=MODEL,
        embedding_dimension=DIM,
        testing=True,
    )
    with patch("src.services.gemini_embed.get_settings", return_value=settings):
        yield settings


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

async def test_embed_text_returns_list_of_floats(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text

    result = await embed_text("Hypertension treatment guidelines.")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


async def test_embed_text_returns_correct_dimension(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text

    result = await embed_text("Sample medical text.")
    assert len(result) == DIM


async def test_embed_text_uses_retrieval_document_task_type(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text
    from google.genai import types

    await embed_text("Diabetes management protocol.")

    _, call_kwargs = mock_genai_client.call_args
    config: types.EmbedContentConfig = call_kwargs["config"]
    assert config.task_type == "RETRIEVAL_DOCUMENT"


async def test_embed_text_passes_correct_model(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text

    await embed_text("Some text.")

    _, call_kwargs = mock_genai_client.call_args
    assert call_kwargs["model"] == MODEL


async def test_embed_text_passes_output_dimensionality(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text
    from google.genai import types

    await embed_text("Some text.")

    _, call_kwargs = mock_genai_client.call_args
    config: types.EmbedContentConfig = call_kwargs["config"]
    assert config.output_dimensionality == DIM


async def test_embed_text_passes_contents_as_string(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_text

    text = "Patient exhibits signs of pneumonia."
    await embed_text(text)

    _, call_kwargs = mock_genai_client.call_args
    assert call_kwargs["contents"] == text


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------

async def test_embed_query_returns_list_of_floats(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_query

    result = await embed_query("What is the treatment for sepsis?")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


async def test_embed_query_returns_correct_dimension(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_query

    result = await embed_query("Dosage for ibuprofen?")
    assert len(result) == DIM


async def test_embed_query_uses_retrieval_query_task_type(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_query
    from google.genai import types

    await embed_query("What are the symptoms of appendicitis?")

    _, call_kwargs = mock_genai_client.call_args
    config: types.EmbedContentConfig = call_kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"


async def test_embed_query_task_type_differs_from_embed_text(
    mock_settings, mock_genai_client
):
    """embed_query and embed_text must use different task types."""
    from src.services.gemini_embed import embed_query, embed_text
    from google.genai import types

    await embed_text("document text")
    _, doc_kwargs = mock_genai_client.call_args
    doc_config: types.EmbedContentConfig = doc_kwargs["config"]

    await embed_query("query text")
    _, query_kwargs = mock_genai_client.call_args
    query_config: types.EmbedContentConfig = query_kwargs["config"]

    assert doc_config.task_type != query_config.task_type


async def test_embed_query_passes_correct_model(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_query

    await embed_query("test query")

    _, call_kwargs = mock_genai_client.call_args
    assert call_kwargs["model"] == MODEL


# ---------------------------------------------------------------------------
# embed_image
# ---------------------------------------------------------------------------

JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 96


async def test_embed_image_returns_list_of_floats(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_image

    result = await embed_image(JPEG_BYTES, mime_type="image/jpeg")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


async def test_embed_image_returns_correct_dimension(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_image

    result = await embed_image(JPEG_BYTES, mime_type="image/jpeg")
    assert len(result) == DIM


async def test_embed_image_passes_part_object(mock_settings, mock_genai_client):
    """embed_image must pass a Part (not raw bytes) to embed_content."""
    from google.genai import types

    from src.services.gemini_embed import embed_image

    await embed_image(JPEG_BYTES, mime_type="image/jpeg")

    _, call_kwargs = mock_genai_client.call_args
    contents = call_kwargs["contents"]
    assert isinstance(contents, types.Part), (
        f"Expected types.Part, got {type(contents)}"
    )


async def test_embed_image_passes_correct_model(mock_settings, mock_genai_client):
    from src.services.gemini_embed import embed_image

    await embed_image(JPEG_BYTES)

    _, call_kwargs = mock_genai_client.call_args
    assert call_kwargs["model"] == MODEL


async def test_embed_image_passes_output_dimensionality(mock_settings, mock_genai_client):
    from google.genai import types

    from src.services.gemini_embed import embed_image

    await embed_image(JPEG_BYTES)

    _, call_kwargs = mock_genai_client.call_args
    config: types.EmbedContentConfig = call_kwargs["config"]
    assert config.output_dimensionality == DIM


async def test_embed_image_default_mime_is_jpeg(mock_settings, mock_genai_client):
    """Default mime_type should be image/jpeg."""
    from google.genai import types

    from src.services.gemini_embed import embed_image

    await embed_image(JPEG_BYTES)  # no mime_type arg

    _, call_kwargs = mock_genai_client.call_args
    part: types.Part = call_kwargs["contents"]
    assert part.inline_data.mime_type == "image/jpeg"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

async def test_model_and_dim_reads_from_settings():
    """_model_and_dim() must return values from get_settings()."""
    settings = SimpleNamespace(
        gemini_api_key="key",
        gemini_embedding_model="gemini-embedding-2-preview",
        embedding_dimension=1536,
        testing=True,
    )
    with patch("src.services.gemini_embed.get_settings", return_value=settings):
        from src.services.gemini_embed import _model_and_dim

        model, dim = _model_and_dim()
        assert model == "gemini-embedding-2-preview"
        assert dim == 1536


def test_config_default_model():
    from src.utils.config import Settings

    s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
    assert s.gemini_embedding_model == "gemini-embedding-2-preview"


def test_config_default_dimension():
    from src.utils.config import Settings

    s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
    assert s.embedding_dimension == 3072


def test_config_dimension_can_be_overridden():
    from src.utils.config import Settings

    s = Settings(
        testing=True,
        fireworks_api_key="",
        gemini_api_key="",
        embedding_dimension=768,
    )
    assert s.embedding_dimension == 768


# ---------------------------------------------------------------------------
# Error handling — missing API key
# ---------------------------------------------------------------------------

def test_get_client_raises_when_api_key_missing():
    """_get_client() must raise RuntimeError when GEMINI_API_KEY is empty."""
    settings = SimpleNamespace(
        gemini_api_key="",
        gemini_embedding_model=MODEL,
        embedding_dimension=DIM,
        testing=False,
    )
    with patch("src.services.gemini_embed.get_settings", return_value=settings):
        import src.services.gemini_embed as mod

        mod._client = None  # ensure not cached from a previous test
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            mod._get_client()
