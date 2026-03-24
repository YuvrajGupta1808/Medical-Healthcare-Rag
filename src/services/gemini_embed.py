from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

# Model and dimension are driven by config (gemini-embedding-2-preview).
# Dimension is flexible: 128–3072; recommended values: 768, 1536, 3072.
_settings_cache: tuple[str, int] | None = None


def _model_and_dim() -> tuple[str, int]:
    """Return (model_name, output_dimension) from cached settings."""
    global _settings_cache
    if _settings_cache is None:
        s = get_settings()
        _settings_cache = (s.gemini_embedding_model, s.embedding_dimension)
    return _settings_cache


# Limit concurrent AI Studio calls to stay within rate limits
_semaphore = asyncio.Semaphore(10)

# Module-level client — initialised lazily on first embed call
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """
    Return a cached Google AI Studio client.
    Raises RuntimeError if GEMINI_API_KEY is not configured.
    """
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Get a key at https://aistudio.google.com/apikey and add it to .env."
            )
        model, dim = _model_and_dim()
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.info(
            "Google AI Studio client initialised (model=%s, dimension=%d)", model, dim
        )
    return _client


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def embed_text(text: str) -> list[float]:
    """
    Embed a text string using gemini-embedding-2-preview.

    Returns a float list whose length equals ``embedding_dimension`` from
    settings (default 3072). Fully async — no thread pool needed.

    Input is truncated to 8 192 tokens as per model limits.
    """
    model, dim = _model_and_dim()
    async with _semaphore:
        client = _get_client()
        result = await client.aio.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dim,
            ),
        )
        return list(result.embeddings[0].values)


async def embed_query(text: str) -> list[float]:
    """
    Embed a *query* string (uses RETRIEVAL_QUERY task type for better recall).

    Use this for search-time query embedding; use ``embed_text`` for documents.
    """
    model, dim = _model_and_dim()
    async with _semaphore:
        client = _get_client()
        result = await client.aio.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=dim,
            ),
        )
        return list(result.embeddings[0].values)


async def embed_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[float]:
    """
    Embed raw image bytes using gemini-embedding-2-preview multimodal support.

    Supported mime types: image/jpeg, image/png, image/webp, image/gif, etc.
    Returns a float list of length ``embedding_dimension`` from settings.
    """
    model, dim = _model_and_dim()
    async with _semaphore:
        client = _get_client()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        result = await client.aio.models.embed_content(
            model=model,
            contents=image_part,
            config=types.EmbedContentConfig(
                output_dimensionality=dim,
            ),
        )
        logger.info("embed_image: embedded %d bytes (%s)", len(image_bytes), mime_type)
        return list(result.embeddings[0].values)
