"""
Shared test fixtures for the Medical Healthcare RAG test suite.

TESTING=true is set at module level so that:
  - pydantic-settings skips production API-key validation
  - app lifespan skips Weaviate / MinIO connection attempts
This must happen before any src.* imports are triggered.
"""
from __future__ import annotations

import os

# Set BEFORE any app imports so get_settings() caches with testing=True.
# os.environ takes precedence over .env file in pydantic-settings.
os.environ["TESTING"] = "true"
os.environ["CONFIDENCE_THRESHOLD"] = "0.5"
os.environ.setdefault("FIREWORKS_API_KEY", "test_key_placeholder")
os.environ.setdefault("GEMINI_API_KEY", "test_gemini_key_placeholder")

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.api.app import app as _app


# ---------------------------------------------------------------------------
# Mock UploadFile
# ---------------------------------------------------------------------------

class MockUploadFile:
    """
    Lightweight stand-in for FastAPI's UploadFile used in unit tests.
    Supports a single async ``read()`` call.
    """

    def __init__(
        self,
        filename: str,
        content: bytes = b"",
        content_type: str = "application/octet-stream",
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        return self._content


# ---------------------------------------------------------------------------
# Per-modality file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_image_file() -> MockUploadFile:
    """Minimal JPEG header bytes."""
    return MockUploadFile(
        filename="chest_xray.jpg",
        content=b"\xff\xd8\xff\xe0" + b"\x00" * 96,
        content_type="image/jpeg",
    )


@pytest.fixture
def mock_audio_file() -> MockUploadFile:
    """Minimal MP3 sync-word bytes."""
    return MockUploadFile(
        filename="patient_consultation.mp3",
        content=b"\xff\xfb\x90\x00" + b"\x00" * 96,
        content_type="audio/mpeg",
    )


@pytest.fixture
def mock_pdf_file() -> MockUploadFile:
    """Minimal PDF header bytes."""
    return MockUploadFile(
        filename="clinical_trial_report.pdf",
        content=b"%PDF-1.4\n" + b"\x00" * 91,
        content_type="application/pdf",
    )


@pytest.fixture
def mock_video_file() -> MockUploadFile:
    """Minimal MP4 ftyp-box bytes."""
    return MockUploadFile(
        filename="surgical_procedure.mp4",
        content=b"\x00\x00\x00\x18ftyp" + b"\x00" * 90,
        content_type="video/mp4",
    )


# ---------------------------------------------------------------------------
# HTTP test clients
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Return the FastAPI app (lifespan skipped in test mode)."""
    return _app


@pytest.fixture(scope="session")
def client(app):
    """Synchronous TestClient for fast HTTP-level tests."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client() -> AsyncClient:
    """AsyncClient wired to the FastAPI app via ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=_app),
        base_url="http://test",
    ) as ac:
        yield ac
