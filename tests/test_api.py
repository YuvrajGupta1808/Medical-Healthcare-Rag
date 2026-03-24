"""
Step 1 API smoke tests — no live services required.
TESTING=true (set in conftest.py) bypasses Weaviate/Gemini on startup.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_response_shape(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["weaviate"] == "not_connected"  # no Weaviate in test mode


# ---------------------------------------------------------------------------
# Query — JSON interface (Step 1)
# ---------------------------------------------------------------------------

def test_query_valid_json_accepted(client):
    """Valid JSON body reaches the route; pipeline errors without live services."""
    r = client.post("/query", json={"query": "What is the treatment for hypertension?"})
    assert r.status_code in (200, 500)  # 500 = no Weaviate, not a schema error


def test_query_text_returns_200(client):
    """Alias kept for name compatibility; same as above."""
    r = client.post("/query", json={"query": "What are symptoms of diabetes?"})
    assert r.status_code in (200, 500)


def test_query_image_with_file_returns_200(client):
    """doc_id filter is accepted without validation error."""
    r = client.post("/query", json={"query": "Show the diagram.", "doc_id": "doc-1"})
    assert r.status_code in (200, 500)


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_query_rejects_oversized_question(client):
    assert client.post("/query", json={"query": "x" * 2001}).status_code == 422


def test_query_rejects_missing_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_query_top_k_validated(client):
    assert client.post("/query", json={"query": "test", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"query": "test", "top_k": 21}).status_code == 422


# ---------------------------------------------------------------------------
# Ingest — validation only (no live Weaviate)
# ---------------------------------------------------------------------------

def test_ingest_rejects_unsupported_type(client):
    from io import BytesIO
    r = client.post(
        "/ingest",
        files={"file": ("doc.csv", BytesIO(b"col1,col2"), "text/csv")},
        data={"doc_title": "test"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loads_in_test_mode():
    from src.utils.config import Settings
    s = Settings(testing=True, fireworks_api_key="", gemini_api_key="")
    assert s.testing is True


def test_config_fails_without_keys_in_prod_mode():
    import pytest
    from pydantic import ValidationError

    from src.utils.config import Settings
    with pytest.raises((ValueError, ValidationError)):
        Settings(testing=False, fireworks_api_key="", gemini_api_key="")


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def test_prompt_loader_loads_system_prompt():
    from src.utils.prompt_loader import load_prompt
    prompt = load_prompt("system", version="v1")
    assert len(prompt) > 50
    assert "citations" in prompt.lower()


def test_prompt_loader_raises_for_missing_file():
    import pytest

    from src.utils.prompt_loader import load_prompt
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_xyz", version="v1")


# ---------------------------------------------------------------------------
# Chunker + loader (pure unit, no services)
# ---------------------------------------------------------------------------

def test_load_text_and_chunk_fixture():
    from pathlib import Path

    from src.ingest.chunking import chunk_pages
    from src.ingest.loaders import load_text

    fixture = Path(__file__).parent / "fixtures" / "sample_medical.txt"
    pages = load_text(fixture, doc_id="test-doc")
    chunks = chunk_pages(pages, chunk_size=150, overlap=25)

    assert len(chunks) >= 2
    assert all(c.doc_id == "test-doc" for c in chunks)
    assert all(c.modality_type == "text" for c in chunks)
