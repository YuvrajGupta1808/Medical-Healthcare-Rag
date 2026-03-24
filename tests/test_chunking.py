"""
Unit tests for the token-aware chunker.
No live services required.
"""
from __future__ import annotations

from pathlib import Path

import tiktoken

from src.ingest.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_pages, make_chunk_id
from src.ingest.loaders import RawPage

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_medical.txt"
_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(text: str, doc_id: str = "doc1", page: int = 1) -> RawPage:
    return RawPage(text=text, page_number=page, section="Test", doc_id=doc_id, doc_title="Test Doc")


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chunk_id_is_deterministic():
    """Same doc_id + index always produces the same chunk_id."""
    assert make_chunk_id("doc123", 0) == make_chunk_id("doc123", 0)
    assert make_chunk_id("doc123", 0) != make_chunk_id("doc123", 1)
    assert make_chunk_id("docA", 0) != make_chunk_id("docB", 0)


def test_chunk_id_length():
    """chunk_id is always 16 hex chars."""
    cid = make_chunk_id("any-doc", 42)
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


def test_single_short_page_produces_one_chunk():
    """A page with fewer tokens than chunk_size (but >= MIN_CHUNK_TOKENS) yields exactly one chunk."""
    # 60+ tokens: well above MIN_CHUNK_TOKENS=50 and far below chunk_size=600
    short_text = ("The patient presented with elevated blood pressure readings across "
                  "three consecutive visits. Lifestyle modifications including dietary "
                  "changes and increased physical activity were recommended before "
                  "initiating pharmacological therapy.")
    assert _token_count(short_text) >= 50, "Test text must meet MIN_CHUNK_TOKENS"
    page = _make_page(short_text)
    chunks = chunk_pages([page], chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].text == short_text.strip()


def test_chunk_sizes_within_bounds():
    """All chunks (except possibly the last) must be <= chunk_size tokens."""
    text = " ".join(["word"] * 2000)  # 2000 tokens worth of content
    page = _make_page(text)
    chunks = chunk_pages([page], chunk_size=600, overlap=100)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:  # all but last must be at or near chunk_size
        count = _token_count(chunk.text)
        assert count <= 600, f"Chunk too large: {count} tokens"


def test_overlap_creates_shared_tokens():
    """Consecutive chunks should share content due to overlap."""
    text = " ".join([f"token{i}" for i in range(1000)])
    page = _make_page(text)
    chunks = chunk_pages([page], chunk_size=200, overlap=50)

    # The end of chunk N should appear somewhere in chunk N+1
    assert len(chunks) >= 2
    end_of_first = chunks[0].text.split()[-10:]  # last 10 words of first chunk
    start_of_second = chunks[1].text
    # At least some overlap words should appear in the second chunk
    assert any(word in start_of_second for word in end_of_first)


def test_metadata_preserved():
    """Page number, section, doc_id, and doc_title pass through to all chunks."""
    # Use text that is well above MIN_CHUNK_TOKENS=50
    text = ("Metformin is the first-line pharmacological treatment for type 2 diabetes "
            "in most clinical guidelines. It works primarily by reducing hepatic glucose "
            "production and improving insulin sensitivity in peripheral tissues. "
            "Common side effects include gastrointestinal discomfort, which can be "
            "minimised by taking the medication with food and using extended-release "
            "formulations. Renal function should be monitored regularly.")
    page = RawPage(
        text=text,
        page_number=7,
        section="Methods",
        doc_id="test-doc-001",
        doc_title="Test Paper",
    )
    chunks = chunk_pages([page])
    assert len(chunks) >= 1
    assert chunks[0].page == 7
    assert chunks[0].section == "Methods"
    assert chunks[0].doc_id == "test-doc-001"
    assert chunks[0].doc_title == "Test Paper"
    assert chunks[0].modality_type == "text"


def test_empty_page_list_returns_empty():
    """Empty input returns empty list without error."""
    assert chunk_pages([]) == []


def test_empty_text_page_is_skipped():
    """Pages with only whitespace are skipped."""
    page = _make_page("   \n\n\t  ")
    chunks = chunk_pages([page])
    assert chunks == []


def test_fixture_file_chunks_correctly():
    """The sample medical fixture produces multiple chunks with correct structure."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    page = _make_page(text, doc_id="fixture-doc")
    chunks = chunk_pages([page], chunk_size=200, overlap=30)

    assert len(chunks) >= 2
    # All chunk IDs are unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    # No empty text chunks
    assert all(c.text.strip() for c in chunks)


def test_multi_page_doc_has_unique_chunk_ids():
    """Chunks across multiple pages must all have unique IDs."""
    pages = [
        RawPage(text="Page one content. " * 50, page_number=1, doc_id="multi", doc_title="D"),
        RawPage(text="Page two content. " * 50, page_number=2, doc_id="multi", doc_title="D"),
    ]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found across pages"
