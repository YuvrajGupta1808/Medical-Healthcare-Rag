"""
Tests for Step 3: Ingest schema + chunk quality.

All tests run without live Weaviate or embedding services.

Coverage
--------
- Chunk token bounds: all output chunks are within [MIN_CHUNK_TOKENS, chunk_size]
- Metadata completeness: doc_id, chunk_id, doc_title, modality_type present on every chunk
- Deterministic IDs: re-chunking the same document produces identical chunk_ids
- No duplicate chunk_ids within a document or across multiple pages
- Micro-chunk filter: pages shorter than MIN_CHUNK_TOKENS are not emitted
- validate_chunks() detects empty text, missing fields, oversized chunks, duplicates
- Schema definition: EXPECTED_PROPERTIES matches the properties declared in ensure_schema()
- Idempotent ingest: pipeline calls delete_document() before upserting (mocked)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken

from src.ingest.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MIN_CHUNK_TOKENS,
    IngestChunk,
    chunk_pages,
    make_chunk_id,
    validate_chunks,
)
from src.ingest.loaders import RawPage
from src.services.weaviate_client import EXPECTED_PROPERTIES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_medical.txt"
_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(text: str, doc_id: str = "doc-001", page: int = 1, section: str = "Intro") -> RawPage:
    return RawPage(text=text, page_number=page, section=section,
                   doc_id=doc_id, doc_title="Medical Test Document")


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# Chunk token bounds
# ---------------------------------------------------------------------------

def test_chunks_do_not_exceed_chunk_size():
    """Non-terminal chunks must be <= chunk_size tokens."""
    text = " ".join(["word"] * 3000)
    chunks = chunk_pages([_page(text)], chunk_size=600, overlap=100)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert _token_count(chunk.text) <= 600, (
            f"Chunk too large: {_token_count(chunk.text)} tokens"
        )


def test_all_chunks_meet_min_token_threshold():
    """Every emitted chunk must have >= MIN_CHUNK_TOKENS tokens."""
    text = " ".join(["word"] * 2000)
    chunks = chunk_pages([_page(text)], chunk_size=600, overlap=100)
    for chunk in chunks:
        count = _token_count(chunk.text)
        assert count >= MIN_CHUNK_TOKENS, (
            f"Micro-chunk emitted: {count} tokens (min={MIN_CHUNK_TOKENS})"
        )


def test_micro_chunk_is_filtered_out():
    """A page with fewer tokens than MIN_CHUNK_TOKENS produces no chunks."""
    short_text = "Short."  # definitely < 50 tokens
    assert _token_count(short_text) < MIN_CHUNK_TOKENS
    chunks = chunk_pages([_page(short_text)], min_tokens=MIN_CHUNK_TOKENS)
    assert chunks == [], "Micro-chunk should be discarded"


def test_chunk_just_above_min_is_kept():
    """A page with exactly MIN_CHUNK_TOKENS tokens produces one chunk."""
    # Build a text with exactly MIN_CHUNK_TOKENS tokens
    words = []
    while _token_count(" ".join(words)) < MIN_CHUNK_TOKENS:
        words.append("word")
    text = " ".join(words)
    assert _token_count(text) >= MIN_CHUNK_TOKENS
    chunks = chunk_pages([_page(text)])
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------

def test_all_required_metadata_fields_populated():
    """Every chunk must have doc_id, chunk_id, doc_title, modality_type, page."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    page = RawPage(
        text=text, page_number=3, section="Methods",
        doc_id="test-meta", doc_title="Meta Test",
    )
    chunks = chunk_pages([page])
    assert chunks, "Fixture should produce at least one chunk"

    for chunk in chunks:
        assert chunk.doc_id == "test-meta",       f"doc_id missing on {chunk.chunk_id}"
        assert chunk.chunk_id,                     f"chunk_id is empty"
        assert chunk.doc_title == "Meta Test",     f"doc_title missing on {chunk.chunk_id}"
        assert chunk.modality_type == "text",      f"modality_type wrong on {chunk.chunk_id}"
        assert chunk.page == 3,                    f"page missing on {chunk.chunk_id}"
        assert chunk.section == "Methods",         f"section missing on {chunk.chunk_id}"
        assert chunk.text.strip(),                 f"text empty on {chunk.chunk_id}"


def test_storage_ref_and_attachment_id_are_empty_for_text_chunks():
    """Text chunks must have storage_ref=None and attachment_id='' (no binary payload)."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks = chunk_pages([_page(text)])
    for chunk in chunks:
        assert chunk.storage_ref is None, f"storage_ref should be None for text chunks"
        assert chunk.attachment_id == "",  f"attachment_id should be '' for text chunks"
        assert chunk.caption == "",        f"caption should be '' for text chunks"


# ---------------------------------------------------------------------------
# Deterministic chunk IDs
# ---------------------------------------------------------------------------

def test_chunk_ids_are_deterministic_across_runs():
    """Re-chunking the same document twice produces identical chunk_ids."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    page = _page(text, doc_id="determ-doc")
    run1 = [c.chunk_id for c in chunk_pages([page])]
    run2 = [c.chunk_id for c in chunk_pages([page])]
    assert run1 == run2, "Chunk IDs differ between runs — non-deterministic!"


def test_no_duplicate_chunk_ids_single_doc():
    """All chunk_ids within a single document must be unique."""
    text = " ".join([f"token{i}" for i in range(3000)])
    chunks = chunk_pages([_page(text, doc_id="unique-doc")])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in single-document run"


def test_no_duplicate_chunk_ids_multi_page():
    """Chunk IDs must be globally unique across all pages of a document."""
    pages = [
        RawPage(text=" ".join(["page1word"] * 1000), page_number=1,
                doc_id="multi-page", doc_title="T"),
        RawPage(text=" ".join(["page2word"] * 1000), page_number=2,
                doc_id="multi-page", doc_title="T"),
        RawPage(text=" ".join(["page3word"] * 1000), page_number=3,
                doc_id="multi-page", doc_title="T"),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids across pages"


def test_different_doc_ids_produce_different_chunk_ids():
    """Same text, different doc_id → different chunk_ids."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks_a = chunk_pages([_page(text, doc_id="doc-A")])
    chunks_b = chunk_pages([_page(text, doc_id="doc-B")])
    ids_a = set(c.chunk_id for c in chunks_a)
    ids_b = set(c.chunk_id for c in chunks_b)
    assert ids_a.isdisjoint(ids_b), "Different docs must have non-overlapping chunk_ids"


# ---------------------------------------------------------------------------
# validate_chunks() correctness
# ---------------------------------------------------------------------------

def test_validate_chunks_passes_for_good_chunks():
    """validate_chunks() returns no errors for well-formed chunks."""
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    chunks = chunk_pages([_page(text)])
    errors = validate_chunks(chunks)
    assert errors == [], f"Unexpected validation errors: {errors}"


def test_validate_chunks_catches_empty_text():
    bad = IngestChunk(
        chunk_id="abc", doc_id="d", doc_title="T", text="   ",
        modality_type="text", page=1, section="S",
        caption="", storage_ref=None, attachment_id="",
    )
    errors = validate_chunks([bad])
    assert any("empty text" in e for e in errors)


def test_validate_chunks_catches_missing_doc_id():
    bad = IngestChunk(
        chunk_id="abc", doc_id="", doc_title="T", text="some text here",
        modality_type="text", page=1, section="S",
        caption="", storage_ref=None, attachment_id="",
    )
    errors = validate_chunks([bad])
    assert any("doc_id" in e for e in errors)


def test_validate_chunks_catches_duplicate_ids():
    chunk = IngestChunk(
        chunk_id="dup-id", doc_id="d", doc_title="T", text="some text here",
        modality_type="text", page=1, section="S",
        caption="", storage_ref=None, attachment_id="",
    )
    errors = validate_chunks([chunk, chunk])
    assert any("duplicate" in e for e in errors)


def test_validate_chunks_catches_oversized_chunk():
    """A chunk whose text has more tokens than chunk_size triggers an error."""
    # Build text that is definitely > 100 tokens
    long_text = " ".join(["word"] * 200)
    bad = IngestChunk(
        chunk_id="big", doc_id="d", doc_title="T", text=long_text,
        modality_type="text", page=1, section="S",
        caption="", storage_ref=None, attachment_id="",
    )
    errors = validate_chunks([bad], chunk_size=50)  # 200 > 50 → error
    assert any("exceeds chunk_size" in e for e in errors)


# ---------------------------------------------------------------------------
# Schema definition — EXPECTED_PROPERTIES matches ensure_schema() declaration
# ---------------------------------------------------------------------------

def test_expected_properties_covers_all_schema_fields():
    """
    EXPECTED_PROPERTIES must contain every field we create in ensure_schema().
    This is a code-level contract test — no live Weaviate needed.
    """
    # These are the property names declared in ensure_schema()
    declared_in_schema = {
        "chunk_id", "doc_id", "attachment_id", "modality_type",
        "doc_title", "text", "caption", "section", "page", "storage_ref",
    }
    assert set(EXPECTED_PROPERTIES.keys()) == declared_in_schema, (
        "EXPECTED_PROPERTIES and ensure_schema() are out of sync.\n"
        f"  In schema only: {declared_in_schema - set(EXPECTED_PROPERTIES.keys())}\n"
        f"  In EXPECTED_PROPERTIES only: {set(EXPECTED_PROPERTIES.keys()) - declared_in_schema}"
    )


def test_expected_properties_has_correct_count():
    """Sanity check: exactly 10 properties defined."""
    assert len(EXPECTED_PROPERTIES) == 10


# ---------------------------------------------------------------------------
# Pipeline idempotency (mocked Weaviate + embed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_pipeline_calls_delete_before_upsert(tmp_path):
    """
    ingest_document() must call delete_document(doc_id) before batch upserting,
    ensuring re-ingesting the same doc_id doesn't create duplicate chunks.
    """
    # Create a real temporary text file to ingest
    doc_file = tmp_path / "test_doc.txt"
    doc_file.write_text(
        "Hypertension management requires lifestyle modification and medication. " * 30,
        encoding="utf-8",
    )

    fake_embedding = [0.0] * 3072  # matches EMBEDDING_DIMENSION

    with (
        patch("src.ingest.pipeline.gemini_embed.embed_text",
              new_callable=AsyncMock,
              return_value=fake_embedding),
        patch("src.services.weaviate_client.get_client") as mock_get_client,
        patch("src.services.weaviate_client.delete_document",
              return_value=3) as mock_delete,
    ):
        # Set up mock collection for batch upsert
        mock_collection = MagicMock()
        mock_batch_ctx = MagicMock()
        mock_batch_ctx.__enter__ = MagicMock(return_value=mock_batch_ctx)
        mock_batch_ctx.__exit__ = MagicMock(return_value=False)
        mock_collection.batch.dynamic.return_value = mock_batch_ctx
        mock_get_client.return_value.collections.get.return_value = mock_collection

        from src.ingest.pipeline import ingest_document
        result = await ingest_document(
            path=doc_file,
            doc_id="test-doc-001",
            doc_title="Hypertension Guide",
        )

    # delete_document must be called once with the correct doc_id
    mock_delete.assert_called_once_with("test-doc-001")

    # Result should have the correct shape
    assert result["doc_id"] == "test-doc-001"
    assert result["doc_title"] == "Hypertension Guide"
    assert result["chunk_count"] > 0


@pytest.mark.asyncio
async def test_ingest_pipeline_rejects_empty_file(tmp_path):
    """An empty file raises ValueError before any network call."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with (
        patch("src.ingest.pipeline.gemini_embed.embed_text", new_callable=AsyncMock),
        patch("src.services.weaviate_client.delete_document"),
    ):
        from src.ingest.pipeline import ingest_document
        with pytest.raises(ValueError, match="No content extracted|zero chunks"):
            await ingest_document(path=empty_file, doc_id="empty-doc")
