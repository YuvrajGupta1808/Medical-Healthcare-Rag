import os
from pathlib import Path

import tiktoken

from src.ingest.chunking import chunk_pages
from src.ingest.loaders import load_document


def test_standalone_image_loader():
    """Verify that standalone images load into image modal RawPages."""
    dummy_img = Path("/tmp/dummy_image.png")
    # create dummy non-empty file header
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    try:
        pages = load_document(dummy_img, doc_id="img-001", doc_title="Dummy Image")
        assert len(pages) == 1
        page = pages[0]
        assert page.modality_type == "image"
        assert page.image_bytes is not None
        assert page.doc_id == "img-001"
    finally:
        if dummy_img.exists():
            dummy_img.unlink()


def test_standalone_image_chunking():
    """Verify that image RawPages turn into image IngestChunks."""
    # Build text that is definitely empty to assert bypass of empty-text rule
    from src.ingest.loaders import RawPage
    page = RawPage(
        text="",
        page_number=1,
        doc_id="img-001",
        doc_title="Image Doc",
        modality_type="image",
        image_bytes=b"\x89PNG..."
    )

    chunks = chunk_pages([page])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.modality_type == "image"
    assert chunk.image_bytes == b"\x89PNG..."
    assert chunk.caption == "Image from Image Doc (Page 1)"
