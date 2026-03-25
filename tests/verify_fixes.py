import sys
import os
from pathlib import Path

# Add src to pythonpath
sys.path.append(str(Path(__file__).parent))

from src.ingest.chunking import chunk_pages, validate_chunks, IngestChunk
from src.ingest.loaders import RawPage

def verify_image_chunk_fix():
    print("=== Verifying Image Chunk and Validation Fix ===")

    # 1. Simulate an image RawPage (from PDF or standalone)
    image_page = RawPage(
        text="",  # Empty text, as loaded for images
        page_number=3,
        section="Fig 1. MRI Scan",
        doc_id="doc-img-001",
        doc_title="Brain MRI Study",
        modality_type="image",
        image_bytes=b"fake_image_bytes_xyz"
    )

    print("\n--- 1. Testing Chunking on Image Page ---")
    chunks = chunk_pages([image_page])
    print(f"Produced {len(chunks)} chunks")

    if chunks:
        chunk = chunks[0]
        print(f"  Chunk Detail:")
        print(f"    ID:            {chunk.chunk_id}")
        print(f"    Modality:      {chunk.modality_type}")
        print(f"    Text:          {repr(chunk.text)}")
        print(f"    Caption:       {chunk.caption}")

    print("\n--- 2. Validating Image Chunks (Should pass now) ---")
    errors = validate_chunks(chunks)
    if not errors:
        print("✅ Success! Image chunks with empty text pass validation.")
    else:
        print(f"❌ Failed! Validation returned errors: {errors}")

    print("\n--- 3. Testing Token Range Logging Fix ---")
    # Verify that pipeline doesn't crash on pipeline-range logging logic
    import tiktoken
    from src.ingest.chunking import _ENCODING
    enc = tiktoken.get_encoding(_ENCODING)
    
    # Simulate list comprehension from pipeline.py
    token_counts = [len(enc.encode(c.text)) for c in chunks if c.modality_type != "image"]
    print(f"Token counts for non-image chunks: {token_counts}")
    print(f"Min token count: {min(token_counts) if token_counts else 0}")
    print(f"Max token count: {max(token_counts) if token_counts else 0}")
    print("✅ Token counting logic verified without crashing on image empty strings!")

if __name__ == "__main__":
    verify_image_chunk_fix()
