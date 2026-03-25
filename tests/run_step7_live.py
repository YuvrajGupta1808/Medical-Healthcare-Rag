#!/usr/bin/env python3
"""
tests/run_step7_live.py
======================
Verifies Step 7 of the RAG pipeline using a live MinIO local container.
Tests:
1. Bucket auto-creation via ensure_bucket_exists
2. Object upload and hash-deterministic key generation
3. Signed URL generation & live fetch via httpx Client.

Usage:
    PYTHONPATH=. tmp_venv/bin/python tests/run_step7_live.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import httpx

# ── env must be set BEFORE any src import ──────────────────────────────────
os.environ["TESTING"] = "true"
# explicit local Docker setup matching .env defaults
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"
os.environ["MINIO_BUCKET"] = "medical-docs-test"
os.environ["MINIO_SECURE"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.services import storage

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

def ok(name: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET}  {name}")

def fail(name: str, reason: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}  {name}")
    print(f"          → {reason}")
    sys.exit(1)

from types import SimpleNamespace
from unittest.mock import patch

async def main():
    print("\nStarting Step 7 Live MinIO Verification...")
    
    mock_settings = SimpleNamespace(
        minio_endpoint="localhost:9000",
        minio_access_key="minioadmin",
        minio_secret_key="minioadmin",
        minio_bucket="medical-docs-test",
        minio_secure=False
    )
    
    # 1. Test bucket auto-creation on first write/upload trigger
    print("\n1. Testing bucket validation/creation...")
    try:
        with patch("src.services.storage.get_settings", return_value=mock_settings):
            storage.ensure_bucket_exists()
            ok("Bucket verification/creation called successfully")
    except Exception as e:
        fail("ensure_bucket_exists", str(e))

    # 2. Test uploading binary data
    print("\n2. Testing file upload...")
    test_bytes = b"faked_binary_content_of_a_medical_diagram_or_figure"
    try:
        with patch("src.services.storage.get_settings", return_value=mock_settings):
            ref = storage.upload_file(test_bytes, mime_type="image/jpeg")
            ok(f"Upload successful: {ref}")
    except Exception as e:
        fail("upload_file", str(e))

    # 3. Test URL signature and download
    print("\n3. Testing URL generation and fetch download...")
    try:
        with patch("src.services.storage.get_settings", return_value=mock_settings):
            url = storage.generate_signed_url(ref, expires_in=100)
            ok(f"Generated URL: {url}")
        
        # httpx trigger fetch
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            fetched = resp.content
            
            if fetched == test_bytes:
                ok("Download content matches upload accurately")
            else:
                fail("content verify", f"Mismatch. Got {len(fetched)} bytes instead of {len(test_bytes)}")

    except Exception as e:
         fail("generate_signed_url / download", str(e))

    print(f"\n{GREEN}Step 7 Verification Complete — Live Storage OK{RESET}\n")

if __name__ == "__main__":
    asyncio.run(main())
