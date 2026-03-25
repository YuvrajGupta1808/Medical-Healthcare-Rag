import sys
from unittest.mock import MagicMock, patch

# Mock modules that require heavy pip installs not present in global py
sys.modules['boto3'] = MagicMock()
sys.modules['botocore'] = MagicMock()
sys.modules['botocore.client'] = MagicMock()
sys.modules['botocore.exceptions'] = MagicMock()

import asyncio
from src.router.output_router import output_route_node
from src.pipeline.state import Chunk

async def test_output_route_node_image():
    from unittest.mock import patch
    import src.services.storage
    
    state = {
        "prefer_retrieved_image": True,
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "t1",
                "text": "sample text",
                "modality_type": "image",
                "page": 1,
                "section": "s1",
                "caption": "c",
                "score": 0.8,
                "storage_ref": "bucket/images/key.jpg"
            }
        ]
    }
    
    with patch("src.services.storage.generate_signed_url", return_value="http://signed_url"):
        result = await output_route_node(state)
        assert result["image_url"] == "http://signed_url"
        assert result["confidence_score"] == 0.8
        assert result["low_confidence"] is None

async def test_output_route_node_pdf():
    state = {
        "include_pdf_export": True,
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "doc_title": "t1",
                "text": "sample text",
                "modality_type": "pdf",
                "page": 1,
                "section": "s1",
                "caption": "c",
                "score": 0.7,
                "storage_ref": "http://pdf_url"
            }
        ]
    }
    result = await output_route_node(state)
    assert result["pdf_url"] == "http://pdf_url"
    assert result["confidence_score"] == 0.7

async def test_output_route_node_low_confidence():
    state = {
        "retrieved_chunks": [
            {"score": 0.4},
            {"score": 0.5}
        ]
    }
    result = await output_route_node(state)
    assert result["confidence_score"] == 0.45
    assert result["low_confidence"] is True

if __name__ == "__main__":
    async def run_all():
        print("Running test_output_route_node_image...")
        await test_output_route_node_image()
        print("Running test_output_route_node_pdf...")
        await test_output_route_node_pdf()
        print("Running test_output_route_node_low_confidence...")
        await test_output_route_node_low_confidence()
        print("All tests passed!")

    asyncio.run(run_all())
