import sys
import os
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent))

# Bypass production API key validation for this test
os.environ["TESTING"] = "true"

import asyncio
from src.pipeline.state import Chunk, RAGState
from src.router.output_router import output_route_node
from src.utils.config import get_settings

async def verify_output_router():
    print("=== Verifying Output Router Confidence Threshold ===")

    # 1. Create mock chunks with scores
    # Weaviate hybrid scores are rank-fusion values, typical range below 1.0
    mock_chunks = [
        {"chunk_id": "1", "score": 0.55, "text": "evidence 1", "modality_type": "text"},
        {"chunk_id": "2", "score": 0.52, "text": "evidence 2", "modality_type": "text"},
    ]

    # Assemble state
    state: RAGState = {
        "query_text": "testing confidence threshold",
        "retrieved_chunks": mock_chunks,
    }

    print("\n--- 1. Testing with default threshold (0.3) ---")
    # Make sure we read from environment if loaded
    settings = get_settings()
    print(f"Current Config confidence_threshold: {getattr(settings, 'confidence_threshold', 'NOT_SET')}")
    
    # Run node
    result = await output_route_node(state)
    
    print(f"Confidence Score Average: {result.get('confidence_score')}")
    print(f"Low Confidence Flag:     {result.get('low_confidence')}")

    if result.get("confidence_score") == 0.535:  # (0.55 + 0.52) / 2
        print("✅ Average score calculated correctly.")
    else:
        print("❌ Average score calculation mismatch.")

    # Since 0.535 > 0.3, it should NOT be low confidence
    if result.get("low_confidence") is None or result.get("low_confidence") is False:
        print("✅ Correct! 0.535 is above 0.3, low_confidence is NOT flagging.")
    else:
        print("❌ Error! 0.535 should NOT flag low_confidence above 0.3.")

    print("\n--- 2. Testing with forced higher threshold via setting Override ---")
    # Temporarily override settings if possible during execute (Settings is lru_cache)
    settings.confidence_threshold = 0.6
    print(f"Overridden Config confidence_threshold: {settings.confidence_threshold}")
    
    result_high = await output_route_node(state)
    print(f"Confidence Score Average: {result_high.get('confidence_score')}")
    print(f"Low Confidence Flag:     {result_high.get('low_confidence')}")

    # Since 0.535 < 0.6, it SHOULD activate low_confidence
    if result_high.get("low_confidence") is True:
        print("✅ Correct! 0.535 is below 0.6 threshold, low_confidence correctly triggering.")
    else:
        print("❌ Error! 0.535 should trigger low_confidence under 0.6 threshold.")

if __name__ == "__main__":
    # Ensure event loop
    asyncio.run(verify_output_router())
