import asyncio
import sys
import os

ws = "/Users/Yuvraj/Medical-Healthcare-Rag"
site_packages = os.path.join(ws, ".venv", "lib", "python3.12", "site-packages")
sys.path.insert(0, "/tmp/pip-packages")
sys.path.insert(0, site_packages)
sys.path.insert(0, ws)

import src.utils.config

class MockSettings:
    cohere_api_key = ""
    cohere_model = "rerank-english-v3.0"
    rerank_top_k = 2

def mock_get_settings():
    return MockSettings()

src.utils.config.get_settings = mock_get_settings

# Now import the nodes using get_settings
from src.retrieval.rerank import rerank_node
from src.generation.citation_gate import citation_gate_node, _ABSTAIN_ANSWER
from src.pipeline.state import Chunk, Citation

async def test_rerank_fallback():
    print("=== Testing Rerank Node Fallback ===")
    # Simulate state with top-M chunks
    chunks = [
        {"chunk_id": "c1", "text": "Medical evidence 1", "score": 0.9, "doc_id": "d1", "doc_title": "T1", "modality_type": "text"},
        {"chunk_id": "c2", "text": "Medical evidence 2", "score": 0.8, "doc_id": "d1", "doc_title": "T1", "modality_type": "text"},
        {"chunk_id": "c3", "text": "Medical evidence 3", "score": 0.7, "doc_id": "d1", "doc_title": "T1", "modality_type": "text"},
    ]
    state = {
        "query_text": "What is the treatment?",
        "retrieved_chunks": chunks
    }
    
    # We shouldn't have cohere key set, so it should fall back to top-K reduction
    # Let's ensure top-k is 2 in config mock if we want slice verification
    result = await rerank_node(state)
    print(f"Reranked to {len(result.get('retrieved_chunks', []))} chunks.")
    for c in result.get('retrieved_chunks', []):
        print(f" - Chunk ID: {c.get('chunk_id')}, score: {c.get('score')}")
    print("SUCCESS: Rerank node ran without crashing.")

async def test_citation_gate_abstain():
    print("\n=== Testing Citation Gate Abstains ===")
    state = {
        "answer": "This answers the query with facts.",
        "citations": []  # Empty citations
    }
    result = await citation_gate_node(state)
    print(f"Result Answer: {result.get('answer')}")
    if result.get('answer') == _ABSTAIN_ANSWER:
         print("SUCCESS: Correctly abstained upon zero citations.")
    else:
         print("FAILURE: Did not abstain.")

async def test_citation_gate_passes():
    print("\n=== Testing Citation Gate Passes ===")
    state = {
        "answer": "This answers the query correctly.",
        "citations": [Citation(chunk_id="c1", doc_id="d1", quote="evidence", doc_title="T1")]
    }
    result = await citation_gate_node(state)
    print(f"Result Update: {result}")
    if result == {}:
         print("SUCCESS: Gate passed through valid answers.")
    else:
         print("FAILURE: Gate blocked valid answer.")

async def main():
    await test_rerank_fallback()
    await test_citation_gate_abstain()
    await test_citation_gate_passes()

if __name__ == "__main__":
    asyncio.run(main())
