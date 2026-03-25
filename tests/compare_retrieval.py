import os
import asyncio
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force connection to real services for this comparison script
os.environ["TESTING"] = "false"

# Ensure API Keys are present (from .env if not set, or placeholders if not connecting for real generation)
# But retrieval needs Google API Key for embeddings!
# Let's see if there's any API keys we can salvage or if .env loaded correctly.
from dotenv import load_dotenv
load_dotenv()

from src.services import weaviate_client
from src.retrieval.dense import retrieve_node as dense_retrieve
from src.retrieval.hybrid import retrieve_node as hybrid_retrieve
from src.pipeline.state import RAGState

async def compare_queries():
    print("Connecting to Weaviate...")
    weaviate_client.connect()
    
    queries = [
        "acute suppurative appendicitis",  # Exact
        "laparoscopic appendectomy",     # Exact
        "How was Emily's condition treated?", # Semantic
        "What are the postoperative instructions?" # Semantic
    ]
    
    print("\n--- RETRIEVAL COMPARISON REPORT ---")
    
    for query in queries:
        print(f"\n=================================================================")
        print(f"QUERY: '{query}'")
        print(f"=================================================================")
        
        state: RAGState = {"query_text": query}
        
        # Dense
        print("\n[DENSE SEARCH]")
        try:
            dense_res = await dense_retrieve(state)
            chunks = dense_res.get("retrieved_chunks", [])
            print(f"Found {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2], 1):
                score = chunk.get('score', 0.0)
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                print(f" {i}. Score: {score_str} | Section: {chunk['section']} (Page {chunk['page'] or '?'})")
                print(f"    Text: {chunk['text'][:150].replace('\n', ' ')}...")
        except Exception as e:
            print(f"Dense failed: {e}")
            
        # Hybrid
        print("\n[HYBRID SEARCH]")
        try:
            hybrid_res = await hybrid_retrieve(state)
            chunks = hybrid_res.get("retrieved_chunks", [])
            print(f"Found {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2], 1):
                score = chunk.get('score', 0.0)
                score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                print(f" {i}. Score: {score_str} | Section: {chunk['section']} (Page {chunk['page'] or '?'})")
                print(f"    Text: {chunk['text'][:150].replace('\n', ' ')}...")
        except Exception as e:
            print(f"Hybrid failed: {e}")

    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(compare_queries())
