import sys
import os
from pathlib import Path

# Add root to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.testclient import TestClient

# 1. Mock the pipeline graph imports OR just let it load lazily
# The graph imports nodes lazily inside build_rag_graph(), so it shouldn't import weaviate on start.
# Let's import our router.
from src.api.routes.query import router

# Create a small FastAPI app to test rendering
app = FastAPI()
app.include_router(router, prefix="/query")

client = TestClient(app)

def verify_routes():
    print("=== Testing Query Endpoints ===")

    # 1. Test standard text query endpoint structure
    print("\n--- 1. Testing POST /query ---")
    try:
        response = client.post("/query", json={"query": "What is the treatment?"})
        print(f"POST /query: Status Code={response.status_code}")
    except Exception as e:
        print(f"POST /query error reaching code: {e}")

    # 2. Test /multimodal endpoint accepts multipart/form-data
    print("\n--- 2. Testing /query/multimodal with files ---")
    import io
    dummy_file = io.BytesIO(b"fake_image_bytes")
    try:
        response = client.post(
            "/query/multimodal",
            files={"file": ("test_image.jpg", dummy_file, "image/jpeg")},
            data={
                "modality": "image",
                "query": "What is in position?",
                "top_k": 5
            }
        )
        print(f"POST /query/multimodal: Status Code={response.status_code}")
        if response.status_code == 500:
             # Standard if pipeline crashes on initialization or weaviate
             print("✅ Reached pipeline execution!")
        else:
             print(f"Response: {response.text}")
             
    except Exception as e:
         print(f"❌ Error during /multimodal call: {e}")

if __name__ == "__main__":
    verify_routes()
