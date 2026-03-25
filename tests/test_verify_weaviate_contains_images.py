import sys
import os
from pathlib import Path

# Add local sandbox modules to solve dependency lockdowns
sys.path.append("/tmp/my_modules")
# Add root to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

# Bypass pydantic Settings validation for mock runs
os.environ["FIREWORKS_API_KEY"] = "fake_key"
os.environ["GEMINI_API_KEY"] = "fake_key"

from src.services import weaviate_client

def check_weaviate_images():
    print("=== Checking Weaviate for Image Chunks ===")
    try:
        # Connect to Weaviate
        weaviate_client.connect()
        client = weaviate_client.get_client()
        collection = client.collections.get(weaviate_client.COLLECTION_NAME)
        
        # Query for modality_type == "image"
        import weaviate.classes as wvc
        response = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("modality_type").equal("image"),
            limit=10,
            return_properties=["chunk_id", "doc_id", "doc_title", "modality_type"]
        )
        
        print(f"Found {len(response.objects)} image chunks.")
        for obj in response.objects:
            p = obj.properties
            print(f"- Doc: {p.get('doc_title')} ({p.get('doc_id')}) | Chunk: {p.get('chunk_id')}")

        if not response.objects:
             print("⚠️ No image chunks found in Weaviate! Typical cause: PDFs ingested before image extraction was added, or PDFs had no images.")
             
    except Exception as e:
        print(f"❌ Error querying Weaviate: {e}")
    finally:
        try:
            weaviate_client.close()
        except Exception:
            pass

if __name__ == "__main__":
    check_weaviate_images()
