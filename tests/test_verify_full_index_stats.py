import sys
import os
from pathlib import Path

# Add local sandbox modules to solve dependency lockdowns
sys.path.append("/tmp/my_modules")
# Add root to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

# Set keys passed by user to bypass Pydantic errors
os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"

from src.services import weaviate_client

def check_index_stats():
    print("=== Checking Weaviate Full Index Stats ===")
    try:
        # Connect to Weaviate
        weaviate_client.connect()
        client = weaviate_client.get_client()
        collection = client.collections.get(weaviate_client.COLLECTION_NAME)
        
        # 1. Total Chunks Count
        response = collection.aggregate.over_all()
        total_objects = getattr(response, "total_count", 0)
        print(f"Total Chunks in Index: {total_objects}")
        
        if total_objects > 0:
            # 2. Breakdown by modality
            import weaviate.classes as wvc
            for m in ["text", "image", "audio", "video"]:
                count_resp = collection.aggregate.over_all(
                    filters=wvc.query.Filter.by_property("modality_type").equal(m),
                    total_count=True
                )
                print(f" - {m.title()} Chunks: {count_resp.total_count}")
                
            # 3. Pull top 5 recent chunks to see what's in there
            fetch_resp = collection.query.fetch_objects(limit=5, return_properties=["doc_title", "modality_type", "text"])
            print("\n--- Sample Chunks Setup ---")
            for obj in fetch_resp.objects:
                 p = obj.properties
                 txt_hint = p.get("text") or ""
                 print(f"- Doc: {p.get('doc_title')} | Type: {p.get('modality_type')} | Text: {repr(txt_hint[:60])}...")
        else:
             print("⚠️ The database index is completely EMPTY! No documents have been indexed.")

    except Exception as e:
         print(f"❌ Error during Weaviate diagnostics: {e}")
    finally:
         try:
             weaviate_client.close()
         except Exception:
             pass

if __name__ == "__main__":
    check_index_stats()
