import sys
import os
from pathlib import Path

sys.path.append("/tmp/my_modules")
sys.path.append(str(Path(__file__).parent.parent))

os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"

from src.services import weaviate_client

def check_doc():
    try:
        weaviate_client.connect()
        client = weaviate_client.get_client()
        collection = client.collections.get(weaviate_client.COLLECTION_NAME)
        
        # Query chunks for doc_id
        import weaviate.classes as wvc
        filters = wvc.query.Filter.by_property("doc_id").equal("6495cefd-e349-4b32-afc2-6d6e0eee7ec4")
        response = collection.query.fetch_objects(
            filters=filters,
            limit=50,
            return_properties=["chunk_id", "doc_id", "modality_type", "storage_ref", "text"]
        )
        
        objects = response.objects
        print(f"Total chunks found for doc: {len(objects)}")
        
        images = 0
        texts = 0
        for obj in objects:
            p = obj.properties
            m_type = p.get('modality_type')
            if m_type == "image":
                images += 1
                print(f"IMAGE CHUNK: id={p.get('chunk_id')}, storage_ref={p.get('storage_ref')}")
            elif m_type == "text":
                texts += 1
                
        print(f"Summary: {texts} Text Chunks, {images} Image Chunks.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
             weaviate_client.close()
        except:
             pass

check_doc()
