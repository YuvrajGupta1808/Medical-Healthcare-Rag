import sys
import os
import asyncio
from pathlib import Path

# Add local sandbox modules to solve dependency lockdowns
sys.path.append("/tmp/my_modules")
# Add root to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

# Set keys passed by user to bypass Pydantic errors
os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"

# Also set Minio just in case it triggers on mock setups
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"

from src.ingest.pipeline import ingest_document
from src.services import weaviate_client

async def test_ingest_works():
    print("=== Testing Live Ingestion Workflow ===")
    
    # Create a dummy text file with enough content for chunking
    test_file = Path("/tmp/test_medical_doc.txt")
    with open(test_file, "w") as f:
        f.write("\n".join([
            "Hypertension is defined as a systolic blood pressure (SBP) of 130 mmHg or more, or a diastolic blood pressure (DBP) of 80 mmHg or more.",
            "It is a major risk factor for cardiovascular disease. Management includes lifestyle changes such as diet and exercise.",
            "Weight loss of approximately 1 kg is associated with a 1 mmHg reduction in SBP.",
            "Pharmacological therapies include diuretics, beta-blockers, ACE inhibitors, and calcium channel blockers.",
            "Regular monitoring is advised to avoid hypertensive crises and associated end-organ damage such as renal failure."
        ] * 10)) # Repeat to make it long

    try:
        # Connect first
        weaviate_client.connect()
        weaviate_client.ensure_schema()
        
        result = await ingest_document(test_file, doc_title="Diagnostics Test Doc")
        print(f"✅ Ingest Success Result: {result}")
        
    except Exception as e:
        print(f"❌ Ingestion Failed during execution: {e}")
    finally:
         try:
             weaviate_client.close()
         except Exception:
             pass

if __name__ == "__main__":
    asyncio.run(test_ingest_works())
