import sys, os, asyncio
from pathlib import Path
sys.path.append("/tmp/my_modules")
sys.path.append(str(Path(__file__).parent.parent))
os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"

from src.ingest.pipeline import ingest_document
from src.services import weaviate_client

async def test():
    try:
        weaviate_client.connect()
        weaviate_client.ensure_schema()
        res = await ingest_document(Path("test.pdf"), "Test")
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Failure:", repr(e))

asyncio.run(test())
