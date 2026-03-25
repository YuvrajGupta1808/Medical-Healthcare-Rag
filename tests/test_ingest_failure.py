import sys, os, asyncio
from pathlib import Path
sys.path.append("/tmp/my_modules")
sys.path.append(str(Path(__file__).parent.parent))
os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"

from src.ingest.pipeline import ingest_document

async def test():
    with open("mini.txt", "w") as f:
        f.write("Line 1\nLine 2")
    try:
        from src.services import weaviate_client
        weaviate_client.connect()
        res = await ingest_document(Path("mini.txt"), "Test")
        print(res)
    except Exception as e:
        print("Failure:", repr(e))

asyncio.run(test())
