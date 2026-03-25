import sys
import os
import asyncio
from pathlib import Path

sys.path.append("/tmp/my_modules")
sys.path.append(str(Path(__file__).parent.parent))

os.environ["FIREWORKS_API_KEY"] = "fw_USPQyCvhmnxJdrRNFhXCKM"
os.environ["GEMINI_API_KEY"] = "AIzaSyDIghApxHTO6rewl-EsWB0zPUycASx9wuM"

import httpx
from src.utils.config import get_settings

async def test_rerank():
    settings = get_settings()
    model = settings.fireworks_rerank_model
    print(f"Testing Rerank Model: {model}")
    
    headers = {
        "Authorization": f"Bearer {settings.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": "Follow-up schedule",
        "documents": ["The follow up schedule is 1 week", "Patient was discharged."],
        "model": model,
        "top_n": 2,
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.fireworks_base_url}/rerank",
                headers=headers,
                json=payload
            )
            print(f"Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error Body: {response.text}")
            else:
                print("Success!")
                print(response.json())
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_rerank())
