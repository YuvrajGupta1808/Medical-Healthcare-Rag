import sys
import os
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from src.router.input_router import InputRouter
from src.schemas.query import Modality

# We need a proper FastAPI UploadFile mock for testing router.route()
from fastapi import UploadFile
import io

# Mock UploadFile that works in pydantic / router
class MockUploadFile(UploadFile):
    def __init__(self, filename, content, content_type="image/jpeg"):
         # FastAPI 0.110+ requires passing file object
         super().__init__(filename=filename, file=io.BytesIO(content), size=len(content), headers=None)
         self.content_type = content_type

async def verify_stubs():
    print("=== Verifying Fireworks AI Stubs in Input Router ===")
    
    # 1. Test Image Description Stub
    print("\n--- 1. Testing _stub_describe_image ---")
    router = InputRouter()
    
    # Simple valid 1x1 transparent GIF to ensure vision parser makes it to LLM
    # (keeps payload safe from corrupt corrupt headers)
    gif_1x1 = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    
    # Call stub directly first
    desc = await router._stub_describe_image(gif_1x1, "image/gif")
    print(f"Vision Response: {desc}")
    
    if "Error" not in desc and "skipped" not in desc:
         print("✅ Vision stub hit Fireworks successfully!")
    else:
         print("⚠️ Fireworks image vision call bypassed/failed, see response above.")

    # 2. Test Audio Transcription Stub
    print("\n--- 2. Testing _stub_transcribe_audio ---")
    try:
        # Just passing 100 bytes of silence / garbage just to see if it reaches the API
        audio_dummy = b"\x00" * 100
        transcript = await router._stub_transcribe_audio(audio_dummy, "audio/mp3")
        print(f"Audio Response: {transcript}")
        if "Error" not in transcript and "skipped" not in transcript:
             print("✅ Audio stub hit Fireworks successfully!")
        else:
             print("⚠️ Fireworks audio check returned error or bypass, see response above.")
    except Exception as e:
        print(f"❌ Audio stub crashed: {e}")

if __name__ == "__main__":
    # Load .env variables since config pulls from there
    from src.utils.config import get_settings
    # Trigger setting load 
    s = get_settings()
    # verify its populated
    if not s.fireworks_api_key:
        print("⚠️ Warning: FIREWORKS_API_KEY is blank in settings config.")

    asyncio.run(verify_stubs())
