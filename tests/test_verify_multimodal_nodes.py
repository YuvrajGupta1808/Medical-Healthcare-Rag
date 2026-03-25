import sys
import os
import asyncio
from pathlib import Path

# Add local sandbox modules to solve dependency lockdowns
sys.path.append("/tmp/my_modules")
# Add root to pythonpath
sys.path.append(str(Path(__file__).parent.parent))

# Bypass pydantic Settings validation for mock runs
os.environ["FIREWORKS_API_KEY"] = "fake_key"
os.environ["GEMINI_API_KEY"] = "fake_key"

from src.router.input_router import InputRouter
from src.utils.config import get_settings

# Mocking OpenAI Client responses
class MockChoices:
    def __init__(self, content):
        self.message = type('obj', (object,), {'content': content})

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoices(content)]

class MockAudioResponse:
    def __init__(self, text):
        self.text = text

async def test_ocr_mock():
    print("=== Testing OCR Image Sub-Node (Mocked) ===")
    router = InputRouter()
    
    # 1. Mock client.chat.completions.create
    from unittest.mock import AsyncMock, patch
    
    mock_complete = AsyncMock(return_value=MockResponse("Mocked Medical OCR: Patient has normal blood pressure."))
    
    # Set fake API key in settings memory just for bypass skips
    s = get_settings()
    s.fireworks_api_key = "fake_key"

    with patch('openai.resources.chat.completions.AsyncCompletions.create', mock_complete):
        description = await router._stub_describe_image(b"fake_bytes", "image/png")
        print(f"OCR Description Result: {description}")
        if "normal blood pressure" in description:
             print("✅ OCR Node Flow Verified Successfully!")
        else:
             print("❌ OCR Node Flow Failed output matches.")

async def test_audio_mock():
    print("\n=== Testing Audio Transcribe Sub-Node (Mocked) ===")
    router = InputRouter()
    
    from unittest.mock import AsyncMock, patch
    
    mock_transcribe = AsyncMock(return_value=MockAudioResponse("Mocked Audio: Patient reports chest pain."))
    
    with patch('openai.resources.audio.transcriptions.AsyncTranscriptions.create', mock_transcribe):
        description = await router._stub_transcribe_audio(b"fake_audio_bytes", "audio/mp3")
        print(f"Audio Transcription Result: {description}")
        if "chest pain" in description:
             print("✅ Audio Node Flow Verified Successfully!")
        else:
             print("❌ Audio Node Flow Failed output matches.")

if __name__ == "__main__":
    asyncio.run(test_ocr_mock())
    asyncio.run(test_audio_mock())
