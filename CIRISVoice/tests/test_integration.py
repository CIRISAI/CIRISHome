from unittest.mock import AsyncMock

import pytest
from src.bridge import CIRISWyomingHandler
from src.config import Config
from wyoming.asr import Transcript
from wyoming.tts import Synthesize


@pytest.fixture
def mock_config():
    return Config(
        stt={"provider": "openai", "api_key": "test"},
        tts={"provider": "openai", "api_key": "test"},
        ciris={"api_url": "http://localhost:8080"},
    )


@pytest.mark.asyncio
async def test_voice_to_ciris_flow(mock_config):
    handler = CIRISWyomingHandler(mock_config)
    handler.stt_service.transcribe = AsyncMock(return_value="Hello CIRIS")
    handler.ciris_client.send_message = AsyncMock(
        return_value={"content": "Hello! I'm CIRIS, your ethical AI assistant."}
    )
    handler.tts_service.synthesize = AsyncMock(return_value=b"audio_data")
    from wyoming.audio import AudioChunk, AudioStart, AudioStop

    await handler.handle_event(AudioStart())
    await handler.handle_event(AudioChunk(audio=b"fake_audio_data"))
    events = await handler.handle_event(AudioStop())
    assert len(events) == 2
    assert isinstance(events[0], Transcript)
    assert events[0].text == "Hello CIRIS"
    assert isinstance(events[1], Synthesize)
    assert "Hello! I'm CIRIS" in events[1].text


@pytest.mark.asyncio
async def test_direct_text_input(mock_config):
    handler = CIRISWyomingHandler(mock_config)
    handler.ciris_client.send_message = AsyncMock(return_value={"content": "I understand your request."})
    event = await handler.handle_event(Transcript(text="What is the weather?"))
    assert isinstance(event, Synthesize)
    assert event.text == "I understand your request."
