from src.config import TTSConfig
from src.tts_service import create_tts_service


def test_create_tts_service_openai():
    cfg = TTSConfig(provider="openai", api_key="k")
    service = create_tts_service(cfg)
    assert service.__class__.__name__ == "OpenAITTSService"
