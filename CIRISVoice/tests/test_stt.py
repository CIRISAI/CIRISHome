from src.config import STTConfig
from src.stt_service import create_stt_service


def test_create_stt_service_openai():
    cfg = STTConfig(provider="openai", api_key="k")
    service = create_stt_service(cfg)
    assert service.__class__.__name__ == "OpenAISTTService"
