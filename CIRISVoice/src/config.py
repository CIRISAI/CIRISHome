import os
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class STTConfig(BaseModel):
    provider: Literal["openai", "google"] = "google"
    api_key: str = Field(default_factory=lambda: os.getenv("STT_API_KEY", ""))
    language: str = "en-US"
    model: str = "whisper-1"  # For OpenAI
    google_credentials_path: Optional[str] = "/config/google_cloud_key.json"
    google_language_code: str = "en-US"


class TTSConfig(BaseModel):
    provider: Literal["openai", "google"] = "google"
    api_key: str = Field(default_factory=lambda: os.getenv("TTS_API_KEY", ""))
    voice: str = "en-US-Chirp3-HD-Achernar"
    model: str = "tts-1"
    speed: float = 1.0
    google_credentials_path: Optional[str] = "/config/google_cloud_key.json"
    google_voice_name: str = "en-US-Chirp3-HD-Achernar"


class CIRISConfig(BaseModel):
    api_url: str = "http://localhost:8080"
    api_key: Optional[str] = Field(default_factory=lambda: os.getenv("CIRIS_API_KEY"))
    timeout: int = 30
    channel_id: str = "home_assistant"
    profile: str = "home_assistant"


class WyomingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 10300
    name: str = "CIRIS Voice Assistant"


class Config(BaseModel):
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    ciris: CIRISConfig = CIRISConfig()
    wyoming: WyomingConfig = WyomingConfig()
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
