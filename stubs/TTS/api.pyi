"""Type stubs for Coqui TTS API."""

from typing import Any, List, Optional, Union

class TTS:
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        vocoder_config_path: Optional[str] = None,
        progress_bar: bool = True,
        gpu: bool = False,
        **kwargs: Any
    ): ...
    def tts_to_file(
        self,
        text: str,
        file_path: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> str: ...
    def tts(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        emotion: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs: Any
    ) -> Any: ...
