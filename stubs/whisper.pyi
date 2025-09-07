"""Type stubs for OpenAI Whisper."""

from typing import Any, Dict, Optional, Union

import numpy as np

class WhisperModel:
    def transcribe(
        self, audio: Union[str, np.ndarray[Any, Any]], **kwargs: Any
    ) -> Dict[str, Any]: ...

def load_model(
    name: str,
    device: Optional[str] = None,
    download_root: Optional[str] = None,
    in_memory: bool = False,
) -> WhisperModel: ...
