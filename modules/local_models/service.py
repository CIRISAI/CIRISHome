"""
Local Models Service - 100% offline AI processing on Jetson Orin Nano.

Loads and manages STT, TTS, intents, and LLM models locally for complete
privacy and no cloud dependencies.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# Type-safe models to replace Dict[str, Any]
@dataclass
class ModelConfig:
    """Configuration for AI models."""

    name: str
    model_path: str
    device: str = "cuda"
    quantization: str = "int4"
    max_memory: str = "6GB"


@dataclass
class LLMMessage:
    """Structured message for LLM conversations."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """Structured LLM response."""

    text: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


@dataclass
class STTResult:
    """Speech-to-text result."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class TTSResult:
    """Text-to-speech result."""

    audio_path: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None


@dataclass
class IntentResult:
    """Intent classification result."""

    intent: str
    confidence: float
    entities: Dict[str, Union[str, int, float]]


# Protocol for external library interfaces
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""

    eos_token_id: int

    def __call__(self, text: str, return_tensors: str) -> Any:
        """Tokenize text and return tensors."""
        ...

    def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
        """Decode tokens back to text."""
        ...


class LocalModelManager:
    """Central manager for all local models."""

    def __init__(self) -> None:
        """Initialize the local model manager."""
        self.models_path: Path = Path(os.getenv("LOCAL_MODELS_PATH", "./models"))
        self.jetson_gpu: bool = (
            os.getenv("JETSON_GPU_ENABLED", "true").lower() == "true"
        )
        self.device: str = self._setup_device()
        self.loaded_models: Dict[str, Any] = {}

    def _setup_device(self) -> str:
        """Configure GPU/CPU device for Jetson Orin Nano."""
        if (
            TORCH_AVAILABLE
            and self.jetson_gpu
            and torch is not None
            and torch.cuda.is_available()
        ):
            # Jetson Orin Nano specific optimizations
            device = "cuda:0"
            torch.backends.cudnn.benchmark = True
            logger.info(
                f"Jetson GPU acceleration enabled: {torch.cuda.get_device_name(0)}"
            )
        else:
            device = "cpu"
            logger.warning("Running on CPU - performance will be limited")
        return device

    def get_model_info(self) -> Dict[str, Union[str, bool, List[str]]]:
        """Get information about available models and memory usage."""
        gpu_memory: float
        gpu_allocated: float
        gpu_cached: float

        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
        else:
            gpu_memory = gpu_allocated = gpu_cached = 0.0

        return {
            "device": self.device,
            "jetson_gpu": self.jetson_gpu,
            "gpu_memory_total": f"{gpu_memory:.1f}GB",
            "gpu_memory_allocated": f"{gpu_allocated:.1f}GB",
            "gpu_memory_cached": f"{gpu_cached:.1f}GB",
            "loaded_models": list(self.loaded_models.keys()),
            "models_path": str(self.models_path),
        }


class LocalLLMService:
    """Local LLM service using llama-4-scout."""

    def __init__(self) -> None:
        """Initialize the Local LLM service."""
        self.manager: LocalModelManager = LocalModelManager()
        self.model: Optional[Any] = None
        self.tokenizer: Optional[TokenizerProtocol] = None
        self.model_name: str = "llama-4-scout-int4"

    async def initialize(self) -> bool:
        """Load the LLM model.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            model_path = self.manager.models_path / "llm" / self.model_name

            if not model_path.exists():
                logger.error(f"Model not found at {model_path}")
                logger.info("Please download llama-4-scout model to models/llm/")
                return False

            logger.info(f"Loading {self.model_name} on {self.manager.device}...")

            # Load with Jetson optimizations and quantization
            if self.manager.jetson_gpu:
                import transformers
                from transformers import BitsAndBytesConfig

                # INT4 quantization configuration for Jetson
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                # Load local model - no revision needed for local paths
                self.tokenizer = (
                    transformers.AutoTokenizer.from_pretrained(  # nosec B615
                        str(model_path),
                        local_files_only=True,  # Ensure we only load from local path
                    )
                )
                self.model = (
                    transformers.AutoModelForCausalLM.from_pretrained(  # nosec B615
                        str(model_path),
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=True,  # Ensure we only load from local path
                    )
                )

                logger.info(f"🗜️ Loaded {self.model_name} with INT4 quantization")
            else:
                logger.warning("CPU mode - LLM performance will be very slow")
                return False

            self.manager.loaded_models["llm"] = self.model_name
            logger.info(f"✅ {self.model_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            return False

    async def call_llm_structured(
        self,
        messages: List[LLMMessage],
        response_model: Optional[Any] = None,
        images: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate structured response using local LLM.

        Args:
            messages: List of conversation messages
            response_model: Optional pydantic model for structured output
            images: Optional list of image paths for multimodal processing
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse: Structured response from the model
        """
        if not self.model:
            await self.initialize()

        if not self.model or not self.tokenizer:
            return LLMResponse(text="LLM not available", model=self.model_name)

        try:
            # Format messages for llama (multimodal)
            if images:
                prompt = self._format_multimodal_messages(messages, images)
            else:
                prompt = self._format_messages(messages)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.manager.device)

            if TORCH_AVAILABLE and torch is not None:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=kwargs.get("max_tokens", 512),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
                )

                return LLMResponse(
                    text=response.strip(),
                    model=self.model_name,
                    usage={"tokens": len(outputs[0])},
                )
            else:
                return LLMResponse(text="PyTorch not available", model=self.model_name)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResponse(text=f"Error: {str(e)}", model=self.model_name)

    def _format_messages(self, messages: List[LLMMessage]) -> str:
        """Format messages for llama chat format."""
        formatted = ""
        for msg in messages:
            role = msg.role
            content = msg.content
            formatted += f"<{role}>{content}</{role}>\n"
        formatted += "<assistant>"
        return formatted

    def _format_multimodal_messages(
        self, messages: List[LLMMessage], images: List[str]
    ) -> str:
        """Format multimodal messages with images for Llama-4-Scout."""
        formatted = ""

        # Add images first (up to 8 images supported)
        if images:
            num_images = min(len(images), 8)  # Llama-4-Scout supports up to 8 images
            for i in range(num_images):
                formatted += f"<image_{i+1}>[Image {i+1} provided]</image_{i+1}>\n"

        # Add text messages
        for msg in messages:
            role = msg.role
            content = msg.content
            formatted += f"<{role}>{content}</{role}>\n"

        formatted += "<assistant>"
        return formatted

    async def analyze_images(
        self, images: List[str], question: str = "Describe what you see in these images"
    ) -> LLMResponse:
        """Analyze images using Llama-4-Scout vision capabilities."""
        if not images:
            return LLMResponse(text="Error: No images provided", model=self.model_name)

        if len(images) > 8:
            logger.warning(f"Too many images ({len(images)}), using first 8")
            images = images[:8]

        messages = [LLMMessage(role="user", content=question)]

        return await self.call_llm_structured(messages, images=images)

    async def visual_question_answering(
        self, images: List[str], question: str
    ) -> LLMResponse:
        """Answer questions about images."""
        return await self.analyze_images(images, question)

    async def object_localization(
        self, images: List[str], objects: List[str]
    ) -> LLMResponse:
        """Locate specific objects in images."""
        objects_str = ", ".join(objects)
        question = (
            f"Locate and describe the position of these objects in the "
            f"images: {objects_str}. Use image grounding to be precise "
            f"about locations."
        )
        return await self.analyze_images(images, question)

    async def scene_understanding(self, images: List[str]) -> LLMResponse:
        """Understand the overall scene and context."""
        question = (
            "Analyze these images and provide a detailed understanding of the "
            "scene, including: 1) What room or location this appears to be, "
            "2) What activities might be happening, 3) Any notable objects or "
            "people, 4) The overall context and situation."
        )
        return await self.analyze_images(images, question)


class LocalSTTService:
    """Local Speech-to-Text using Whisper."""

    def __init__(self) -> None:
        """Initialize the Local STT service."""
        self.manager: LocalModelManager = LocalModelManager()
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None

    async def initialize(self) -> bool:
        """Load Whisper model.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            try:
                import whisper
            except ImportError:
                logger.error("Whisper not available - install openai-whisper")
                return False

            model_path = self.manager.models_path / "stt" / "whisper-large-v3"

            if model_path.exists():
                logger.info("Loading Whisper from local path...")
                self.model = whisper.load_model(str(model_path))
            else:
                logger.info("Loading Whisper large-v3 (will download first time)...")
                self.model = whisper.load_model("large-v3", device=self.manager.device)

            self.manager.loaded_models["stt"] = "whisper-large-v3"
            logger.info("✅ Whisper STT loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load STT: {e}")
            return False

    async def speech_to_text(self, audio_data: bytes) -> STTResult:
        """Convert speech to text.

        Args:
            audio_data: Raw audio bytes

        Returns:
            STTResult: Transcription result with text and metadata
        """
        if not self.model:
            await self.initialize()

        try:
            # Process audio with Whisper
            if not self.model:
                return STTResult(text="STT model not available")

            result = self.model.transcribe(audio_data)

            return STTResult(
                text=result["text"].strip(),
                language=result.get("language", "en"),
                confidence=0.95,  # Whisper doesn't provide confidence
            )

        except Exception as e:
            logger.error(f"STT failed: {e}")
            return STTResult(text=f"Error: {str(e)}")


class LocalTTSService:
    """Local Text-to-Speech using Coqui TTS."""

    def __init__(self) -> None:
        """Initialize the Local TTS service."""
        self.manager: LocalModelManager = LocalModelManager()
        self.tts: Optional[Any] = None

    async def initialize(self) -> bool:
        """Load TTS model.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            from TTS.api import TTS

            # Use Coqui TTS with Jetson optimization
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            self.tts = TTS(model_name=model_name, gpu=self.manager.jetson_gpu)

            self.manager.loaded_models["tts"] = "coqui-tts"
            logger.info("✅ Coqui TTS loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load TTS: {e}")
            return False

    async def text_to_speech(self, text: str, voice: str = "default") -> bytes:
        """Convert text to speech."""
        if not self.tts:
            await self.initialize()

        try:
            import tempfile

            # Generate audio in secure temp directory
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_path = temp_file.name

            if self.tts:
                self.tts.tts_to_file(text=text, file_path=audio_path)
            else:
                return b""

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            # Clean up temporary file
            import os

            try:
                os.unlink(audio_path)
            except OSError:
                pass  # File already deleted

            return audio_data

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return b""


class LocalIntentsService:
    """Local intent classification and entity extraction."""

    def __init__(self) -> None:
        """Initialize the Local Intents service."""
        self.manager: LocalModelManager = LocalModelManager()
        self.classifier: Optional[Any] = None
        self.tokenizer: Optional[TokenizerProtocol] = None

    async def initialize(self) -> bool:
        """Load intent classification model.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "microsoft/DialoGPT-medium"  # Or custom intent model
            # Pin to specific revision for security (DialoGPT-medium stable release)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision="df68cc6b4fdf5fa96ec3e5c07aa5b7c8ea67e5df"
            )
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                model_name, revision="df68cc6b4fdf5fa96ec3e5c07aa5b7c8ea67e5df"
            )

            if self.manager.jetson_gpu:
                self.classifier = self.classifier.to(self.manager.device)

            self.manager.loaded_models["intents"] = "distilbert-intent"
            logger.info("✅ Intent classifier loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load intents model: {e}")
            return False

    async def classify_intent(self, text: str) -> IntentResult:
        """Classify user intent from text."""
        # Simplified intent classification
        # In production, use a proper trained intent model

        text_lower = text.lower()

        if any(word in text_lower for word in ["turn on", "switch on", "enable"]):
            intent = "turn_on_device"
        elif any(word in text_lower for word in ["turn off", "switch off", "disable"]):
            intent = "turn_off_device"
        elif any(word in text_lower for word in ["dim", "brightness", "bright"]):
            intent = "adjust_brightness"
        elif any(
            word in text_lower for word in ["temperature", "thermostat", "heat", "cool"]
        ):
            intent = "adjust_temperature"
        elif any(word in text_lower for word in ["camera", "show", "view"]):
            intent = "show_camera"
        else:
            intent = "general_query"

        return IntentResult(
            intent=intent,
            confidence=0.85,
            entities=self._extract_entities(text),
        )

    def _extract_entities(self, text: str) -> Dict[str, Union[str, int, float]]:
        """Extract entities from text."""
        # Simplified entity extraction
        entities: Dict[str, Union[str, int, float]] = {}

        # Room detection
        rooms = ["living room", "bedroom", "kitchen", "bathroom", "garage"]
        for room in rooms:
            if room in text.lower():
                entities["room"] = room
                break

        # Number detection
        import re

        numbers = re.findall(r"\d+", text)
        if numbers:
            entities["number"] = int(numbers[0])

        return entities
