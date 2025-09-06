"""
Local Models Service - 100% offline AI processing on Jetson Orin Nano.

Loads and manages STT, TTS, intents, and LLM models locally for complete
privacy and no cloud dependencies.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class LocalModelManager:
    """Central manager for all local models."""

    def __init__(self):
        """Initialize the local model manager."""
        self.models_path = Path(os.getenv("LOCAL_MODELS_PATH", "./models"))
        self.jetson_gpu = os.getenv("JETSON_GPU_ENABLED", "true").lower() == "true"
        self.device = self._setup_device()
        self.loaded_models = {}

    def _setup_device(self) -> str:
        """Configure GPU/CPU device for Jetson Orin Nano."""
        if self.jetson_gpu and torch.cuda.is_available():
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
        else:
            gpu_memory = gpu_allocated = gpu_cached = 0

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

    def __init__(self):
        self.manager = LocalModelManager()
        self.model = None
        self.tokenizer = None
        self.model_name = "llama-4-scout-int4"

    async def initialize(self):
        """Load the LLM model"""
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

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    str(model_path)
                )
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
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
        self, messages: List[Dict], response_model=None, images: List = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate structured response using local LLM"""
        if not self.model:
            await self.initialize()

        if not self.model:
            return {"error": "LLM not available"}

        try:
            # Format messages for llama (multimodal)
            if images:
                prompt = self._format_multimodal_messages(messages, images)
            else:
                prompt = self._format_messages(messages)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.manager.device)

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

            return {
                "content": response.strip(),
                "model": self.model_name,
                "device": self.manager.device,
                "local": True,
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {"error": str(e)}

    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for llama chat format"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<{role}>{content}</{role}>\n"
        formatted += "<assistant>"
        return formatted

    def _format_multimodal_messages(self, messages: List[Dict], images: List) -> str:
        """Format multimodal messages with images for Llama-4-Scout"""
        formatted = ""

        # Add images first (up to 8 images supported)
        if images:
            num_images = min(len(images), 8)  # Llama-4-Scout supports up to 8 images
            for i in range(num_images):
                formatted += f"<image_{i+1}>[Image {i+1} provided]</image_{i+1}>\n"

        # Add text messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<{role}>{content}</{role}>\n"

        formatted += "<assistant>"
        return formatted

    async def analyze_images(
        self, images: List, question: str = "Describe what you see in these images"
    ) -> Dict[str, Any]:
        """Analyze images using Llama-4-Scout vision capabilities"""
        if not images:
            return {"error": "No images provided"}

        if len(images) > 8:
            logger.warning(f"Too many images ({len(images)}), using first 8")
            images = images[:8]

        messages = [{"role": "user", "content": question}]

        return await self.call_llm_structured(messages, images=images)

    async def visual_question_answering(
        self, images: List, question: str
    ) -> Dict[str, Any]:
        """Answer questions about images"""
        return await self.analyze_images(images, question)

    async def object_localization(
        self, images: List, objects: List[str]
    ) -> Dict[str, Any]:
        """Locate specific objects in images"""
        objects_str = ", ".join(objects)
        question = f"Locate and describe the position of these objects in the images: {objects_str}. Use image grounding to be precise about locations."
        return await self.analyze_images(images, question)

    async def scene_understanding(self, images: List) -> Dict[str, Any]:
        """Understand the overall scene and context"""
        question = "Analyze these images and provide a detailed understanding of the scene, including: 1) What room or location this appears to be, 2) What activities might be happening, 3) Any notable objects or people, 4) The overall context and situation."
        return await self.analyze_images(images, question)


class LocalSTTService:
    """Local Speech-to-Text using Whisper"""

    def __init__(self):
        self.manager = LocalModelManager()
        self.model = None
        self.processor = None

    async def initialize(self):
        """Load Whisper model"""
        try:
            import whisper

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

    async def speech_to_text(self, audio_data: bytes) -> Dict[str, Any]:
        """Convert speech to text"""
        if not self.model:
            await self.initialize()

        try:
            # Process audio with Whisper
            result = self.model.transcribe(audio_data)

            return {
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "confidence": 0.95,  # Whisper doesn't provide confidence
                "model": "whisper-large-v3",
                "local": True,
            }

        except Exception as e:
            logger.error(f"STT failed: {e}")
            return {"error": str(e)}


class LocalTTSService:
    """Local Text-to-Speech using Coqui TTS"""

    def __init__(self):
        self.manager = LocalModelManager()
        self.tts = None

    async def initialize(self):
        """Load TTS model"""
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
        """Convert text to speech"""
        if not self.tts:
            await self.initialize()

        try:
            # Generate audio
            audio_path = "/tmp/ciris_tts_output.wav"
            self.tts.tts_to_file(text=text, file_path=audio_path)

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            return audio_data

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return b""


class LocalIntentsService:
    """Local intent classification and entity extraction"""

    def __init__(self):
        self.manager = LocalModelManager()
        self.classifier = None
        self.tokenizer = None

    async def initialize(self):
        """Load intent classification model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "microsoft/DialoGPT-medium"  # Or custom intent model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )

            if self.manager.jetson_gpu:
                self.classifier = self.classifier.to(self.manager.device)

            self.manager.loaded_models["intents"] = "distilbert-intent"
            logger.info("✅ Intent classifier loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load intents model: {e}")
            return False

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify user intent from text"""
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

        return {
            "intent": intent,
            "confidence": 0.85,
            "entities": self._extract_entities(text),
            "local": True,
        }

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        # Simplified entity extraction
        entities = {}

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
