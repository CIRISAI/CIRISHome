"""
Multi-modal HOME Pipeline Orchestrator.

Orchestrates the complete voice-to-action pipeline across Voice PE pucks,
Jetson Orin Nano AI processing, and Home Assistant automation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..local_models.service import (
    LocalIntentsService,
    LocalLLMService,
    LocalSTTService,
    LocalTTSService,
)
from .config import VoicePEDevice, VoicePEPipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context information for a pipeline execution."""

    request_id: str
    device_id: str
    location: str
    timestamp: float
    audio_data: Optional[bytes] = None
    transcript: Optional[str] = None
    intent: Optional[str] = None
    llm_response: Optional[str] = None
    home_assistant_action: Optional[Dict[str, Any]] = None
    audio_response: Optional[bytes] = None
    latencies: Optional[Dict[str, float]] = None
    errors: Optional[List[str]] = None


class WyomingProtocolBridge:
    """Bridge for Wyoming protocol communication with Voice PE pucks."""

    def __init__(self, config: VoicePEPipelineConfig):
        """Initialize Wyoming protocol bridge."""
        self.config = config
        self.connected_devices: Dict[str, bool] = {}

    async def connect_device(self, device: VoicePEDevice) -> bool:
        """Connect to a Voice PE device via Wyoming protocol."""
        try:
            # Simulate Wyoming connection for now
            # In production, this would establish actual Wyoming protocol connection
            logger.info(
                f"Connecting to {device.device_id} at {device.ip_address}:{device.wyoming_port}"
            )

            # Mock connection success
            await asyncio.sleep(0.1)  # Simulate connection time
            self.connected_devices[device.device_id] = True

            logger.info(f"Successfully connected to {device.device_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {device.device_id}: {e}")
            self.connected_devices[device.device_id] = False
            return False

    async def send_audio_to_device(self, device_id: str, audio_data: bytes) -> bool:
        """Send TTS audio back to Voice PE device for playback."""
        if not self.connected_devices.get(device_id, False):
            logger.error(f"Device {device_id} not connected")
            return False

        try:
            # In production, this would stream audio via Wyoming protocol
            logger.debug(f"Sending {len(audio_data)} bytes of audio to {device_id}")
            await asyncio.sleep(0.1)  # Simulate transmission time
            return True

        except Exception as e:
            logger.error(f"Failed to send audio to {device_id}: {e}")
            return False

    async def get_device_status(self, device_id: str) -> Dict[str, Any]:
        """Get status information from a Voice PE device."""
        return {
            "device_id": device_id,
            "connected": self.connected_devices.get(device_id, False),
            "last_ping": time.time(),
            "capabilities": ["stt", "tts", "wake_word"],
        }


class HomeAssistantBridge:
    """Bridge for Home Assistant integration."""

    def __init__(self, config: VoicePEPipelineConfig):
        """Initialize Home Assistant bridge."""
        self.config = config
        self.ha_config = config.home_assistant_config

    async def execute_action(
        self, intent: str, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Execute home automation action based on intent and entities."""
        try:
            # Parse intent to determine action
            if intent == "turn_on_device":
                return await self._handle_turn_on(entities, context)
            elif intent == "turn_off_device":
                return await self._handle_turn_off(entities, context)
            elif intent == "adjust_brightness":
                return await self._handle_brightness(entities, context)
            elif intent == "adjust_temperature":
                return await self._handle_temperature(entities, context)
            elif intent == "show_camera":
                return await self._handle_camera(entities, context)
            else:
                return await self._handle_general_query(entities, context)

        except Exception as e:
            logger.error(f"Failed to execute HA action for intent {intent}: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_response": "I'm sorry, I couldn't perform that action right now.",
            }

    async def _handle_turn_on(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle turn on device intent."""
        room = entities.get("room", context.location)
        device_type = "lights"  # Default to lights

        # Check if request is for medical sensors (safety check)
        device_description = f"turn on {device_type} in {room}"
        if self.config.is_medical_sensor_blocked("", device_description):
            return {
                "success": False,
                "error": "Medical device access blocked for safety",
                "fallback_response": "I cannot control medical devices for safety reasons.",
            }

        # Simulate Home Assistant service call
        await asyncio.sleep(0.1)
        logger.info(f"Turning on {device_type} in {room}")

        return {
            "success": True,
            "action": "light.turn_on",
            "entity_id": f"light.{room}_{device_type}",
            "response": f"I've turned on the {device_type} in the {room}.",
        }

    async def _handle_turn_off(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle turn off device intent."""
        room = entities.get("room", context.location)
        device_type = "lights"

        await asyncio.sleep(0.1)
        logger.info(f"Turning off {device_type} in {room}")

        return {
            "success": True,
            "action": "light.turn_off",
            "entity_id": f"light.{room}_{device_type}",
            "response": f"I've turned off the {device_type} in the {room}.",
        }

    async def _handle_brightness(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle brightness adjustment intent."""
        room = entities.get("room", context.location)
        brightness = entities.get("number", 50)  # Default 50%

        await asyncio.sleep(0.1)
        logger.info(f"Setting brightness to {brightness}% in {room}")

        return {
            "success": True,
            "action": "light.turn_on",
            "entity_id": f"light.{room}_lights",
            "brightness_pct": brightness,
            "response": f"I've set the brightness to {brightness}% in the {room}.",
        }

    async def _handle_temperature(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle temperature adjustment intent."""
        room = entities.get("room", context.location)
        temperature = entities.get("number", 72)  # Default 72°F

        await asyncio.sleep(0.1)
        logger.info(f"Setting temperature to {temperature}°F in {room}")

        return {
            "success": True,
            "action": "climate.set_temperature",
            "entity_id": f"climate.{room}_thermostat",
            "temperature": temperature,
            "response": f"I've set the temperature to {temperature} degrees in the {room}.",
        }

    async def _handle_camera(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle camera viewing intent."""
        camera_location = entities.get("room", "front_door")

        await asyncio.sleep(0.1)
        logger.info(f"Accessing camera: {camera_location}")

        return {
            "success": True,
            "action": "camera.snapshot",
            "entity_id": f"camera.{camera_location}_camera",
            "response": f"I've accessed the {camera_location} camera for you.",
        }

    async def _handle_general_query(
        self, entities: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Handle general information queries."""
        await asyncio.sleep(0.1)

        return {
            "success": True,
            "action": "information_query",
            "response": "I've processed your request. Is there anything specific you'd like me to help with?",
        }


class VoicePEPipelineOrchestrator:
    """Main orchestrator for the multi-modal HOME pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline orchestrator."""
        self.config = VoicePEPipelineConfig(config_path)
        self.wyoming_bridge = WyomingProtocolBridge(self.config)
        self.ha_bridge = HomeAssistantBridge(self.config)

        # AI services (running on Jetson Orin Nano)
        self.stt_service = LocalSTTService()
        self.intents_service = LocalIntentsService()
        self.llm_service = LocalLLMService()
        self.tts_service = LocalTTSService()

        # Runtime state
        self.active_contexts: Dict[str, PipelineContext] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "stt_latency": [],
            "intent_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "total_latency": [],
        }

    async def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing Voice PE pipeline orchestrator...")

            # Initialize AI services
            services = [
                ("STT", self.stt_service),
                ("Intents", self.intents_service),
                ("LLM", self.llm_service),
                ("TTS", self.tts_service),
            ]

            for service_name, service in services:
                logger.info(f"Initializing {service_name} service...")
                success = await service.initialize()
                if not success:
                    logger.error(f"Failed to initialize {service_name} service")
                    return False

            # Connect to Voice PE devices
            for device in self.config.voice_pe_devices:
                await self.wyoming_bridge.connect_device(device)

            logger.info("Pipeline orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False

    async def process_voice_request(
        self, device_id: str, audio_data: bytes, request_id: Optional[str] = None
    ) -> PipelineContext:
        """Process complete voice request through the multi-modal pipeline."""
        if request_id is None:
            request_id = f"{device_id}_{int(time.time() * 1000)}"

        device = self.config.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"Unknown device ID: {device_id}")

        context = PipelineContext(
            request_id=request_id,
            device_id=device_id,
            location=device.location,
            timestamp=time.time(),
            audio_data=audio_data,
            latencies={},
            errors=[],
        )

        self.active_contexts[request_id] = context

        try:
            # Stage 1: Speech-to-Text (Jetson Orin Nano)
            await self._process_stt_stage(context)

            # Stage 2: Intent Classification (Jetson Orin Nano)
            await self._process_intent_stage(context)

            # Stage 3: LLM Processing (Jetson Orin Nano)
            await self._process_llm_stage(context)

            # Stage 4: Home Assistant Action (if needed)
            await self._process_ha_stage(context)

            # Stage 5: Text-to-Speech (Jetson Orin Nano)
            await self._process_tts_stage(context)

            # Stage 6: Audio Delivery (Voice PE Puck)
            await self._process_audio_delivery(context)

            # Calculate total latency
            total_latency = time.time() - context.timestamp
            context.latencies["total"] = total_latency * 1000  # Convert to ms

            # Update performance metrics
            self._update_performance_metrics(context)

            logger.info(f"Completed voice request {request_id} in {total_latency:.2f}s")
            return context

        except Exception as e:
            logger.error(f"Pipeline error for request {request_id}: {e}")
            context.errors.append(str(e))
            return context

        finally:
            # Clean up active context
            self.active_contexts.pop(request_id, None)

    async def _process_stt_stage(self, context: PipelineContext) -> None:
        """Process Speech-to-Text stage."""
        start_time = time.time()

        try:
            stt_result = await self.stt_service.speech_to_text(context.audio_data)
            context.transcript = stt_result.text
            context.latencies["stt"] = (time.time() - start_time) * 1000

            logger.debug(f"STT result: {context.transcript}")

        except Exception as e:
            context.errors.append(f"STT stage failed: {e}")
            raise

    async def _process_intent_stage(self, context: PipelineContext) -> None:
        """Process Intent Classification stage."""
        if not context.transcript:
            raise ValueError("No transcript available for intent classification")

        start_time = time.time()

        try:
            intent_result = await self.intents_service.classify_intent(
                context.transcript
            )
            context.intent = intent_result.intent
            context.latencies["intent"] = (time.time() - start_time) * 1000

            logger.debug(f"Intent: {context.intent}")

        except Exception as e:
            context.errors.append(f"Intent stage failed: {e}")
            raise

    async def _process_llm_stage(self, context: PipelineContext) -> None:
        """Process LLM reasoning stage."""
        if not context.transcript or not context.intent:
            raise ValueError("Missing transcript or intent for LLM processing")

        start_time = time.time()

        try:
            # Create conversation context
            messages = [
                {
                    "role": "user",
                    "content": f"User in {context.location} said: '{context.transcript}'. Intent: {context.intent}. Provide helpful response.",
                }
            ]

            from ..local_models.service import LLMMessage

            llm_messages = [
                LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages
            ]

            llm_result = await self.llm_service.call_llm_structured(llm_messages)
            context.llm_response = llm_result.text
            context.latencies["llm"] = (time.time() - start_time) * 1000

            logger.debug(f"LLM response: {context.llm_response}")

        except Exception as e:
            context.errors.append(f"LLM stage failed: {e}")
            raise

    async def _process_ha_stage(self, context: PipelineContext) -> None:
        """Process Home Assistant action stage."""
        if context.intent in [
            "turn_on_device",
            "turn_off_device",
            "adjust_brightness",
            "adjust_temperature",
            "show_camera",
        ]:
            start_time = time.time()

            try:
                # Extract entities (simplified)
                entities = {"room": context.location}

                ha_result = await self.ha_bridge.execute_action(
                    context.intent, entities, context
                )
                context.home_assistant_action = ha_result
                context.latencies["ha_action"] = (time.time() - start_time) * 1000

                # Update LLM response with HA action result
                if ha_result.get("success"):
                    context.llm_response = ha_result.get(
                        "response", context.llm_response
                    )

            except Exception as e:
                context.errors.append(f"Home Assistant stage failed: {e}")
                # Continue with original LLM response

    async def _process_tts_stage(self, context: PipelineContext) -> None:
        """Process Text-to-Speech stage."""
        if not context.llm_response:
            raise ValueError("No LLM response available for TTS")

        start_time = time.time()

        try:
            audio_data = await self.tts_service.text_to_speech(context.llm_response)
            context.audio_response = audio_data
            context.latencies["tts"] = (time.time() - start_time) * 1000

            logger.debug(f"Generated {len(audio_data)} bytes of TTS audio")

        except Exception as e:
            context.errors.append(f"TTS stage failed: {e}")
            raise

    async def _process_audio_delivery(self, context: PipelineContext) -> None:
        """Deliver audio response back to Voice PE puck."""
        if not context.audio_response:
            raise ValueError("No audio response available for delivery")

        start_time = time.time()

        try:
            success = await self.wyoming_bridge.send_audio_to_device(
                context.device_id, context.audio_response
            )

            if not success:
                raise RuntimeError(
                    f"Failed to send audio to device {context.device_id}"
                )

            context.latencies["audio_delivery"] = (time.time() - start_time) * 1000

        except Exception as e:
            context.errors.append(f"Audio delivery failed: {e}")
            raise

    def _update_performance_metrics(self, context: PipelineContext) -> None:
        """Update performance metrics with context latencies."""
        for metric, latency in context.latencies.items():
            if metric in self.performance_metrics:
                self.performance_metrics[metric].append(latency)
                # Keep only last 100 measurements
                if len(self.performance_metrics[metric]) > 100:
                    self.performance_metrics[metric].pop(0)

    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health and performance metrics."""
        health_data = {
            "timestamp": time.time(),
            "active_requests": len(self.active_contexts),
            "connected_devices": sum(self.wyoming_bridge.connected_devices.values()),
            "total_devices": len(self.config.voice_pe_devices),
            "performance_metrics": {},
        }

        # Calculate performance statistics
        for metric, values in self.performance_metrics.items():
            if values:
                health_data["performance_metrics"][metric] = {
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "samples": len(values),
                }

        return health_data

    async def shutdown(self) -> None:
        """Gracefully shutdown the pipeline orchestrator."""
        logger.info("Shutting down pipeline orchestrator...")

        # Cancel any active requests
        for request_id in list(self.active_contexts.keys()):
            logger.info(f"Cancelling active request: {request_id}")
            self.active_contexts.pop(request_id, None)

        logger.info("Pipeline orchestrator shutdown complete")
