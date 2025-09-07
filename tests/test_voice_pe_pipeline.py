"""Tests for Voice PE pipeline configuration and orchestration."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.voice_pe_pipeline.config import VoicePEDevice, VoicePEPipelineConfig
from modules.voice_pe_pipeline.orchestrator import (
    HomeAssistantBridge,
    PipelineContext,
    VoicePEPipelineOrchestrator,
    WyomingProtocolBridge,
)


class TestVoicePEConfiguration:
    """Test Voice PE pipeline configuration loading."""

    def test_config_loads_successfully(self):
        """Test configuration loads from YAML file."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"

        config = VoicePEPipelineConfig(str(config_path))

        assert config is not None
        assert len(config.voice_pe_devices) >= 2

    def test_voice_pe_devices_configuration(self):
        """Test Voice PE device configuration parsing."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        devices = config.voice_pe_devices
        assert len(devices) >= 3  # living_room, kitchen, bedroom

        for device in devices:
            assert isinstance(device, VoicePEDevice)
            assert device.device_id.startswith("voice_pe_")
            assert device.ip_address.startswith("192.168.1.")
            assert device.wyoming_port == 10302
            assert "speech_to_text" in device.capabilities
            assert "text_to_speech" in device.capabilities

    def test_pipeline_configuration(self):
        """Test pipeline stage configuration."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        pipeline_config = config.pipeline_config
        assert len(pipeline_config.stages) >= 5

        # Check required stages
        stage_names = list(pipeline_config.stages.keys())
        assert any("wake_word" in stage for stage in stage_names)
        assert any("speech_to_text" in stage for stage in stage_names)
        assert any("intent" in stage for stage in stage_names)
        assert any("llm" in stage for stage in stage_names)
        assert any("text_to_speech" in stage for stage in stage_names)

        # Check latency targets
        latency_targets = pipeline_config.latency_targets
        assert latency_targets.get("total_pipeline", 0) <= 3000  # Under 3 seconds
        assert latency_targets.get("wake_word_detection", 0) <= 100  # Under 100ms

    def test_security_configuration(self):
        """Test security and privacy configuration."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        security_config = config.security_config

        # Verify privacy settings
        assert security_config.privacy.get("audio_data_retention") == "none"
        assert security_config.privacy.get("external_api_calls") == "prohibited"
        assert security_config.privacy.get("medical_data_access") == "blocked"

        # Verify compliance
        assert security_config.compliance.get("local_processing_only") is True
        assert security_config.compliance.get("medical_grade_privacy") is True
        assert security_config.compliance.get("gdpr_compliant") is True

    def test_medical_sensor_filtering(self):
        """Test medical sensor filtering functionality."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        # Medical sensors should be blocked
        medical_entities = [
            ("sensor.heart_rate_monitor", "Heart Rate Monitor"),
            ("sensor.blood_pressure", "Blood Pressure Sensor"),
            ("sensor.medical_device", "Medical Device"),
            ("sensor.patient_monitor", "Patient Room Monitor"),
        ]

        for entity_id, friendly_name in medical_entities:
            assert config.is_medical_sensor_blocked(
                entity_id, friendly_name
            ), f"Medical sensor not blocked: {entity_id}"

        # Safe sensors should not be blocked
        safe_entities = [
            ("sensor.living_room_temperature", "Living Room Temperature"),
            ("sensor.outdoor_humidity", "Outdoor Humidity"),
            ("sensor.air_quality", "Air Quality Monitor"),
        ]

        for entity_id, friendly_name in safe_entities:
            assert not config.is_medical_sensor_blocked(
                entity_id, friendly_name
            ), f"Safe sensor incorrectly blocked: {entity_id}"

    def test_device_lookup_methods(self):
        """Test device lookup functionality."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        # Test lookup by device ID
        device = config.get_device_by_id("voice_pe_living_room")
        assert device is not None
        assert device.location == "living_room"

        # Test lookup by location
        kitchen_device = config.get_device_by_location("kitchen")
        assert kitchen_device is not None
        assert kitchen_device.device_id == "voice_pe_kitchen"

        # Test non-existent lookups
        assert config.get_device_by_id("non_existent") is None
        assert config.get_device_by_location("garage") is None


class TestWyomingProtocolBridge:
    """Test Wyoming protocol bridge functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.voice_pe_devices = [
            VoicePEDevice(
                device_id="voice_pe_test",
                friendly_name="Test Device",
                location="test_room",
                ip_address="192.168.1.100",
                wyoming_port=10302,
                capabilities=["stt", "tts"],
                model_config={},
                privacy_settings={},
            )
        ]
        return config

    @pytest.mark.asyncio
    async def test_device_connection(self, mock_config):
        """Test device connection via Wyoming protocol."""
        bridge = WyomingProtocolBridge(mock_config)
        device = mock_config.voice_pe_devices[0]

        success = await bridge.connect_device(device)

        assert success is True
        assert bridge.connected_devices["voice_pe_test"] is True

    @pytest.mark.asyncio
    async def test_audio_transmission(self, mock_config):
        """Test audio data transmission to device."""
        bridge = WyomingProtocolBridge(mock_config)
        device = mock_config.voice_pe_devices[0]

        # First connect the device
        await bridge.connect_device(device)

        # Test audio transmission
        audio_data = b"mock_audio_data"
        success = await bridge.send_audio_to_device("voice_pe_test", audio_data)

        assert success is True

    @pytest.mark.asyncio
    async def test_device_status(self, mock_config):
        """Test device status retrieval."""
        bridge = WyomingProtocolBridge(mock_config)
        device = mock_config.voice_pe_devices[0]

        await bridge.connect_device(device)
        status = await bridge.get_device_status("voice_pe_test")

        assert status["device_id"] == "voice_pe_test"
        assert status["connected"] is True
        assert "capabilities" in status


class TestHomeAssistantBridge:
    """Test Home Assistant integration bridge."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.home_assistant_config = MagicMock()
        config.home_assistant_config.allowed_services = ["light", "climate", "camera"]
        config.is_medical_sensor_blocked = MagicMock(return_value=False)
        return config

    @pytest.fixture
    def sample_context(self):
        """Create sample pipeline context for testing."""
        return PipelineContext(
            request_id="test_123",
            device_id="voice_pe_living_room",
            location="living_room",
            timestamp=123456789.0,
        )

    @pytest.mark.asyncio
    async def test_turn_on_device_action(self, mock_config, sample_context):
        """Test turn on device home automation action."""
        bridge = HomeAssistantBridge(mock_config)

        entities = {"room": "living_room"}
        result = await bridge.execute_action("turn_on_device", entities, sample_context)

        assert result["success"] is True
        assert result["action"] == "light.turn_on"
        assert "living_room" in result["entity_id"]
        assert "turned on" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_medical_device_blocking(self, mock_config, sample_context):
        """Test that medical devices are blocked for safety."""
        # Configure mock to block medical devices
        mock_config.is_medical_sensor_blocked.return_value = True

        bridge = HomeAssistantBridge(mock_config)

        entities = {"room": "bedroom"}
        result = await bridge.execute_action("turn_on_device", entities, sample_context)

        assert result["success"] is False
        assert "medical device access blocked" in result["error"].lower()
        assert "safety" in result["fallback_response"].lower()

    @pytest.mark.asyncio
    async def test_brightness_adjustment(self, mock_config, sample_context):
        """Test brightness adjustment action."""
        bridge = HomeAssistantBridge(mock_config)

        entities = {"room": "kitchen", "number": 75}
        result = await bridge.execute_action(
            "adjust_brightness", entities, sample_context
        )

        assert result["success"] is True
        assert result["action"] == "light.turn_on"
        assert result["brightness_pct"] == 75
        assert "75%" in result["response"]

    @pytest.mark.asyncio
    async def test_temperature_control(self, mock_config, sample_context):
        """Test temperature control action."""
        bridge = HomeAssistantBridge(mock_config)

        entities = {"room": "living_room", "number": 72}
        result = await bridge.execute_action(
            "adjust_temperature", entities, sample_context
        )

        assert result["success"] is True
        assert result["action"] == "climate.set_temperature"
        assert result["temperature"] == 72
        assert "72 degrees" in result["response"]


class TestVoicePEPipelineOrchestrator:
    """Test complete pipeline orchestration."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create orchestrator with mocked services."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"

        with patch.multiple(
            "modules.voice_pe_pipeline.orchestrator",
            LocalSTTService=MagicMock,
            LocalIntentsService=MagicMock,
            LocalLLMService=MagicMock,
            LocalTTSService=MagicMock,
        ):
            orchestrator = VoicePEPipelineOrchestrator(str(config_path))

            # Mock service responses
            orchestrator.stt_service.speech_to_text = AsyncMock(
                return_value=MagicMock(text="Turn on the living room lights")
            )
            orchestrator.stt_service.initialize = AsyncMock(return_value=True)

            orchestrator.intents_service.classify_intent = AsyncMock(
                return_value=MagicMock(intent="turn_on_device")
            )
            orchestrator.intents_service.initialize = AsyncMock(return_value=True)

            orchestrator.llm_service.call_llm_structured = AsyncMock(
                return_value=MagicMock(
                    text="I'll turn on the lights in the living room for you."
                )
            )
            orchestrator.llm_service.initialize = AsyncMock(return_value=True)

            orchestrator.tts_service.text_to_speech = AsyncMock(
                return_value=b"mock_audio_response"
            )
            orchestrator.tts_service.initialize = AsyncMock(return_value=True)

            return orchestrator

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_orchestrator):
        """Test orchestrator initialization."""
        success = await mock_orchestrator.initialize()

        assert success is True
        # Verify all services were initialized
        mock_orchestrator.stt_service.initialize.assert_called_once()
        mock_orchestrator.intents_service.initialize.assert_called_once()
        mock_orchestrator.llm_service.initialize.assert_called_once()
        mock_orchestrator.tts_service.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_voice_pipeline(self, mock_orchestrator):
        """Test complete voice processing pipeline."""
        await mock_orchestrator.initialize()

        # Mock Wyoming bridge
        mock_orchestrator.wyoming_bridge.send_audio_to_device = AsyncMock(
            return_value=True
        )

        # Process voice request
        audio_input = b"mock_audio_input"
        context = await mock_orchestrator.process_voice_request(
            "voice_pe_living_room", audio_input
        )

        # Verify pipeline stages completed
        assert context.transcript == "Turn on the living room lights"
        assert context.intent == "turn_on_device"
        assert context.llm_response is not None
        assert context.audio_response == b"mock_audio_response"
        assert len(context.errors) == 0

        # Verify latency tracking
        assert "stt" in context.latencies
        assert "intent" in context.latencies
        assert "llm" in context.latencies
        assert "tts" in context.latencies
        assert "total" in context.latencies

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_orchestrator):
        """Test pipeline error handling and recovery."""
        await mock_orchestrator.initialize()

        # Make STT service fail
        mock_orchestrator.stt_service.speech_to_text = AsyncMock(
            side_effect=Exception("STT service unavailable")
        )

        # Process should handle error gracefully
        audio_input = b"mock_audio_input"
        context = await mock_orchestrator.process_voice_request(
            "voice_pe_living_room", audio_input
        )

        assert len(context.errors) > 0
        assert "STT service unavailable" in str(context.errors)

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, mock_orchestrator):
        """Test system health and performance monitoring."""
        await mock_orchestrator.initialize()

        health = await mock_orchestrator.get_system_health()

        assert "timestamp" in health
        assert "active_requests" in health
        assert "connected_devices" in health
        assert "total_devices" in health
        assert "performance_metrics" in health

        assert health["total_devices"] >= 3  # From config file

    @pytest.mark.asyncio
    async def test_concurrent_voice_requests(self, mock_orchestrator):
        """Test handling multiple concurrent voice requests."""
        await mock_orchestrator.initialize()

        # Mock Wyoming bridge
        mock_orchestrator.wyoming_bridge.send_audio_to_device = AsyncMock(
            return_value=True
        )

        # Process multiple concurrent requests
        audio_input = b"mock_audio_input"
        tasks = [
            mock_orchestrator.process_voice_request(
                "voice_pe_living_room", audio_input
            ),
            mock_orchestrator.process_voice_request("voice_pe_kitchen", audio_input),
            mock_orchestrator.process_voice_request("voice_pe_bedroom", audio_input),
        ]

        contexts = await asyncio.gather(*tasks)

        # All requests should complete successfully
        assert len(contexts) == 3
        for context in contexts:
            assert len(context.errors) == 0
            assert context.transcript is not None
            assert context.audio_response is not None

    @pytest.mark.asyncio
    async def test_latency_requirements(self, mock_orchestrator):
        """Test that pipeline meets latency requirements."""
        await mock_orchestrator.initialize()
        mock_orchestrator.wyoming_bridge.send_audio_to_device = AsyncMock(
            return_value=True
        )

        audio_input = b"mock_audio_input"
        context = await mock_orchestrator.process_voice_request(
            "voice_pe_living_room", audio_input
        )

        # Check latency requirements from config
        latency_targets = mock_orchestrator.config.get_processing_latency_budget()

        if "total_pipeline" in latency_targets:
            max_total_latency = latency_targets["total_pipeline"]
            actual_latency = context.latencies.get("total", 0)

            # In testing with mocked services, should be very fast
            assert (
                actual_latency < max_total_latency
            ), f"Pipeline too slow: {actual_latency}ms > {max_total_latency}ms"


class TestVoicePEPrivacyCompliance:
    """Test privacy and safety compliance."""

    def test_local_processing_enforcement(self):
        """Test that all processing is configured for local-only."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        assert config.is_local_processing_only() is True

        # Check pipeline stages are all local
        for stage_name, stage in config.pipeline_config.stages.items():
            assert stage.processing_location in [
                "edge_device",
                "local_jetson",
            ], (
                f"Non-local processing in stage {stage_name}: "
                f"{stage.processing_location}"
            )

    def test_audio_data_retention_policy(self):
        """Test audio data retention compliance."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        privacy_config = config.security_config.privacy
        assert privacy_config.get("audio_data_retention") == "none"

        # Check device privacy settings
        for device in config.voice_pe_devices:
            assert device.privacy_settings.get("data_retention_policy") == "none"
            assert device.privacy_settings.get("local_processing_only") is True

    def test_medical_grade_privacy_compliance(self):
        """Test medical grade privacy compliance."""
        config_path = Path(__file__).parent.parent / "config" / "voice_pe_pucks.yml"
        config = VoicePEPipelineConfig(str(config_path))

        compliance = config.security_config.compliance
        assert compliance.get("medical_grade_privacy") is True
        assert compliance.get("gdpr_compliant") is True
        assert compliance.get("responsibility_acceptance_required") is True

        # Verify no external API calls
        privacy = config.security_config.privacy
        assert privacy.get("external_api_calls") == "prohibited"
        assert privacy.get("medical_data_access") == "blocked"
