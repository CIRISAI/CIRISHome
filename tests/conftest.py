"""
CIRISHome pytest configuration and global fixtures.

Using established patterns from Home Assistant, ESPHome, and Wyoming
protocol ecosystems.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

# Import ecosystem fixtures when available
try:
    from pytest_homeassistant import aioclient_mock, hass

    PYTEST_HOMEASSISTANT_AVAILABLE = True
except ImportError:
    PYTEST_HOMEASSISTANT_AVAILABLE = False

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "ciris-engine"))


# ============================================================================
# Environment Setup
# ============================================================================


@pytest.fixture(scope="session")
def ciris_home_config() -> Dict[str, Any]:
    """Global CIRISHome configuration."""
    return {
        "jetson_nano_ip": "192.168.1.100",
        "jetson_nano_port": 11434,
        "llm_model": "llama-4-scout-int4",
        "enable_voice": True,
        "voice_timeout": 30,
        "ha_token": "test-token-12345",
        "responsibility_accepted": True,
        "enable_cameras": True,
        "camera_urls": "front_door:rtsp://127.0.0.1:8554/front_door",
        "event_detection_enabled": True,
        "confidence_threshold": 0.7,
        "replace_google_events": True,
        "enable_geo_wisdom": True,
        "enable_weather_wisdom": True,
        "enable_sensor_wisdom": True,
    }


@pytest.fixture
def env_vars(
    monkeypatch: pytest.MonkeyPatch, ciris_home_config: Dict[str, Any]
) -> Dict[str, str]:
    """Set up environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-jetson-key",
        "OPENAI_API_BASE": (
            f"http://{ciris_home_config['jetson_nano_ip']}:"
            f"{ciris_home_config['jetson_nano_port']}/v1"
        ),
        "OPENAI_MODEL_NAME": ciris_home_config["llm_model"],
        "HOME_ASSISTANT_URL": "http://homeassistant.local:8123",
        "HOME_ASSISTANT_TOKEN": ciris_home_config["ha_token"],
        "I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY": "true",
        "LOCAL_PROCESSING_ONLY": "true",
        "MEDICAL_GRADE_PRIVACY": "true",
        "CIRIS_ENABLE_GEO_WISDOM": "true",
        "CIRIS_ENABLE_WEATHER_WISDOM": "true",
        "CIRIS_ENABLE_SENSOR_WISDOM": "true",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """Sample Home Assistant entities."""
    return [
        {
            "entity_id": "sensor.living_room_temperature",
            "state": "72.5",
            "attributes": {
                "unit_of_measurement": "°F",
                "friendly_name": "Living Room Temperature",
                "device_class": "temperature",
            },
        },
        {
            "entity_id": "sensor.living_room_humidity",
            "state": "45",
            "attributes": {
                "unit_of_measurement": "%",
                "friendly_name": "Living Room Humidity",
                "device_class": "humidity",
            },
        },
        {
            "entity_id": "light.living_room_lights",
            "state": "off",
            "attributes": {
                "friendly_name": "Living Room Lights",
                "brightness": 0,
            },
        },
        {
            "entity_id": "binary_sensor.front_door_motion",
            "state": "off",
            "attributes": {
                "friendly_name": "Front Door Motion",
                "device_class": "motion",
            },
        },
    ]


@pytest.fixture
def medical_entities() -> List[Dict[str, Any]]:
    """Medical/health entities that should be filtered out."""
    return [
        {
            "entity_id": "sensor.heart_rate_monitor",
            "state": "75",
            "attributes": {
                "unit_of_measurement": "bpm",
                "friendly_name": "Heart Rate",
                "device_class": "heart_rate",
            },
        },
        {
            "entity_id": "sensor.blood_pressure",
            "state": "120",
            "attributes": {
                "unit_of_measurement": "mmHg",
                "friendly_name": "Blood Pressure",
                "device_class": "blood_pressure",
            },
        },
        {
            "entity_id": "sensor.patient_temperature",
            "state": "98.6",
            "attributes": {
                "unit_of_measurement": "°F",
                "friendly_name": "Patient Temperature",
                "device_class": "body_temperature",
            },
        },
    ]


@pytest.fixture
def sample_audio() -> bytes:
    """Sample audio data for testing."""
    # Mock WAV file header + data
    # Mock WAV file header + data
    wav_header = (
        b"RIFF$\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00"
    )
    return wav_header + b"\x00" * 1000


@pytest.fixture
def sample_image() -> bytes:
    """Sample image data for testing."""
    # Simple PNG data
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d"
        b"\x00\x00\x00d\x08\x06\x00\x00\x00p\xe2\x95D"
    )
    return png_data


# ============================================================================
# Jetson Nano Fixtures
# ============================================================================


@pytest.fixture
def jetson_responses() -> Dict[str, Any]:
    """Sample Jetson Nano AI responses."""
    return {
        "stt": {
            "text": "What's the temperature in the living room?",
            "confidence": 0.96,
            "processing_time": 2.1,
        },
        "llm_simple": {
            "response": "The living room temperature is currently 72.5°F.",
            "tokens_used": 25,
            "inference_time": 1.2,
        },
        "llm_control": {
            "response": "I've turned on the living room lights for you.",
            "action": {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.living_room_lights",
            },
            "tokens_used": 30,
            "inference_time": 1.5,
        },
        "tts": {
            "audio_data": b"mock_tts_response_audio",
            "duration": 2.1,
            "processing_time": 1.5,
        },
    }


@pytest.fixture
def mock_jetson(jetson_responses: Dict[str, Any]) -> Mock:
    """Mock Jetson Nano service."""
    jetson = Mock()

    # Health check
    jetson.health = AsyncMock(
        return_value={
            "status": "healthy",
            "gpu_memory": "4.2GB/8.0GB",
            "models_loaded": [
                "llama-4-scout-int4",
                "whisper-large-v3",
                "coqui-tts",
            ],
        }
    )

    # STT - return different responses based on input
    def mock_transcribe(audio_data: bytes, **kwargs: Any) -> Dict[str, Any]:
        if b"lights" in audio_data:
            return {
                "text": "Turn on the living room lights",
                "confidence": 0.95,
                "processing_time": 2.0,
            }
        return jetson_responses["stt"]  # type: ignore[no-any-return]

    jetson.transcribe = AsyncMock(side_effect=mock_transcribe)

    # LLM
    def mock_generate(prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if "lights" in prompt.lower():
            # type: ignore[no-any-return]
            return jetson_responses["llm_control"]
        return jetson_responses["llm_simple"]  # type: ignore[no-any-return]

    jetson.generate = AsyncMock(side_effect=mock_generate)

    # TTS
    jetson.synthesize = AsyncMock(return_value=jetson_responses["tts"])

    return jetson


# ============================================================================
# Home Assistant Fixtures
# ============================================================================


@pytest.fixture
def mock_homeassistant(sample_entities: List[Dict[str, Any]]) -> Mock:
    """Mock Home Assistant API."""
    ha = Mock()

    # States API
    ha.get_states = AsyncMock(return_value=sample_entities)
    ha.get_state = AsyncMock()

    # Services API
    ha.call_service = AsyncMock(return_value={"success": True})

    # Config API
    ha.get_config = AsyncMock(
        return_value={
            "location_name": "Home",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone": "America/New_York",
        }
    )

    return ha


@pytest.fixture
def service_calls() -> tuple[List[Dict[str, Any]], Any]:
    """Track Home Assistant service calls."""
    calls: List[Dict[str, Any]] = []

    def add_call(domain: str, service: str, **kwargs: Any) -> Dict[str, bool]:
        calls.append({"domain": domain, "service": service, "data": kwargs})
        return {"success": True}

    return calls, add_call


# ============================================================================
# Voice PE Fixtures
# ============================================================================


@pytest.fixture
def voice_pe_devices() -> List[Dict[str, Any]]:
    """Voice PE puck configurations."""
    return [
        {
            "device_id": "voice_pe_living_room",
            "ip_address": "192.168.1.150",
            "location": "living_room",
        },
        {
            "device_id": "voice_pe_kitchen",
            "ip_address": "192.168.1.151",
            "location": "kitchen",
        },
    ]


@pytest.fixture
def wyoming_messages() -> Dict[str, Dict[str, Any]]:
    """Wyoming protocol message samples."""
    return {
        "info": {
            "type": "info",
            "asr": [{"name": "whisper"}],
            "tts": [{"name": "coqui"}],
        },
        "transcript": {
            "type": "transcript",
            "text": "Turn on the living room lights",
        },
        "audio": {
            "type": "audio",
            "rate": 22050,
            "audio": b"mock_audio_response",
        },
    }


@pytest.fixture
def mock_wyoming() -> Mock:
    """Mock Wyoming protocol handler."""
    wyoming = Mock()
    wyoming.connect = AsyncMock()
    wyoming.send_audio = AsyncMock()
    wyoming.receive_transcript = AsyncMock()
    wyoming.send_tts = AsyncMock()
    wyoming.receive_audio = AsyncMock()
    return wyoming


# ============================================================================
# Wisdom Module Fixtures
# ============================================================================


@pytest.fixture
def weather_data() -> Dict[str, Any]:
    """Sample weather data."""
    return {
        "location": "Anytown, NY",
        "temperature": 72,
        "conditions": "Partly Cloudy",
        "humidity": 65,
        "wind_speed": 8,
    }


@pytest.fixture
def geo_data() -> Dict[str, Any]:
    """Sample geographic data."""
    return {
        "address": "123 Main Street",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "route": {
            "distance": 2.5,
            "duration": 8,
            "steps": ["Head north", "Turn right", "Arrive"],
        },
    }


@pytest.fixture
def mock_wisdom_modules(
    weather_data: Dict[str, Any],
    geo_data: Dict[str, Any],
    sample_entities: List[Dict[str, Any]],
) -> Mock:
    """Mock wisdom modules."""
    wisdom = Mock()

    # Geo wisdom
    wisdom.geo = Mock()
    wisdom.geo.geocode = AsyncMock(return_value=geo_data)
    wisdom.geo.route = AsyncMock(return_value=geo_data["route"])

    # Weather wisdom
    wisdom.weather = Mock()
    wisdom.weather.current = AsyncMock(return_value=weather_data)
    wisdom.weather.forecast = AsyncMock(return_value=[weather_data])

    # Sensor wisdom (with filtering)
    wisdom.sensor = Mock()
    wisdom.sensor.get_safe_entities = AsyncMock(return_value=sample_entities)
    wisdom.sensor.filter_medical = Mock(return_value=[])  # Filters out medical

    return wisdom


# ============================================================================
# Integration Fixtures
# ============================================================================


@pytest.fixture
async def integration_setup(
    mock_jetson: Mock,
    mock_homeassistant: Mock,
    mock_wyoming: Mock,
    mock_wisdom_modules: Mock,
) -> Dict[str, Mock]:
    """Set up complete integration environment."""
    return {
        "jetson": mock_jetson,
        "homeassistant": mock_homeassistant,
        "wyoming": mock_wyoming,
        "wisdom": mock_wisdom_modules,
    }


# ============================================================================
# Home Assistant Ecosystem Fixtures
# ============================================================================

if not PYTEST_HOMEASSISTANT_AVAILABLE:

    @pytest.fixture
    def hass() -> Mock:  # noqa: F811
        """Mock Home Assistant instance following official patterns."""
        mock_hass = Mock()
        mock_hass.config_entries = Mock()
        mock_hass.config_entries.async_setup = AsyncMock(return_value=True)
        mock_hass.config_entries.async_unload = AsyncMock(return_value=True)
        mock_hass.services = Mock()
        mock_hass.services.async_call = AsyncMock()
        mock_hass.states = Mock()
        mock_hass.states.get = Mock(return_value=None)
        mock_hass.states.async_set = AsyncMock()
        mock_hass.bus = Mock()
        mock_hass.bus.async_fire = AsyncMock()
        return mock_hass

    @pytest.fixture
    def aioclient_mock() -> Generator[Mock, None, None]:  # noqa: F811
        """Mock aiohttp client session following HA patterns."""
        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession"
        ) as mock:
            mock_session = Mock()
            mock_session.get = AsyncMock()
            mock_session.post = AsyncMock()
            mock_session.put = AsyncMock()
            mock_session.delete = AsyncMock()
            mock.return_value = mock_session
            yield mock_session


# ============================================================================
# Wyoming Protocol Fixtures
# ============================================================================


@pytest.fixture
def wyoming_message() -> Dict[str, Any]:
    """Mock Wyoming protocol message."""
    return {
        "type": "transcribe",
        "data": {"text": "Turn on the living room lights", "confidence": 0.95},
    }


@pytest.fixture
def wyoming_client() -> Mock:
    """Mock Wyoming protocol client."""
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_message = AsyncMock()
    client.receive_message = AsyncMock()
    return client


@pytest.fixture
def wyoming_server() -> Mock:
    """Mock Wyoming protocol server."""
    server = Mock()
    server.start = AsyncMock()
    server.stop = AsyncMock()
    server.handle_client = AsyncMock()
    return server


# ============================================================================
# ESPHome Voice Assistant Fixtures
# ============================================================================


@pytest.fixture
def esphome_device() -> Mock:
    """Mock ESPHome voice assistant device."""
    device = Mock()
    device.name = "voice-puck-01"
    device.mac_address = "AA:BB:CC:DD:EE:FF"
    device.ip_address = "192.168.1.150"
    device.firmware_version = "2024.1.0"
    device.voice_assistant = Mock()
    device.voice_assistant.start_listening = AsyncMock()
    device.voice_assistant.stop_listening = AsyncMock()
    device.microphone = Mock()
    device.speaker = Mock()
    device.led_ring = Mock()
    return device


@pytest.fixture
def voice_pe_config() -> Dict[str, Any]:
    """Voice PE puck configuration."""
    return {
        "device_name": "voice-puck-01",
        "microphone": {
            "platform": "i2s_audio",
            "sample_rate": 16000,
            "bits_per_sample": 16,
        },
        "speaker": {
            "platform": "i2s_audio",
            "sample_rate": 22050,
            "bits_per_sample": 16,
        },
        "voice_assistant": {
            "microphone": "voice_microphone",
            "speaker": "voice_speaker",
            "use_wake_word": True,
            "wake_word": "hey_assistant",
        },
    }


# ============================================================================
# Jetson Nano GPU/CUDA Fixtures
# ============================================================================


@pytest.fixture
def mock_cuda() -> Generator[None, None, None]:
    """Mock CUDA environment for testing without GPU."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            with patch("torch.cuda.get_device_name", return_value="NVIDIA Tegra X1"):
                yield


@pytest.fixture
def jetson_gpu_info() -> Dict[str, Any]:
    """Jetson Nano GPU information."""
    return {
        "name": "NVIDIA Tegra X1",
        "memory_total": "4GB",
        "memory_free": "2.5GB",
        "cuda_version": "10.2",
        "driver_version": "32.7.1",
        "compute_capability": "5.3",
    }


@pytest.fixture
def quantized_model_config() -> Dict[str, Any]:
    """Provide configuration for quantized models on Jetson."""
    return {
        "model_name": "llama-4-scout-int4",
        "quantization": "int4",
        "memory_usage": "4.2GB",
        "inference_time": "2.1s",
        "tokens_per_second": 15.2,
    }


# ============================================================================
# Async Test Setup
# ============================================================================

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# CIRIS Engine WiseBus Fixtures
# ============================================================================


@pytest.fixture
def wise_bus_prohibited_capabilities() -> set[str]:
    """Return standard set of CIRIS Engine WiseBus prohibited capabilities."""
    return {
        "domain:medical",
        "domain:health",
        "domain:clinical",
        "domain:patient",
        "modality:sensor:medical",
        "modality:sensor:health",
        "modality:sensor:vital_signs",
        "action:medical_device_control",
        "capability:patient_data_access",
        "domain:home_automation",  # Should be removed by home_enabler
        "action:home_control",  # Should be removed by home_enabler
        "capability:device_automation",  # Should be removed by home_enabler
        "automation:lighting",  # Should be removed by home_enabler
        "automation:hvac",  # Should be removed by home_enabler
        "domain:unknown_future",  # Should remain (not home-related)
        "action:data_processing",  # Should remain (not home-related)
    }


@pytest.fixture
def mock_wise_bus(wise_bus_prohibited_capabilities: set[str]) -> Mock:
    """Mock WiseBus class with comprehensive capability system."""
    wise_bus = Mock()
    wise_bus.PROHIBITED_CAPABILITIES = wise_bus_prohibited_capabilities.copy()

    # Add realistic WiseBus methods and properties
    wise_bus.get_capabilities = Mock(
        return_value=list(wise_bus_prohibited_capabilities)
    )
    wise_bus.is_capability_prohibited = Mock()
    wise_bus.add_prohibited_capability = Mock()
    wise_bus.remove_prohibited_capability = Mock()

    # Mock the logger for tracking capability changes
    wise_bus.logger = Mock()

    return wise_bus


@pytest.fixture
def mock_wise_bus_module(mock_wise_bus: Mock) -> Mock:
    """Mock the entire ciris_engine.logic.buses.wise_bus module."""
    module = Mock()
    module.WiseBus = mock_wise_bus
    return module


@pytest.fixture
def wise_bus_import_success(mock_wise_bus_module: Mock) -> Any:
    """Context manager for successful WiseBus import."""
    # Use a more targeted approach - patch the specific import location
    return patch.dict(
        "sys.modules",
        {"ciris_engine.logic.buses.wise_bus": mock_wise_bus_module},
    )


@pytest.fixture
def wise_bus_import_failure() -> Any:
    """Context manager for WiseBus import failure."""
    # Mock the import to fail by making sys.modules raise ImportError

    def _failing_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ciris_engine.logic.buses.wise_bus":
            raise ImportError("No module named 'ciris_engine'")
        # Call the original __import__ for other modules
        return original_import(name, *args, **kwargs)

    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )
    return patch("builtins.__import__", side_effect=_failing_import)


@pytest.fixture
def wise_bus_exception() -> Any:
    """Context manager for WiseBus runtime exceptions."""
    # Create mock module that raises exception when
    # WiseBus.PROHIBITED_CAPABILITIES is accessed
    mock_module = Mock()
    mock_wise_bus = Mock()

    # Use PropertyMock to make PROHIBITED_CAPABILITIES access raise exception
    type(mock_wise_bus).PROHIBITED_CAPABILITIES = PropertyMock(
        side_effect=RuntimeError("WiseBus internal error")
    )

    mock_module.WiseBus = mock_wise_bus

    return patch.dict("sys.modules", {"ciris_engine.logic.buses.wise_bus": mock_module})


@pytest.fixture
def capability_audit_tracker() -> Mock:
    """Track capability changes for security auditing."""
    changes = []

    def track_change(
        action: str, capabilities: set[str], timestamp: Optional[Any] = None
    ) -> None:
        import datetime

        if timestamp is None:
            timestamp = datetime.datetime.now()
        changes.append(
            {
                "action": action,
                "capabilities": capabilities,
                "timestamp": timestamp,
            }
        )

    mock_tracker = Mock()
    mock_tracker.changes = changes
    mock_tracker.track_change = track_change
    return mock_tracker


# ============================================================================
# Test Categories
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_hardware: Hardware tests")
    config.addinivalue_line("markers", "safety: Safety/security tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Auto-mark tests based on location."""
    for item in items:
        # Auto-mark by directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark safety tests
        if "medical" in item.name or "filter" in item.name or "security" in item.name:
            item.add_marker(pytest.mark.safety)
