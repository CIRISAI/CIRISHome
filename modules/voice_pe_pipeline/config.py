"""
Voice PE Puck Pipeline Configuration Module.

Loads and manages configuration for multi-modal HOME automation pipeline
connecting Voice PE pucks, Jetson Orin Nano, and Home Assistant Yellow.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class VoicePEDevice:
    """Configuration for a single Voice PE puck device."""

    device_id: str
    friendly_name: str
    location: str
    ip_address: str
    wyoming_port: int
    capabilities: List[str]
    model_config: Dict[str, Any]
    privacy_settings: Dict[str, Any]


@dataclass
class WyomingConfig:
    """Wyoming protocol configuration."""

    server_config: Dict[str, Any]
    protocol_version: str
    supported_services: List[str]


@dataclass
class PipelineStage:
    """Configuration for a single pipeline processing stage."""

    service: str
    model: str
    processing_location: str
    privacy_level: str


@dataclass
class PipelineConfig:
    """Multi-modal pipeline configuration."""

    stages: Dict[str, PipelineStage]
    data_flow: Dict[str, str]
    latency_targets: Dict[str, int]


@dataclass
class SecurityConfig:
    """Security and privacy configuration."""

    encryption: Dict[str, bool]
    authentication: Dict[str, str]
    privacy: Dict[str, str]
    compliance: Dict[str, bool]


@dataclass
class HomeAssistantConfig:
    """Home Assistant integration configuration."""

    connection: Dict[str, Any]
    entity_filtering: Dict[str, Any]
    allowed_services: List[str]


@dataclass
class MonitoringConfig:
    """Monitoring and diagnostics configuration."""

    health_checks: Dict[str, int]
    metrics_collection: Dict[str, bool]
    alerting: Dict[str, Any]


class VoicePEPipelineConfig:
    """Main configuration class for Voice PE pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check environment variable first
        env_config = os.getenv("VOICE_PE_CONFIG_PATH")
        if env_config:
            return env_config

        # Default to config directory
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "voice_pe_pucks.yml")

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(self.config_path, "r", encoding="utf-8") as file:
                self._config = yaml.safe_load(file)

            logger.info(
                f"Loaded Voice PE pipeline configuration from {self.config_path}"
            )
            self._validate_config()

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        required_sections = [
            "voice_pe_devices",
            "wyoming",
            "pipeline",
            "home_assistant",
            "security",
            "monitoring",
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate device configurations
        devices = self._config.get("voice_pe_devices", [])
        if not devices:
            raise ValueError("No Voice PE devices configured")

        for device in devices:
            required_device_fields = ["device_id", "ip_address", "location"]
            for field in required_device_fields:
                if field not in device:
                    raise ValueError(f"Missing required device field: {field}")

        logger.info(f"Configuration validation passed for {len(devices)} devices")

    @property
    def voice_pe_devices(self) -> List[VoicePEDevice]:
        """Get configured Voice PE devices."""
        devices = []
        for device_config in self._config.get("voice_pe_devices", []):
            device = VoicePEDevice(
                device_id=device_config["device_id"],
                friendly_name=device_config["friendly_name"],
                location=device_config["location"],
                ip_address=device_config["ip_address"],
                wyoming_port=device_config.get("wyoming_port", 10302),
                capabilities=device_config.get("capabilities", []),
                model_config=device_config.get("model_config", {}),
                privacy_settings=device_config.get("privacy_settings", {}),
            )
            devices.append(device)
        return devices

    @property
    def wyoming_config(self) -> WyomingConfig:
        """Get Wyoming protocol configuration."""
        wyoming_data = self._config.get("wyoming", {})
        return WyomingConfig(
            server_config=wyoming_data.get("server_config", {}),
            protocol_version=wyoming_data.get("protocol_version", "1.5.0"),
            supported_services=wyoming_data.get("supported_services", []),
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        pipeline_data = self._config.get("pipeline", {})

        # Parse stages
        stages = {}
        stage_data = pipeline_data.get("stages", {})
        for stage_name, stage_config in stage_data.items():
            stages[stage_name] = PipelineStage(
                service=stage_config["service"],
                model=stage_config["model"],
                processing_location=stage_config["processing_location"],
                privacy_level=stage_config["privacy_level"],
            )

        return PipelineConfig(
            stages=stages,
            data_flow=pipeline_data.get("data_flow", {}),
            latency_targets=pipeline_data.get("latency_targets", {}),
        )

    @property
    def home_assistant_config(self) -> HomeAssistantConfig:
        """Get Home Assistant configuration."""
        ha_data = self._config.get("home_assistant", {})
        return HomeAssistantConfig(
            connection=ha_data.get("connection", {}),
            entity_filtering=ha_data.get("entity_filtering", {}),
            allowed_services=ha_data.get("allowed_services", []),
        )

    @property
    def security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_data = self._config.get("security", {})
        return SecurityConfig(
            encryption=security_data.get("encryption", {}),
            authentication=security_data.get("authentication", {}),
            privacy=security_data.get("privacy", {}),
            compliance=security_data.get("compliance", {}),
        )

    @property
    def monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_data = self._config.get("monitoring", {})
        return MonitoringConfig(
            health_checks=monitoring_data.get("health_checks", {}),
            metrics_collection=monitoring_data.get("metrics_collection", {}),
            alerting=monitoring_data.get("alerting", {}),
        )

    def get_device_by_id(self, device_id: str) -> Optional[VoicePEDevice]:
        """Get device configuration by device ID."""
        for device in self.voice_pe_devices:
            if device.device_id == device_id:
                return device
        return None

    def get_device_by_location(self, location: str) -> Optional[VoicePEDevice]:
        """Get device configuration by location."""
        for device in self.voice_pe_devices:
            if device.location == location:
                return device
        return None

    def is_medical_sensor_blocked(
        self, entity_id: str, friendly_name: str = ""
    ) -> bool:
        """Check if sensor should be blocked for medical/safety reasons."""
        ha_config = self.home_assistant_config
        entity_filtering = ha_config.entity_filtering

        # Check prohibited keywords
        prohibited_keywords = entity_filtering.get("prohibited_keywords", [])
        text_to_check = f"{entity_id} {friendly_name}".lower()

        return any(keyword in text_to_check for keyword in prohibited_keywords)

    def get_processing_latency_budget(self) -> Dict[str, int]:
        """Get latency budget for each processing stage."""
        return self.pipeline_config.latency_targets

    def is_local_processing_only(self) -> bool:
        """Check if system is configured for local processing only."""
        compliance_settings = self.security_config.compliance
        return compliance_settings.get("local_processing_only", True) is True

    def get_pipeline_stages_order(self) -> List[str]:
        """Get ordered list of pipeline stage names."""
        stages = list(self.pipeline_config.stages.keys())
        # Sort by stage number (e.g., "1_wake_word", "2_speech_to_text")
        return sorted(stages, key=lambda x: int(x.split("_")[0]) if "_" in x else 999)


def load_voice_pe_config(config_path: Optional[str] = None) -> VoicePEPipelineConfig:
    """Load Voice PE pipeline configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        VoicePEPipelineConfig: Loaded configuration object

    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If configuration is invalid
    """
    return VoicePEPipelineConfig(config_path)
