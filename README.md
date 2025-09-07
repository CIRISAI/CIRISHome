[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-DEPLOYMENT%20READY-brightgreen.svg)](#current-status)
[![Platform](https://img.shields.io/badge/Platform-Home%20Assistant%20+%20Jetson%20Orin-blue.svg)](#hardware-stack)
[![Tests](https://img.shields.io/badge/Tests-63%20passing-brightgreen.svg)](#testing-status)

# CIRIS Home

**Copyright © 2025 Eric Moore and CIRIS L3C** | **Apache 2.0 License**

**Multi-modal HOME automation platform with local AI processing**

**DEPLOYMENT READY** | 3-Box Hardware Stack | Voice PE + Jetson Orin Nano + HA Yellow

**CIRISHome** is a complete multi-modal home automation platform running on local hardware with **100% privacy-focused processing**. No cloud required.

## What It Does

**Voice-First Home Control**:

- **Voice Commands**: "Turn on living room lights", "What's the weather?"
- **Local AI Processing**: Llama-4-Scout INT4 quantized for Jetson Orin Nano
- **Complete Pipeline**: STT (Whisper) → Intent Classification → LLM → TTS (Coqui) → Home Assistant actions

**CIRIS Wisdom Modules Integration**:

- **Geographic**: Address lookup, routing (OpenStreetMap - no tracking)
- **Weather**: NOAA forecasts, alerts (free government data)
- **Smart Sensors**: Environmental data with medical sensor filtering for safety

**Multi-Modal Capabilities**:

- **Vision Processing**: Local camera analysis for home security/automation
- **Audio Intelligence**: Voice commands, acoustic event detection
- **Sensor Fusion**: Environmental, motion, and automation sensors (medical filtered)

**Privacy & Safety**:

- **100% Local**: All AI processing on your Jetson, no cloud required
- **Medical Filtering**: Automatically blocks medical sensors for liability protection
- **Home-Only Focus**: Environmental automation, not health monitoring

## Quick Start

### Hardware Requirements

1. **Home Assistant Yellow** (or similar HA device)
2. **Jetson Orin Nano** (8GB recommended)
3. **Voice PE Pucks** (ESPHome voice assistants)

### 3-Step Deployment

```bash
# 1. Configure your setup
cp .env.example .env
# Edit with your Jetson IP and Home Assistant token

# 2. Deploy complete stack
export JETSON_NANO_IP=192.168.1.100
export HA_TOKEN=your-home-assistant-token
export RESPONSIBILITY_ACCEPTED=true

./deploy.sh  # Builds Docker containers and tests full pipeline

# 3. Development mode (includes simulators)
docker-compose --profile dev up
```

**New to the setup?** → **[Complete Installation Guide](INSTALLATION.md)**

## Current Status: DEPLOYMENT READY

**Complete Stack Ready - September 2025**

**Voice Pipeline**: STT → Intent Classification → LLM → TTS → Home Assistant service calls  
**Jetson Integration**: Llama-4-Scout INT4 quantized, GPU optimized  
**Wisdom Modules**: Geo (OpenStreetMap), Weather (NOAA), Sensors (HA filtered)  
**Medical Filtering**: Comprehensive keyword detection for safety compliance  
**Testing**: 63 tests passing, 0 failures - ecosystem fixtures (HA, ESPHome, Wyoming, Jetson)  
**CI/CD**: GitHub Actions, pre-commit hooks, Docker Compose, security scanning  
**Multi-modal Support**: Audio, vision, sensor fusion capabilities ready

**Next Phase**: Voice PE puck configuration, advanced Jetson GPU acceleration

## Architecture

### 3-Box Hardware Stack

```mermaid
graph LR
    A[Voice PE Pucks] -->|Wyoming Protocol| B[Home Assistant Yellow]
    B -->|CIRIS Engine| C[Jetson Orin Nano]
    C -->|Local AI Models| B
    C -->|STT/LLM/TTS| A
```

**Processing Flow:**

1. **Voice PE Pucks** - Capture audio via Wyoming protocol
2. **Home Assistant Yellow** - Run CIRIS Engine, coordinate devices
3. **Jetson Orin Nano** - Process all AI (Whisper STT, DistilBERT Intent, Llama-4-Scout LLM, Coqui TTS)

### Key Components

- **Voice Pipeline**: Complete STT → Intent Classification → LLM → TTS → HA service call chain
- **CIRIS Wisdom**: Geographic, Weather, Sensor modules with safety filtering
- **Medical Filtering**: Automatic blocking of medical sensors for liability protection
- **Home Integration**: Native Home Assistant add-on with Docker Compose deployment

**Design Principles:**

- **100% Local Processing** - No cloud dependency for core functions
- **Privacy-First** - Medical-grade data protection
- **Resource Efficient** - Works on consumer hardware (Jetson Orin Nano 8GB)
- **Safety Focused** - Comprehensive medical sensor filtering

## Testing Status

**63 Tests Passing, 0 Failures**

```bash
# Run complete test suite
pytest tests/ -v

# Test categories
pytest tests/ -m "unit"         # Fast isolated component tests (7 tests)
pytest tests/ -m "integration"  # Component interaction tests (16 tests)
pytest tests/ -m "safety"       # Security/medical filtering tests (8 tests)

# Docker testing
docker-compose --profile test up test-runner
```

**Test Coverage by Component:**

- **Home Automation** (7 tests): Capability enablement, medical filtering
- **Jetson Integration** (16 tests): AI models, GPU utilization, error recovery
- **Security Filtering** (8 tests): Medical sensor detection, privacy compliance
- **Voice Pipeline** (14 tests): STT → Intent Classification → LLM → TTS → HA service calls
- **Wisdom Modules** (18 tests): Geo, weather, sensor integration with safety

**Ecosystem Integration:**

- Home Assistant fixtures using official `hass` and `aioclient_mock` patterns
- Wyoming Protocol fixtures for Voice PE puck communication
- ESPHome fixtures for voice assistant device and configuration testing
- Jetson Nano fixtures with CUDA mocking and quantized model testing

## Documentation

**[Complete Documentation Hub](docs/README.md)**

**Quick Links:**

- **[Installation Guide](docs/INSTALLATION.md)** - Setup and configuration
- **[Home Assistant Integration](docs/HA_INTEGRATION.md)** - Working with existing HA
- **[Vision Pipeline](docs/VISION.md)** - Camera and image processing
- **[Audio Processing](docs/AUDIO.md)** - Voice and sound analysis
- **[Medical Handoff](docs/MEDICAL_BRIDGE.md)** - CIRISMedical integration

## Relationship to CIRIS Ecosystem

**CIRISHome** serves as the multi-modal development platform:

- Develops vision, audio, and sensor capabilities
- Tests integration patterns with Home Assistant
- Provides foundation for CIRISMedical deployments
- Maintains strict separation from medical logic

**Related Repositories:**

- **[CIRISAgent](https://github.com/CIRISAI/CIRISAgent)** - Core AI engine (public)
- **[CIRISMedical](https://github.com/CIRISAI/CIRISMedical)** - Medical AI implementation (private)

## Development

### Prerequisites **READY**

- **Python 3.11/3.12** (tested on both versions)
- **Docker & Docker Compose** (for containerized development)
- **Home Assistant Yellow** + **Jetson Orin Nano** (for full hardware testing)

### Development Environment

```bash
# Install dependencies
pip install -r requirements-test.txt

# Development with simulators (no hardware needed)
docker-compose --profile dev up    # HA + Jetson simulators included

# Run comprehensive test suite
pytest tests/ -v                   # All 63 tests
pre-commit run --all-files         # Code quality checks

# CI/CD pipeline locally
./deploy.sh                        # Full build, test, and deploy check
```

### Code Quality Standards **ENFORCED**

- **Pre-commit Hooks**: Black formatting, flake8 linting, mypy type checking
- **Security Scanning**: Bandit security analysis, detect-secrets scanning
- **Test Coverage**: 63 tests covering all major components and integrations
- **CI/CD Pipeline**: GitHub Actions with Python 3.11/3.12 matrix testing

### Contributing

1. Read [CLAUDE.md](CLAUDE.md) - Understand the mission and architecture
2. Follow [Mission Driven Development](docs/MDD.md) principles
3. Ensure medical-grade privacy in all multi-modal processing
4. Test on resource-limited hardware
5. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines

## Privacy & Security

**Medical-Grade Privacy:**

- All processing happens locally
- No cloud services required for core functions
- Encrypted data handling
- Clear consent mechanisms
- Audit trails for all data processing

**Resource Efficiency:**

- Designed for limited hardware
- Optimized for underserved communities
- No luxury smart home dependencies
- Accessible deployment models

## Support

- **Issues**: [GitHub Issues](https://github.com/CIRISAI/CIRISHome/issues)
- **Security**: [SECURITY.md](SECURITY.md)
- **Home Assistant**: [HA Community Forum](https://community.home-assistant.io/)

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

**CIRIS Home: Multi-modal AI capabilities for medical applications**  
_Developing the vision, audio, and sensor processing needed to serve those who need medical AI most._

**Ready to contribute?** → **[Get started →](docs/CONTRIBUTING.md)**
