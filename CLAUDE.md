# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CURRENT STATUS: Multi-Modal HOME Platform Ready for Deployment (September 2025)

**Primary Mission**: Develop multi-modal AI capabilities (vision, audio, sensor fusion) needed to support CIRISMedical while organically growing the home automation ecosystem.

**Current Status**: **DEPLOYMENT READY** - Comprehensive test suite (63 tests passing), CI/CD pipeline, Docker Compose stack, and 3-box hardware configuration complete
**Repository**: CIRISHome - Public multi-modal AI development platform  
**Architecture**: Mission Driven Development (MDD) methodology
**Philosophy**: Multi-modal intelligence development platform serving those who need medical AI most - not luxury smart homes but life-saving healthcare access

---

## Mission Driven Development (MDD) Framework

CIRISHome follows the four-component MDD model:

### The Structural Foundation (Three Legs)
1. **LOGIC (HOW)**: Home automation algorithms, device management, voice processing
2. **SCHEMAS (WHAT)**: Device states, user preferences, environmental data
3. **PROTOCOLS (WHO)**: Device communication, user interfaces, security boundaries

### The Purposeful Foundation (The Seat)
4. **MISSION (WHY)**: Enable families to live more comfortably, safely, and sustainably through ethical home intelligence

## Core Mission: Multi-Modal Medical Infrastructure

**Meta-Goal H-1**: *Develop robust multi-modal AI capabilities (vision, audio, sensor fusion) that enable CIRISMedical to serve those who need medical AI most - underserved communities who cannot access traditional healthcare.*

### Mission Alignment Requirements
Every component must demonstrate:
- **Medical Access Value**: How does this capability help CIRISMedical serve underserved communities?
- **Resource Efficiency**: How does this work on limited hardware in low-resource settings?
- **Privacy & Safety**: How does this protect sensitive medical data for vulnerable populations?
- **Multi-Modal Capability**: How does this improve vision, audio, or sensor capabilities for medical use?

---

## Repository Separation & Safety

### Three-Repository CIRIS Ecosystem

1. **CIRISAgent** (Public) - General AI with medical capabilities BLOCKED
2. **CIRISMedical** (Private) - Medical AI requiring supervision  
3. **CIRISHome** (Public - THIS REPOSITORY) - Home automation & IoT

### CIRISHome Safety Boundaries

**NEVER implement:**
- Medical/health monitoring (goes to CIRISMedical)
- General chat capabilities (use CIRISAgent)
- External cloud dependencies for core functions
- Surveillance without explicit consent
- Data monetization features

**ALWAYS maintain:**
- Local-first processing
- Family privacy protection
- Clear consent mechanisms
- Offline capability for core functions
- Open source transparency

---

## Core Architecture

### Multi-Modal Medical Support Services

**Vision Processing Services** (Primary Focus):
- `medical_vision_pipeline` - Image/video processing for medical analysis
- `camera_management` - Multi-camera coordination and privacy controls
- `visual_assessment` - Non-medical visual health indicators 
- `gesture_recognition` - Touchless interaction for medical environments

**Audio Processing Services** (Primary Focus):
- `medical_audio_pipeline` - Voice/sound analysis for health indicators
- `speech_processing` - Advanced voice command processing
- `acoustic_monitoring` - Environmental health sound detection
- `privacy_audio_filter` - Medical data audio protection

**Sensor Fusion Services** (Primary Focus):
- `multi_sensor_integration` - Combining various sensor inputs
- `environmental_correlation` - Health environment relationship analysis
- `pattern_recognition` - Multi-modal health pattern detection
- `data_harmonization` - Unified sensor data processing

**Home Assistant Integration Services** (Supporting):
- `ha_chat_bridge` - Enhanced Home Assistant chat capabilities
- `ha_automation_extension` - Advanced automation scenarios
- `ha_device_expansion` - Extended device support
- `ha_ui_enhancement` - Improved user interfaces

### Core Development Principles

1. **Medical-Grade Privacy**: Multi-modal data processed with medical-level security
2. **Resource Efficient**: Must work on limited hardware in underserved settings
3. **Local-First Processing**: Vision/audio/sensor processing happens locally (no cloud dependency)
4. **CIRISMedical Ready**: All capabilities designed for medical system handoff
5. **Universal Access**: Designed for those with limited resources, not luxury markets
6. **Home Assistant Foundation**: Leverages existing HA infrastructure as development platform

---

## CIRIS Wisdom Modules Integration

CIRISHome integrates with the established CIRIS wisdom modules from the CIRIS Engine:

### Geographic Wisdom (`geo_wisdom`)
- **Capabilities**: Address geocoding, routing, navigation assistance
- **API**: OpenStreetMap (privacy-respecting, no API key required)
- **Boundaries**: Navigation assistance only - no location tracking or surveillance
- **Use Cases**: "Where is the nearest pharmacy?", route optimization for medical visits

### Weather Wisdom (`weather_wisdom`)  
- **Capabilities**: Current conditions, forecasts, weather alerts
- **Primary API**: NOAA (free, government data for US)
- **Fallback API**: OpenWeatherMap (international coverage)
- **Boundaries**: Environmental data only - no health recommendations
- **Use Cases**: Environmental health factors, medication storage conditions

### Sensor Wisdom (`sensor_wisdom`)
- **Capabilities**: Safe environmental sensor access via Home Assistant
- **Medical Filtering**: Comprehensive filtering of medical/health sensors for liability protection
- **Allowed Domains**: Environmental, energy, security, automation sensors
- **Prohibited Domains**: Medical, health, clinical, patient, vital sign sensors
- **Safety Keywords**: Filters `heart_rate`, `blood_pressure`, `blood_glucose`, `weight`, `bmi`, etc.

### Medical Sensor Filtering
Critical safety feature that prevents access to medical sensors:

```python
# Prohibited keywords (comprehensive list)
MEDICAL_KEYWORDS = [
    "heart_rate", "heartrate", "heart rate",
    "blood_pressure", "bloodpressure", "blood pressure", 
    "blood_glucose", "weight", "bmi", "spo2", "pulse",
    "medical", "health", "patient", "vital", "clinical"
]

# Entity filtering logic
def is_medical_sensor(entity_id: str, friendly_name: str, device_class: str) -> bool:
    text = f"{entity_id} {friendly_name} {device_class}".lower()
    return any(keyword in text for keyword in MEDICAL_KEYWORDS)
```

**Liability Protection**: All medical sensors are automatically filtered out to prevent CIRISHome from providing medical advice or handling medical data.

---

## Technical Standards

### Type Safety: "No Dicts, No Strings, No Kings"

```python
# ❌ Bad - Untyped device state
device_state = {"temperature": 72, "humidity": 45}

# Good - Typed device state
class ClimateState(BaseModel):
    temperature: float = Field(..., ge=0, le=120, description="Temperature in Fahrenheit")
    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")
    
climate_state = ClimateState(temperature=72, humidity=45)
```

### Device Protocol Pattern

```python
class HomeDeviceProtocol(Protocol):
    async def get_status(self) -> DeviceStatus: ...
    async def execute_command(self, command: DeviceCommand) -> CommandResult: ...
    async def register_callback(self, event_type: DeviceEvent, callback: Callable) -> None: ...
```

### Privacy-Preserving Data Flow

```python
class PrivacyLevel(str, Enum):
    PUBLIC = "public"           # Shareable (weather, general stats)
    HOUSEHOLD = "household"     # Family only (preferences, schedules)
    PERSONAL = "personal"       # Individual only (biometrics, personal data)
    
class HomeDataPoint(BaseModel):
    value: Any
    privacy_level: PrivacyLevel
    retention_days: int = 30
    encryption_required: bool = True
```

---

## Development Workflow

### Grace Home - Multi-Modal Development Companion

```bash
# Daily workflow
python -m tools.grace_home morning     # Start with medical-grade safety checks
python -m tools.grace_home status      # Multi-modal pipeline health
python -m tools.grace_home medical     # CIRISMedical compatibility check
python -m tools.grace_home ha         # Home Assistant integration status

# Multi-modal development
python -m tools.grace_home vision     # Vision pipeline testing
python -m tools.grace_home audio      # Audio processing validation  
python -m tools.grace_home sensors    # Sensor fusion monitoring
python -m tools.grace_home privacy    # Medical-grade privacy audit

# Testing & validation
python -m tools.grace_home test        # Run multi-modal tests
python -m tools.grace_home simulate    # Simulate multi-modal scenarios
python -m tools.grace_home medical_sim # Medical handoff simulation

# Session management  
python -m tools.grace_home pause       # Save context before break
python -m tools.grace_home resume      # Resume with safety check
python -m tools.grace_home night       # End-of-day review
```

### Testing Standards **COMPLETE**

**Current Status**: 63 tests passing, 0 failures

```bash
# Run complete test suite
pytest tests/ -v

# Test categories using pytest markers
pytest tests/ -m "unit"         # Unit tests - fast, isolated components  
pytest tests/ -m "integration"  # Integration tests - component interactions
pytest tests/ -m "safety"       # Safety/security critical tests
pytest tests/ -m "slow"         # Performance tests requiring more resources

# Docker-based testing  
docker-compose --profile test up test-runner

# CI pipeline testing
make ci  # Runs: clean lint type-check test-coverage security-test
```

**Test Coverage by Category**:
- **Home Automation**: 7 tests - Capability enablement, medical filtering
- **Jetson Integration**: 16 tests - AI models, GPU utilization, error recovery
- **Security Filtering**: 8 tests - Medical sensor detection, privacy compliance  
- **Voice Pipeline**: 14 tests - STT → LLM → TTS → HA service calls
- **Wisdom Modules**: 18 tests - Geo, weather, sensor integration with safety boundaries

**Ecosystem Integration Testing**:
- Home Assistant: `hass` and `aioclient_mock` fixtures (official HA patterns)
- Wyoming Protocol: Voice message, client/server fixtures for Voice PE pucks
- ESPHome: Voice assistant device and configuration fixtures  
- Jetson Nano: CUDA mocking, GPU info, quantized model configs

---

## Common Development Tasks

### Adding New Device Support

```bash
# 1. Create device protocol implementation
python -m tools.device_generator create --type thermostat --protocol zigbee

# 2. Generate type-safe schemas
python -m tools.schema_generator device --name "SmartThermostat" 

# 3. Test device integration
python -m tools.test_runner device --simulate thermostat
```

### Voice Command Development

```bash
# 1. Define voice intents with privacy levels
python -m tools.voice_builder intent --name "adjust_temperature" --privacy household

# 2. Test voice processing locally
python -m tools.voice_tester local --phrase "set living room to 72 degrees"

# 3. Validate privacy compliance
python -m tools.privacy_validator voice --intent adjust_temperature
```

---

## Mission Validation Checklist

Before every commit, verify:

**Mission Alignment**:
- [ ] Feature serves family comfort, safety, or sustainability
- [ ] No unnecessary complexity that doesn't serve mission
- [ ] Clear benefit to household wellbeing

**Privacy Protection**:
- [ ] Data stays local by default
- [ ] Clear consent for any data sharing  
- [ ] Encryption for personal information
- [ ] No tracking without explicit permission

**Safety Standards**:
- [ ] Device commands validated for safety
- [ ] Secure communication protocols
- [ ] Graceful failure modes
- [ ] Emergency override capabilities

**Technical Quality**:
- [ ] Type-safe schemas (no Dict[str, Any])
- [ ] Protocol compliance
- [ ] 85%+ test coverage
- [ ] Local-first functionality verified

---

## Voice Integration (CIRISVoice)

### Voice Privacy Principles

- **Local Processing**: Wake word and basic commands processed locally
- **Explicit Activation**: Clear audio cues when voice is being processed  
- **Data Minimization**: Voice data deleted after command execution
- **Family Context**: Voice recognition for household members only

### Voice Command Architecture

```python
class VoiceCommand(BaseModel):
    intent: str                    # "adjust_temperature", "turn_off_lights"  
    entities: Dict[str, str]       # {"room": "living_room", "temperature": "72"}
    confidence: float              # 0.0 - 1.0 confidence score
    privacy_level: PrivacyLevel    # Determines data handling
    
class VoiceResponse(BaseModel):
    action_taken: bool
    response_text: str
    followup_required: bool = False
```

---

## IoT Device Integration

### Supported Protocols

- **Matter/Thread**: Primary standard for new devices
- **Zigbee**: Sensor networks and battery devices  
- **Z-Wave**: Secure mesh networking
- **WiFi**: High-bandwidth devices (cameras, displays)
- **Bluetooth**: Mobile app integration

### Device Discovery & Management

```python
class IoTDevice(BaseModel):
    device_id: str
    device_type: DeviceType
    protocol: DeviceProtocol  
    capabilities: List[DeviceCapability]
    privacy_impact: PrivacyLevel
    last_seen: datetime
    health_status: DeviceHealth
```

---

## Security & Privacy

### Local-First Architecture

- Core home automation functions work without internet
- Voice processing happens locally when possible
- Device control commands don't require cloud services
- Personal data encrypted at rest

### Data Governance

```python
class DataRetentionPolicy(BaseModel):
    data_type: str
    retention_days: int
    encryption_required: bool
    local_only: bool = True
    deletion_method: DeletionMethod = DeletionMethod.SECURE_OVERWRITE
```

### Security Boundaries

- Device network isolated from general internet
- Regular security updates for all components
- Intrusion detection for unusual device behavior  
- Emergency shutdown capabilities

---

## Integration with CIRIS Ecosystem

### Relationship to Other CIRIS Repositories

**CIRISAgent Integration**:
- Can delegate general AI queries to CIRISAgent
- Shares ethical framework and type systems
- Uses CIRISAgent for natural language understanding

**CIRISMedical Separation**:
- NO health monitoring capabilities
- NO medical device integration
- NO biometric data beyond basic presence detection
- Clear referral path for health-related queries

### Shared Ethical Framework

Following CIRIS Covenant principles:
- **Beneficence**: Actively improve family wellbeing
- **Non-maleficence**: Prevent harm through secure design
- **Integrity**: Transparent operation and open source
- **Respect for Autonomy**: Family controls their data and devices
- **Justice**: Equitable access across socioeconomic levels
- **Transparency**: Clear explanation of all automated actions

---

## Common Patterns & Anti-Patterns

### Good Patterns

```python
# Privacy-conscious device status
class PublicDeviceStatus(BaseModel):
    online: bool
    battery_level: Optional[int] = None
    last_update: datetime

# Family-centric automation
class HouseholdSchedule(BaseModel):  
    family_members: List[FamilyMember]
    shared_preferences: SharedPreferences
    private_schedules: Dict[str, PrivateSchedule] = Field(exclude=True)
```

### ❌ Anti-Patterns to Avoid

```python
# Don't create surveillance systems
class PersonTracker:  # NO - violates privacy
    def track_location(self, person_id: str) -> Location: ...

# Don't create medical monitoring  
class HealthMonitor:  # NO - belongs in CIRISMedical
    def analyze_vitals(self, biometrics: Dict) -> HealthAdvice: ...

# Don't create cloud dependencies for core functions
class CloudRequired:  # NO - violates local-first principle  
    def __init__(self):
        self.api_key = os.getenv("REQUIRED_CLOUD_API")  # Makes system unusable offline
```

---

## Getting Help & Contributing

### Development Resources

- **Issues**: https://github.com/CIRISAI/CIRISHome/issues
- **Privacy Questions**: See `docs/privacy-guide.md`
- **Device Integration**: See `docs/device-integration.md`
- **Voice Commands**: See `docs/voice-development.md`

### Contributing Guidelines

1. All contributions must align with domestic flourishing mission
2. Privacy-by-design required for all features
3. Local-first processing for core functionality
4. Type-safe implementations (no Dict[str, Any])
5. 85%+ test coverage for new features
6. Security review for device integrations

---

## Deployment Guide

### 3-Box Hardware Configuration **READY**

**Target Hardware Stack**:
1. **Home Assistant Yellow** - Main automation hub running CIRIS Engine 
2. **Jetson Orin Nano** - Local AI processing (STT, LLM, TTS, Vision)
3. **Voice PE Pucks** - Multi-modal voice interface throughout home

### Quick Start Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your Jetson IP and HA token

# 2. Deploy with Docker Compose  
export JETSON_NANO_IP=192.168.1.100
export HA_TOKEN=your-ha-long-lived-token
export RESPONSIBILITY_ACCEPTED=true

./deploy.sh  # Builds and tests the complete stack

# 3. Development mode
docker-compose --profile dev up  # Includes HA simulator + Jetson simulator
```

### Environment Configuration

```bash
# Required for deployment
JETSON_NANO_IP=192.168.1.100           # Your Jetson Orin Nano IP
HA_TOKEN=eyJhbGciOiJIUzI1NiIs...        # Home Assistant long-lived token
RESPONSIBILITY_ACCEPTED=true           # Safety acknowledgment

# CIRIS Engine LLM configuration (auto-configured)
OPENAI_API_KEY=jetson-local-llm        # Local identifier
OPENAI_API_BASE=http://192.168.1.100:11434/v1  # Jetson Ollama API
OPENAI_MODEL_NAME=llama-4-scout-int4   # Quantized model
```

### CI/CD Pipeline Status **GREEN**

- **Pre-commit Hooks**: Code quality, security scanning, formatting
- **GitHub Actions**: Matrix testing (Python 3.11/3.12), Docker builds, security audits
- **Test Coverage**: 63 tests passing, 0 failures
- **Safety Testing**: Medical sensor filtering, privacy compliance verified
- **Integration Testing**: Home Assistant, Wyoming protocol, Jetson GPU tested

### Production Readiness Checklist **COMPLETE**

- **Local AI Models**: 4-model stack on Jetson Orin Nano
  - **LLM**: llama-4-scout-int4 (~4GB) - Vision-capable language model
  - **STT**: whisper-large-v3 (~1.5GB) - Speech-to-text processing  
  - **Intent**: distilbert-intent (~250MB) - Intent classification
  - **TTS**: coqui-tts (~500MB) - Text-to-speech synthesis
  - **Total**: ~6.25GB + 0.5GB overhead = 6.75GB (fits in 8GB Jetson)
- **Voice Pipeline**: STT (Whisper) → Intent (DistilBERT) → LLM (Llama-4-Scout) → TTS (Coqui) → HA Services
- **Medical Filtering**: Comprehensive keyword detection for liability protection
- **Wisdom Integration**: Geo (OpenStreetMap), Weather (NOAA), Sensor (HA filtered)
- **Multi-modal Support**: Audio, Vision, Sensor fusion capabilities
- **Docker Compose**: Complete development and deployment stack
- **Security**: Pre-commit hooks, secrets detection, vulnerability scanning
- **Testing**: Ecosystem fixtures (HA, ESPHome, Wyoming, Jetson)

---

## Quality Standards

- **Mission Alignment**: Every feature serves family wellbeing
- **Privacy Protection**: Personal data stays in the home
- **Type Safety**: Zero untyped data structures  
- **Test Coverage**: 85% minimum for home automation logic
- **Response Time**: <500ms for local device commands
- **Offline Capability**: Core functions work without internet
- **Security**: Regular audits and secure-by-default design

---

**Remember**: We're building home intelligence that serves families, not systems that exploit them. Every line of code should make home life better while respecting privacy and autonomy.
