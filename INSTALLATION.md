# CIRISHome Installation Guide

**Complete setup guide for CIRISHome multi-modal AI home automation**

Transform your home into an intelligent, privacy-first smart home using:

- **Home Assistant Yellow** (automation hub)
- **Jetson Orin Nano** (local AI processing)
- **Voice PE Pucks** (voice satellites)

**Total Setup Time: ~35 minutes + 30 minutes model download**

---

## Prerequisites

### Hardware Required

- **Home Assistant Yellow** with SD card (32GB+)
- **Jetson Orin Nano** with SD card (64GB+)
- **Voice PE Puck(s)** with ESPHome firmware
- **Network**: All devices on same local network
- **Optional**: Nest cameras with WebRTC/go2rtc integration

### Software Requirements

- Home Assistant OS (latest)
- JetPack 6.2 for Jetson Orin Nano
- ESPHome firmware for Voice PE pucks

---

## Step 1: Home Assistant Yellow Setup

**Duration: 10 minutes**

### 1.1 Initial Setup

1. **Flash Home Assistant OS**:
   - Download from [home-assistant.io](https://www.home-assistant.io/installation/)
   - Flash to SD card using Raspberry Pi Imager
   - Insert SD card and power on HA Yellow

2. **Complete onboarding**:
   - Navigate to `http://homeassistant.local:8123`
   - Create your admin account
   - Configure location and units

### 1.2 Install HACS

1. Go to **Settings** → **Add-ons** → **Add-on Store**
2. Install **HACS** (Home Assistant Community Store)
3. Restart Home Assistant

### 1.3 Install CIRISHome Add-on

1. **Add Repository**:
   - Go to **Settings** → **Add-ons** → **Add-on Store** → **⋮** → **Repositories**
   - Add: `https://github.com/CIRISAI/CIRISHome`

2. **Install CIRISHome Add-on**:
   - Find "CIRISHome Agent" in the store
   - Click **Install** (may take 5-10 minutes)

3. **Configure Add-on**:

   ```yaml
   jetson_nano_ip: "192.168.1.100" # Your Jetson's IP
   ha_token: "eyJ0eXAi..." # Long-lived access token
   responsibility_accepted: true # Required for home control
   ```

4. **Start Add-on**: Click **Start** and check logs

---

## Step 2: Jetson Orin Nano Setup

**Duration: 45 minutes (15 minutes active + 30 minutes model download)**

### 2.1 Initial Setup

1. **Flash JetPack 6.2**:
   - Download from [NVIDIA Developer](https://developer.nvidia.com/jetpack)
   - Flash to SD card (64GB+ recommended)
   - Insert SD card and power on Jetson

2. **Complete Ubuntu setup**:
   - Follow on-screen setup wizard
   - Create user account (`nvidia` recommended)
   - Connect to same network as HA Yellow

### 2.2 Install CIRISHome

1. **SSH into Jetson**:

```bash
ssh nvidia@<jetson-ip-address>
```

2. **Clone CIRISHome repository**:

```bash
git clone --recurse-submodules https://github.com/CIRISAI/CIRISHome.git
cd CIRISHome
```

3. **Download AI models** (runs in background):

```bash
./scripts/download_models.sh
```

**Models Downloaded:**

- **Llama-4-Scout-INT4** (~4GB) - Local LLM with vision
- **Whisper-Large-v3** (~1.5GB) - Speech-to-text
- **Coqui TTS** (~500MB) - Text-to-speech
- **DistilBERT Intent** (~250MB) - Intent classification
- **Total**: ~6.25GB

### 2.3 Configure Integration

1. **Get Home Assistant details**:
   - **URL**: `http://<yellow-ip>:8123`
   - **Token**: Settings → My → Long-lived access tokens → Create

2. **Start CIRISHome services**:

```bash
# Set environment variables
export HOME_ASSISTANT_URL="http://<yellow-ip>:8123"
export HOME_ASSISTANT_TOKEN="<your-token>"
export I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY="true"

# Start with Jetson optimizations
python main.py --jetson-mode --enable-vision --enable-audio
```

### 2.4 Verify AI Models

```bash
# Test model loading
python test_local_models.py
```

**Expected output**:

```
Testing Local Models on Jetson Orin Nano
GPU: Jetson Orin Nano
GPU Memory: 8.0GB
llm: llama-4-scout-int4 (~4GB)
stt: whisper-large-v3 (~1.5GB)
tts: coqui-tts (~500MB)
intents: distilbert-intent (~250MB)
100% Local Processing - No cloud dependencies!
```

---

## Step 3: Voice PE Puck Setup

**Duration: 5 minutes per puck**

### 3.1 Puck Preparation

1. **Flash ESPHome firmware** to Voice PE puck
2. **Connect to WiFi**: Same network as HA Yellow and Jetson
3. **Note puck IP address** from router/ESPHome logs

### 3.2 Add to Home Assistant

1. **Settings** → **Devices & Services** → **ESPHome** → **Add Integration**
2. Enter puck IP address
3. **Adopt device** when prompted

### 3.3 Create Voice Pipeline

1. **Settings** → **Voice Assistants** → **Add Assistant**
2. **Configure pipeline**:
   - **Name**: "CIRIS Home"
   - **Language**: English
   - **Timeout**: 60 seconds (for complex queries)
   - **STT**: Wyoming (points to Jetson)
   - **LLM**: Wyoming (points to Jetson)
   - **TTS**: Wyoming (points to Jetson)

### 3.4 Assign Pipeline to Puck

1. **Settings** → **Devices & Services** → Find your Voice PE puck
2. **Configure** → **Voice Pipeline** → Select "CIRIS Home"
3. **Save configuration**

---

## Step 4: Network Integration

**Duration: 2 minutes**

### 4.1 Network Topology

```
Voice PE Puck(s) ──WiFi──┐
                          │
                    HA Yellow ──Ethernet── Jetson Orin Nano
                          │
Nest Cameras ──WiFi──────┘
```

### 4.2 Service Communication

- **Voice PE → HA Yellow**: Wyoming protocol (port 10300)
- **HA Yellow → Jetson**: CIRISHome API (port 8080)
- **Cameras → HA Yellow**: WebRTC/go2rtc (port 8554)

### 4.3 Test Connection

**Test voice pipeline**:

1. Say to puck: **"Hey Assistant, hello"**
2. **Expected flow**:
   - Puck captures audio → HA Yellow
   - HA Yellow sends to Jetson STT → Llama-4-Scout → TTS
   - Response plays on puck

---

## Step 5: Camera Integration (Optional)

**Duration: 5 minutes if you have WebRTC/go2rtc setup**

### 5.1 Configure Camera URLs

If you have Nest cameras via WebRTC/go2rtc:

1. **Edit configuration** in `/addons/ciris-home/config.yaml`:

```yaml
environment:
  WEBRTC_CAMERA_URLS: "front_door:rtsp://127.0.0.1:8554/front_door,driveway:rtsp://127.0.0.1:8554/driveway"
  GO2RTC_SERVER_URL: "http://127.0.0.1:8554"
```

2. **Restart CIRISHome addon**

### 5.2 Test Camera Integration

**Voice command**: "Hey Assistant, what do you see on the front door camera?"

**Expected response**: AI analyzes camera feed and describes what it sees

---

## Verification & Testing

### Basic Functionality Test

1. **Voice Response**:

   ```
   You: "Hey Assistant, what time is it?"
   AI: "It's currently 3:45 PM."
   ```

2. **Home Control**:

   ```
   You: "Hey Assistant, turn on the living room lights"
   AI: "I've turned on the living room lights for you."
   ```

3. **Camera Analysis** (if configured):
   ```
   You: "Hey Assistant, is anyone at the front door?"
   AI: [Analyzes camera] "I don't see anyone at the front door right now."
   ```

### Performance Verification

**On Jetson Orin Nano**:

```bash
# Check Grace status
python tools/grace_home status

# Monitor GPU usage
nvidia-smi

# Check model performance
python tools/grace_home dev
```

**Expected performance**:

- **LLM Response**: 15-25 tokens/second
- **STT Latency**: 2-5 seconds
- **TTS Latency**: 1-2 seconds
- **Vision Analysis**: 3-8 seconds per image

---

## Troubleshooting

### Common Issues

#### Voice PE Puck Not Responding

```bash
# Check puck connectivity
ping <puck-ip>

# Verify Wyoming service
curl http://<jetson-ip>:10300/info
```

#### Jetson Models Not Loading

```bash
# Check GPU memory
nvidia-smi

# Verify model files
ls -la models/
python test_local_models.py
```

#### Home Assistant Integration Issues

```bash
# Check addon logs
docker logs ciris-home

# Verify API connectivity
curl http://<jetson-ip>:8080/health
```

### Performance Optimization

#### Jetson Super Mode (if available)

```bash
# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### Memory Management

```bash
# Monitor memory usage
free -h
nvidia-smi

# Restart services if needed
sudo systemctl restart ciris-home
```

---

## Security Considerations

### Network Security

- **Firewall**: Configure firewall to block external access to ports 8080, 10300
- **VPN Access**: Use VPN for remote access instead of port forwarding
- **Local Only**: All AI processing happens locally - nothing sent to cloud

### Data Privacy

- **Voice Data**: Processed locally, deleted after response
- **Camera Data**: Analyzed locally, never transmitted externally
- **Model Data**: All AI models stored locally on Jetson
- **No Telemetry**: Zero data collection or transmission

### Update Management

```bash
# Update CIRISHome
cd CIRISHome
git pull --recurse-submodules

# Update models (if needed)
./scripts/download_models.sh --update
```

---

## Advanced Configuration

### Custom Voice Commands

Edit `/addons/ciris-home/config/voice_intents.yaml` to add custom intents.

### Camera Event Automation

See `examples/ha_automations.yaml` for sample automations using local event detection.

### Multi-Room Voice

Add multiple Voice PE pucks and assign the same "CIRIS Home" pipeline to each.

---

## Support

### Documentation

- **Main Documentation**: [README.md](README.md)
- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **Example Automations**: [examples/ha_automations.yaml](examples/ha_automations.yaml)

### Grace Development Companion

```bash
# Check system health
python tools/grace_home status

# Morning development session
python tools/grace_home morning

# Test specific components
python tools/grace_home vision
python tools/grace_home audio
python tools/grace_home cameras
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/CIRISAI/CIRISHome/issues)
- **Community**: [Home Assistant Community](https://community.home-assistant.io/)

---

## What You Get

### 🎯 **Complete Local AI Home**

- **Voice Control**: Natural language commands via Voice PE pucks
- **Vision Intelligence**: Camera analysis with Llama-4-Scout
- **Smart Automation**: Context-aware home control
- **100% Privacy**: Everything processed locally on Jetson

### **Sample Interactions**

```
"Hey Assistant, good morning"
→ AI: "Good morning! It's 7:30 AM, weather is sunny, and I've turned on your morning lights."

"Is anyone at the front door?"
→ AI: [Analyzes camera] "I see a delivery person placing a package at your front door."

"Turn off all the lights and lock the doors"
→ AI: "I've turned off all lights and locked all doors. Have a good night!"
```

**Welcome to the future of private, intelligent home automation!**
