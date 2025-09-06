# CIRIS Voice Deployment for Home Assistant Yellow + Voice PE

This guide walks through deploying CIRIS Voice on your Home Assistant Yellow with Voice Assistant Preview Edition pucks.

## Your Hardware Setup

- **Home Assistant Yellow**: Raspberry Pi CM4-based hub
- **Voice PE Pucks**: ESP32-S3 based voice satellites
- **Wyoming Protocol**: Built-in support for local voice processing

## Architecture Overview

```
Voice PE Puck → (Wake Word) → HA Yellow → Wyoming Bridge → CIRIS API
                                  ↓
                             Google/OpenAI
                              STT & TTS
```

## Step 1: Deploy Wyoming Bridge on HA Yellow

Since HA Yellow runs Home Assistant OS, we'll deploy the Wyoming bridge as an add-on.

### Option A: Run on the Yellow directly (Recommended)

1. **Enable SSH on HA Yellow**:
   - Install "Terminal & SSH" add-on from Add-on Store
   - Start the add-on and open Web UI

2. **Install dependencies**:
```bash
# In Terminal & SSH
apk add python3 py3-pip git
cd /config
git clone [your-ciris-repo] ciris-voice
cd ciris-voice/CIRISVoice
```

3. **Configure**:
```bash
cp config.example.yaml config.yaml
nano config.yaml
```

Update with your settings:
```yaml
wyoming:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 10300

ciris:
  api_url: "http://[YOUR_CIRIS_IP]:8080"
  api_key: ""
  timeout: 58
  channel_id: "ha_yellow"

# Use OpenAI for quality or Google for speed
stt:
  provider: "openai"
  openai_api_key: "sk-..."  # From secrets.yaml

tts:
  provider: "openai"
  model: "tts-1"
  voice: "nova"
```

### Option B: Docker Add-on (Production)

Create a local add-on:

1. **Create add-on structure**:
```bash
# On your development machine
mkdir -p ciris-wyoming-addon
cd ciris-wyoming-addon
```

2. **Create `config.yaml`**:
```yaml
name: "CIRIS Wyoming Bridge"
description: "Connect CIRIS to Home Assistant Voice"
version: "1.0.0"
slug: "ciris_wyoming"
init: false
arch:
  - aarch64  # For HA Yellow
ports:
  10300/tcp: 10300
ports_description:
  10300/tcp: "Wyoming Protocol"
options:
  ciris_url: "http://localhost:8080"
  ciris_timeout: 58
  stt_provider: "openai"
  tts_provider: "openai"
schema:
  ciris_url: str
  ciris_timeout: int
  stt_provider: list(openai|google)
  tts_provider: list(openai|google)
```

3. **Create `Dockerfile`**:
```dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python and dependencies
RUN apk add --no-cache python3 py3-pip git

# Copy source
COPY CIRISVoice /app
WORKDIR /app

# Install requirements
RUN pip3 install -r requirements.txt

# Copy run script
COPY run.sh /
RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
```

4. **Create `run.sh`**:
```bash
#!/usr/bin/with-contenv bashio

# Read config from add-on options
CIRIS_URL=$(bashio::config 'ciris_url')
CIRIS_TIMEOUT=$(bashio::config 'ciris_timeout')

# Export for Python
export CIRIS_API_URL=$CIRIS_URL
export CIRIS_API_INTERACTION_TIMEOUT=$CIRIS_TIMEOUT

# Get API keys from HA secrets
export OPENAI_API_KEY=$(bashio::secrets 'openai_api_key')

# Run bridge
cd /app
python3 -m src.bridge
```

## Step 2: Configure Voice Assistants

1. **In Home Assistant UI**:
   - Go to **Settings → Voice Assistants**
   - Click **Add Assistant**

2. **Create CIRIS Pipeline**:
   - Name: "CIRIS"
   - Conversation agent: "Home Assistant" (for now)
   - Speech-to-text: "Whisper" (if local) or your STT
   - Text-to-speech: "Piper" (if local) or your TTS
   - Wake word: "Hey Jarvis" (or your choice)

3. **Add Wyoming Integration**:
   - Go to **Settings → Devices & Services**
   - Click **Add Integration**
   - Search for "Wyoming Protocol"
   - Host: `localhost` (if on Yellow) or bridge IP
   - Port: `10300`

4. **Update Pipeline with Wyoming**:
   - Return to Voice Assistants
   - Edit CIRIS pipeline
   - Select Wyoming for STT and/or TTS
   - **Important**: Set timeout to 60 seconds

## Step 3: Configure Voice PE Pucks

Your Voice PE pucks should auto-discover the HA Yellow. To assign them to CIRIS:

1. **Find your puck**:
   - Go to **Settings → Devices & Services → ESPHome**
   - Click on your Voice PE device

2. **Configure the puck**:
   - Click **Configure**
   - Under "Voice Assistant", select "CIRIS"
   - The puck will now use your CIRIS pipeline

3. **Test wake word**:
   - Say "Hey Jarvis" (or your wake word)
   - The LED ring should activate
   - Ask: "What time is it?"
   - Wait for response (up to 60 seconds for complex queries)

## Step 4: Optimize for Your Setup

### Network Configuration

Since CIRIS might be running elsewhere:

```yaml
# In configuration.yaml
http:
  # Allow Wyoming bridge connection
  cors_allowed_origins:
    - http://localhost:10300
```

### Voice PE LED Feedback

The pucks will show:
- **Blue spinning**: Listening
- **Blue pulsing**: Processing (up to 60 seconds)
- **Green**: Success
- **Red**: Error

### Reduce Latency

1. **Run CIRIS on the same network** as HA Yellow
2. **Use wired connection** for HA Yellow if possible
3. **Consider local STT/TTS**:
   - Whisper for STT (runs on Yellow)
   - Piper for TTS (very fast)

## Step 5: Testing

### Basic Test
```
You: "Hey Jarvis"
Puck: *blue spinning*
You: "What time is it?"
CIRIS: "It's 3:45 PM"
```

### Extended Timeout Test
```
You: "Hey Jarvis"
Puck: *blue spinning*
You: "What's the meaning of life?"
Puck: *blue pulsing for up to 60 seconds*
CIRIS: "The meaning of life is..."
```

## Troubleshooting

### Puck doesn't respond to wake word
- Check if puck is assigned to CIRIS pipeline
- Verify wake word is configured
- Check puck firmware is updated

### "Processing" times out
- Verify Wyoming bridge is running: `curl http://localhost:10300`
- Check CIRIS API is accessible from Yellow
- Look at logs: `ha logs` or in Terminal & SSH

### Audio quality issues
- Voice PE pucks have good mics, but check placement
- Avoid near speakers or air vents
- Test with different TTS voices

## Advanced: Multiple Pucks

Each puck can have different settings:

```yaml
# Example: Room-specific responses
ciris:
  channels:
    bedroom_puck: "voice_bedroom"
    kitchen_puck: "voice_kitchen"
    office_puck: "voice_office"
```

CIRIS can then give context-aware responses based on which room you're in.

## Performance Tips

1. **Pre-warm CIRIS**: Keep the API active
2. **Use local caching**: For common queries
3. **Optimize TTS**: Google is faster but OpenAI sounds better
4. **Monitor latency**: Check logs for response times

## Complete Test Sequence

1. **Quick response** (2-5s):
   - "What's the weather?"
   - "Turn on the lights"

2. **Medium response** (10-20s):
   - "Tell me a joke"
   - "What's on my calendar?"

3. **Deep thinking** (30-50s):
   - "Explain quantum computing"
   - "What's the meaning of life?"

Your Voice PE pucks will patiently wait (LED pulsing) while CIRIS thinks!

## Next Steps

- Set up multiple pucks in different rooms
- Create automations triggered by voice
- Integrate with your smart home devices
- Train family members on the extended timeout

The combination of HA Yellow + Voice PE + CIRIS gives you a powerful, patient, and thoughtful voice assistant!
