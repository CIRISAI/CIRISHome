# Voice PE + CIRIS Quick Start

## 5-Minute Setup for Your Hardware

### Prerequisites
- Home Assistant Yellow is running
- Voice PE pucks are connected and discovered
- CIRIS API is accessible from your network
- OpenAI API key (or Google Cloud credentials)

### Step 1: Add CIRIS Wyoming Bridge

#### Option A: Quick Test (Terminal & SSH)
```bash
# In HA Terminal & SSH add-on
cd /config
git clone [your-repo] ciris-voice
cd ciris-voice/CIRISVoice

# Install dependencies
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
nano config.yaml  # Add your CIRIS URL and API keys

# Run
python -m src.bridge
```

#### Option B: Proper Add-on (Recommended)
1. Copy `docker-addon` folder to `/addons/ciris-wyoming/`
2. Go to **Settings → Add-ons → Add-on Store**
3. Click ⋮ menu → **Check for updates**
4. Install "CIRIS Wyoming Bridge" from Local add-ons

### Step 2: Configure in Home Assistant UI

1. **Add Wyoming Integration**:
   - Settings → Devices & Services → Add Integration
   - Search "Wyoming Protocol"
   - Host: `localhost` (or your HA IP if external)
   - Port: `10300`

2. **Create Voice Pipeline**:
   - Settings → Voice assistants → Add Assistant
   - Name: **CIRIS**
   - Wake word: **OK Nabu** (or change it)
   - Speech-to-text: **Wyoming**
   - Text-to-speech: **Wyoming**
   - **Timeout: 60 seconds** ← Important!

3. **Assign to Voice PE Pucks**:
   - Settings → Devices & Services → ESPHome
   - Click your Voice PE device
   - Configure → Voice assistant → Select **CIRIS**

### Step 3: Test It!

Stand near your Voice PE puck:

1. **Quick Test** (2-3 seconds):
   ```
   You: "OK Nabu"
   *LED spins blue*
   You: "What time is it?"
   CIRIS: "It's 3:45 PM"
   ```

2. **Deep Thought Test** (30-50 seconds):
   ```
   You: "OK Nabu"
   *LED spins blue*
   You: "What's the meaning of life?"
   *LED pulses blue for up to 60 seconds*
   CIRIS: "The meaning of life, from my perspective..."
   ```

### LED Indicators on Voice PE

- 🔵 **Spinning Blue**: Listening to you
- 🔵 **Pulsing Blue**: CIRIS is thinking (up to 60s)
- 🟢 **Green Flash**: Success
- 🔴 **Red**: Error occurred
- ⚪ **White**: Wake word detected

### Quick Troubleshooting

**"Entity not found" error**:
- Restart the Wyoming add-on
- Check if port 10300 is accessible

**Puck doesn't respond**:
- Check if assigned to CIRIS pipeline
- Try "OK Nabu" clearly
- Check puck firmware (2023.12.0+)

**Timeout too quick**:
- Verify pipeline timeout is 60s
- Check CIRIS API timeout is 55s
- Check add-on config shows 58s

**No voice output**:
- Verify TTS credentials
- Test with HA's built-in Piper TTS first
- Check speaker volume on puck

### Configuration Checklist

✅ Wyoming Bridge running on port 10300
✅ CIRIS API URL is correct
✅ API timeout set to 58 seconds
✅ Pipeline timeout set to 60 seconds
✅ Voice PE assigned to CIRIS pipeline
✅ Wake word enabled ("OK Nabu")

### Next Steps

1. **Place pucks around your home** - They have good range
2. **Customize wake word** - Change from "OK Nabu" to "Hey CIRIS"
3. **Add context** - Each room can have its own channel
4. **Create automations** - Trigger actions from voice

Your Voice PE pucks are now patient listeners for CIRIS's thoughtful responses!
