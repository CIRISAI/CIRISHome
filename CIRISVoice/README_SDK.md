# CIRIS Voice Wyoming Bridge (SDK Version)

This Wyoming bridge enables Home Assistant voice assistants to interact with CIRIS using the official SDK. It supports extended 60-second timeouts for complex queries.

## Features

- ✅ **Official CIRIS SDK** - Type-safe, reliable API integration
- ✅ **58-second timeout** - Allows CIRIS to think deeply
- ✅ **Session tracking** - Maintains conversation context
- ✅ **Multiple STT/TTS providers** - OpenAI and Google Cloud
- ✅ **Home Assistant compatible** - Works with Voice PE

## Requirements

- Python 3.9+
- Home Assistant with Voice Assistant support
- CIRIS API endpoint (local or remote)
- STT/TTS API keys (OpenAI or Google Cloud)

## Quick Start

1. **Clone and setup**:
```bash
cd CIRISVoice
./setup.sh
```

2. **Configure** (edit `config.yaml`):
```yaml
ciris:
  api_url: "http://localhost:8080"
  api_key: "your-api-key"  # Or use env var
  timeout: 58  # Just under HA's 60s
  channel_id: "wyoming_default"
```

3. **Test locally**:
```bash
source venv/bin/activate
python -m src.bridge
```

4. **Configure Home Assistant**:
   - Go to Settings → Voice Assistants
   - Create new pipeline "CIRIS Extended"
   - Set timeout to 60 seconds
   - Add Wyoming protocol pointing to `your-ip:10300`

## Configuration

### CIRIS API Settings

The SDK client connects to your CIRIS instance:

```yaml
ciris:
  api_url: "http://localhost:8080"  # Your CIRIS API
  api_key: ""  # Optional API key auth
  timeout: 58  # Extended timeout (seconds)
  channel_id: "wyoming_default"  # Voice channel identifier
  profile: "default"  # Voice profile
  language: "en-US"  # Language code
```

### STT/TTS Providers

#### OpenAI (Recommended for quality):
```yaml
stt:
  provider: "openai"
  openai_api_key: "sk-..."  # Or use OPENAI_API_KEY env
  model: "whisper-1"

tts:
  provider: "openai"
  model: "tts-1"
  voice: "nova"  # alloy, echo, fable, onyx, nova, shimmer
```

#### Google Cloud (Recommended for speed):
```yaml
stt:
  provider: "google"
  google_credentials_path: "/path/to/key.json"
  google_language_code: "en-US"

tts:
  provider: "google"
  google_voice_name: "en-US-Wavenet-F"
```

## How It Works

1. **Voice Input** → Wyoming protocol → STT → Text
2. **CIRIS SDK** sends text with 58s timeout
3. **CIRIS thinks** for up to 58 seconds
4. **Response** → TTS → Wyoming protocol → Voice Output

## Monitoring

The bridge logs all interactions with timing:

```
2025-01-09 10:23:45 - INFO - Transcribed: What's the meaning of life?
2025-01-09 10:24:12 - INFO - CIRIS responded in 27.3s: The meaning of life...
```

## Troubleshooting

### "Module ciris_sdk not found"
The CIRIS SDK may not be published yet. Install manually when available:
```bash
pip install ciris-sdk
```

### Timeout errors
- Verify CIRIS API timeout is set to 55+ seconds
- Check network latency between components
- Ensure no proxy/firewall timeouts < 60s

### No audio output
- Check TTS provider credentials
- Verify audio format compatibility
- Test with simple queries first

## Advanced Usage

### Custom Session Context

The SDK client maintains session context automatically. You can extend it:

```python
# In bridge.py
response = await self.ciris_client.send_message(
    text,
    context={
        "location": "living_room",
        "user_preferences": {...}
    }
)
```

### Performance Tuning

For faster responses:
- Use Google Cloud TTS (lower latency)
- Place CIRIS API on same network
- Pre-cache common responses

## Production Deployment

### Systemd Service

```bash
sudo cp ciris-wyoming.service /etc/systemd/system/
sudo systemctl enable ciris-wyoming
sudo systemctl start ciris-wyoming
```

### Docker (Coming Soon)

```dockerfile
FROM python:3.11-slim
# ... Dockerfile in progress
```

## Contributing

Issues and PRs welcome! Please test with extended timeouts before submitting.

## License

Same as CIRIS project.
