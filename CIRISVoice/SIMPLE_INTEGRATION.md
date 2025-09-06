# CIRIS Voice Simple Integration Guide

## Overview

CIRIS may take up to 60 seconds to process complex queries. We'll configure a straightforward synchronous integration with extended timeouts to accommodate this.

## Updated Integration

### 1. Fix the API Endpoint

Update `ciris_client.py` to use the correct endpoint:

```python
class CIRISClient:
    def __init__(self, config):
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.timeout = 58  # Just under our 60s timeout
        self.channel_id = f"voice_{config.channel_id}"

    async def send_message(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "message": content,  # Changed from "content"
                "channel_id": self.channel_id,
                "context": context or {"source": "wyoming_voice"}
            }

            try:
                async with session.post(
                    f"{self.api_url}/v1/agent/interact",  # Correct endpoint
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Extract response from the correct field
                        return {
                            "content": data["data"]["response"],
                            "message_id": data["data"]["message_id"],
                            "processing_time": data["data"]["processing_time_ms"]
                        }
                    else:
                        error = await response.text()
                        logger.error(f"CIRIS API error: {error}")
                        return {"content": "I'm having trouble processing that request."}
            except asyncio.TimeoutError:
                logger.warning("CIRIS API timeout after 58 seconds")
                return {"content": "That took too long to process. Please try again."}
            except Exception as e:
                logger.error(f"CIRIS API exception: {e}")
                return {"content": "I'm experiencing technical difficulties."}
```

### 2. Configure CIRIS API Timeout

The CIRIS API timeout can be configured in multiple ways:

#### Option A: Environment Variable
```bash
export CIRIS_API_INTERACTION_TIMEOUT=55.0
```

#### Option B: Configuration File
```yaml
# In your CIRIS config.yaml or api_config.yaml
api:
  interaction_timeout: 55.0  # Seconds to wait for agent response
```

#### Option C: API Adapter Configuration
```python
# When initializing the API adapter
api_config = APIConfig(
    host="0.0.0.0",
    port=8080,
    interaction_timeout=55.0  # Extended timeout
)
```

### 3. Wyoming Bridge Configuration

```yaml
# config.yaml
wyoming:
  host: "0.0.0.0"
  port: 10300

ciris:
  api_url: "http://localhost:8080"
  api_key: ""  # Or use env var
  timeout: 58  # Just under our 60s timeout
  channel_id: "wyoming_default"

stt:
  provider: "openai"  # or "google"
  model: "whisper-1"
  language: "en"

tts:
  provider: "openai"  # or "google"
  voice: "nova"
  model: "tts-1"
```

### 4. Home Assistant Configuration

#### Via UI (Recommended):
1. Go to **Settings > Voice Assistants**
2. Create new pipeline: "CIRIS Extended Timeout"
3. Configure:
   - Conversation agent: Home Assistant
   - Speech-to-text: Google Cloud (or your STT)
   - Text-to-speech: Google Cloud (or your TTS)
   - **Advanced Settings**: Set timeout to 60 seconds

#### Via configuration.yaml:
```yaml
# Enable extended pipeline
assist_pipeline:
  pipelines:
    - name: "CIRIS Extended Pipeline"
      conversation_engine: "homeassistant"
      conversation_language: "en-US"
      stt_engine: "google_cloud"
      stt_language: "en-US"
      tts_engine: "google_cloud"
      tts_language: "en-US"
      tts_voice: "en-US-Wavenet-F"
      # Note: timeout may need to be set via UI or WebSocket

# For Wyoming satellite
wyoming:
  - host: localhost
    port: 10300
```

#### For ESPHome devices:
```yaml
voice_assistant:
  microphone: mic_id
  speaker: speaker_id
  conversation_timeout: 60s  # Extended timeout
  on_thinking:
    - light.turn_on:
        id: status_led
        effect: "pulsing"
```

## Testing

### Quick Response Test
```
User: "What time is it?"
Expected: Response within 2-3 seconds
```

### Medium Complexity Test
```
User: "Tell me a joke"
Expected: Response within 5-10 seconds
```

### Complex Query Test
```
User: "Explain quantum computing in simple terms"
Expected: Response within 15-40 seconds
```

### Deep Reasoning Test
```
User: "What's the meaning of life?"
Expected: Response within 30-55 seconds
```

### Timeout Test
```
User: "Something that would take > 60 seconds"
Expected: Timeout message at ~58 seconds
```

## Benefits of This Approach

1. **Simple**: No complex async handling needed
2. **Reliable**: Works within existing HA constraints
3. **Patient**: 60 seconds allows CIRIS to think deeply
4. **No polling**: Direct request-response pattern
5. **Easy to debug**: Straightforward flow

## User Experience Note

With a 60-second timeout, users will see:
- The voice assistant LED/indicator will remain active (spinning/pulsing) for up to a minute
- No intermediate "I'm thinking" messages - just patient waiting
- When CIRIS responds, it will speak the complete, well-considered answer
- This mirrors human conversation where complex questions naturally take time to answer thoughtfully

## Important Configuration Notes

### Network Timeouts
Some network components may have shorter timeouts:
- **Nginx/reverse proxy**: Increase `proxy_read_timeout` to 65s
- **Docker networks**: Usually fine with 60s
- **Firewalls**: Check for connection timeout settings

### CIRIS API Server
Make sure the CIRIS API server is configured to handle long timeouts:
```python
# In CIRIS API configuration
uvicorn:
  timeout_keep_alive: 65  # Longer than our 60s timeout
```

## Future Enhancements

If needed later, we can add:
- Response caching for common queries
- WebSocket streaming for real-time feedback
- Further timeout extensions if needed

But for now, this simple synchronous approach with 60-second timeout will handle virtually all use cases effectively.
