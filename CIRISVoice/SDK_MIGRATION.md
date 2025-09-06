# CIRIS Voice SDK Migration Guide

## Overview

The CIRIS Voice Wyoming Bridge currently uses a custom HTTP client to communicate with the CIRIS API. This guide outlines how to migrate it to use the official CIRIS SDK for improved maintainability and feature support.

## Current Implementation Analysis

### Current Architecture
- **Custom Client**: `ciris_client.py` uses `aiohttp` directly
- **Endpoints Used**:
  - `POST /v1/messages` - Send messages (custom endpoint, not in current API)
  - `GET /v1/status` - Get status/last response
- **Authentication**: Bearer token via API key
- **Async**: Uses asyncio/aiohttp for async operations

### Issues with Current Implementation
1. Uses non-existent `/v1/messages` endpoint (API uses `/v1/agent/interact`)
2. Payload structure doesn't match API expectations
3. Manual HTTP handling instead of SDK abstractions
4. No automatic retry, rate limiting, or error handling

## SDK Migration Plan

### 1. Install CIRIS SDK
```bash
pip install ciris-sdk
```

### 2. Replace Custom Client with SDK

Create a new `ciris_sdk_client.py`:

```python
import asyncio
from typing import Optional, Dict, Any
import logging
from ciris_sdk import CIRISClient
from ciris_sdk.exceptions import CIRISError, CIRISTimeoutError

logger = logging.getLogger(__name__)

class CIRISVoiceClient:
    def __init__(self, config):
        # Initialize SDK client
        self.client = CIRISClient(
            base_url=config.api_url,
            api_key=config.api_key,
            timeout=config.timeout
        )
        self.channel_id = f"voice_{config.channel_id}"  # Prefix for voice channels
        self.profile = config.profile

    async def initialize(self):
        """Initialize the client and authenticate."""
        try:
            # If using username/password instead of API key
            if hasattr(self.client.auth, 'login'):
                await self.client.auth.login('voice_user', 'voice_password')
        except CIRISError as e:
            logger.error(f"Failed to authenticate with CIRIS: {e}")
            raise

    async def send_message(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message to CIRIS and get response."""
        try:
            # Merge voice context with provided context
            full_context = {
                "source": "wyoming_voice",
                "profile": self.profile,
                **(context or {})
            }

            # Use the SDK's agent interact method
            response = await self.client.agent.interact(
                message=content,
                channel_id=self.channel_id,
                context=full_context
            )

            # Convert SDK response to expected format
            return {
                "content": response.response,
                "message_id": response.message_id,
                "state": response.state,
                "processing_time": response.processing_time_ms
            }

        except CIRISTimeoutError:
            logger.error("CIRIS API timeout")
            return {"content": "I need more time to think about that. Please try again."}
        except CIRISError as e:
            logger.error(f"CIRIS API error: {e}")
            return {"content": "I'm having trouble processing that request."}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"content": "I'm experiencing technical difficulties."}

    async def get_response(self, message_id: Optional[str] = None) -> Optional[str]:
        """Get agent status or last response."""
        try:
            # Get agent status
            status = await self.client.agent.get_status()

            # Extract last response from status if available
            # Note: The actual structure depends on API implementation
            if hasattr(status, 'last_interaction'):
                return status.last_interaction.response

            # Alternative: Get conversation history
            history = await self.client.agent.get_history(
                channel_id=self.channel_id,
                limit=1
            )

            if history.messages:
                last_message = history.messages[-1]
                if last_message.is_agent:
                    return last_message.content

        except CIRISError as e:
            logger.error(f"Failed to get response: {e}")

        return None

    async def close(self):
        """Clean up client resources."""
        await self.client.close()
```

### 3. Update Bridge Integration

Modify `bridge.py` to use the SDK client:

```python
# In bridge.py
from ciris_sdk_client import CIRISVoiceClient

class WyomingEventHandler:
    def __init__(self, config):
        # ... existing code ...
        self.ciris_client = CIRISVoiceClient(config)

    async def initialize(self):
        """Initialize services."""
        await self.ciris_client.initialize()

    async def handle_transcript(self, transcript: str) -> str:
        """Process user transcript through CIRIS."""
        response = await self.ciris_client.send_message(
            content=transcript,
            context={
                "voice_metadata": {
                    "language": self.config.language,
                    "voice_profile": self.config.profile
                }
            }
        )
        return response["content"]
```

### 4. Configuration Updates

Update configuration to support SDK features:

```python
class CIRISConfig:
    def __init__(self):
        # Existing config
        self.api_url = os.getenv("CIRIS_API_URL", "http://localhost:8080")
        self.api_key = os.getenv("CIRIS_API_KEY", "")

        # New SDK-specific config
        self.username = os.getenv("CIRIS_USERNAME", "voice_user")
        self.password = os.getenv("CIRIS_PASSWORD", "voice_password")
        self.retry_attempts = int(os.getenv("CIRIS_RETRY_ATTEMPTS", "3"))
        self.enable_streaming = os.getenv("CIRIS_ENABLE_STREAMING", "false").lower() == "true"
```

## Advanced Features with SDK

### 1. WebSocket Streaming (Future)
```python
# Stream responses for real-time voice feedback
async def stream_response(self, content: str):
    async with self.client.agent.stream_interact(
        message=content,
        channel_id=self.channel_id
    ) as stream:
        async for chunk in stream:
            yield chunk.content
```

### 2. Memory Integration
```python
# Store voice interaction context in agent memory
async def store_voice_context(self, user_id: str, preferences: Dict[str, Any]):
    await self.client.memory.store(
        node_type="voice_preference",
        node_id=f"voice_pref_{user_id}",
        attributes=preferences
    )
```

### 3. Telemetry Tracking
```python
# Track voice interaction metrics
async def track_interaction(self, duration_ms: int, success: bool):
    await self.client.telemetry.record_metric(
        metric_name="voice.interaction.duration",
        value=duration_ms,
        tags={"success": str(success).lower()}
    )
```

## Benefits of SDK Migration

1. **Correct API Usage**: Uses proper `/v1/agent/interact` endpoint
2. **Automatic Features**:
   - Retry logic with exponential backoff
   - Rate limiting
   - Request/response validation
   - Proper error types
3. **Type Safety**: Full TypeScript/Python type hints
4. **Future Features**: Easy access to new API capabilities
5. **Maintenance**: SDK updates automatically include API changes

## Migration Steps

1. **Test SDK Connection**:
   ```python
   # test_sdk_connection.py
   import asyncio
   from ciris_sdk import CIRISClient

   async def test():
       client = CIRISClient(base_url="http://localhost:8080")
       response = await client.agent.interact(
           message="Hello from voice",
           channel_id="voice_test"
       )
       print(f"Response: {response.response}")

   asyncio.run(test())
   ```

2. **Gradual Migration**:
   - Keep existing client as fallback
   - Add feature flag to switch between implementations
   - Test with real voice interactions
   - Monitor logs for issues

3. **Update Documentation**:
   - Update APIMODE.md to reflect SDK usage
   - Document new configuration options
   - Add troubleshooting guide

## Compatibility Notes

- The SDK expects `message` field, not `content` in requests
- Channel IDs should follow naming convention (e.g., `voice_home_assistant`)
- The SDK handles authentication automatically after initial setup
- Response structure is standardized across all SDK methods

## Example Voice Flow with SDK

```python
# Complete voice interaction flow
async def handle_voice_command(transcript: str, user_id: str):
    client = CIRISVoiceClient(config)

    try:
        # Initialize if needed
        await client.initialize()

        # Send voice command
        response = await client.send_message(
            content=transcript,
            context={
                "user_id": user_id,
                "input_method": "voice",
                "timestamp": datetime.now().isoformat()
            }
        )

        # Return voice response
        return response["content"]

    finally:
        await client.close()
```

This migration will provide a more robust, maintainable, and feature-rich integration between CIRIS Voice and the CIRIS agent.
