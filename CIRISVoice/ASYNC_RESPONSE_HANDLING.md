# CIRIS Voice Async Response Handling

## The Timeout Challenge

Voice assistants like Home Assistant expect quick responses (typically 3-10 seconds), but CIRIS agent processing can take longer due to:
- Complex reasoning chains
- Multiple thought iterations
- External tool calls
- Memory searches

## Solution Architecture

### 1. Immediate Acknowledgment Pattern

```python
class CIRISVoiceClient:
    def __init__(self, config):
        self.client = CIRISClient(base_url=config.api_url)
        self.quick_timeout = 2.0  # 2 seconds for initial response
        self.channel_id = f"voice_{config.channel_id}"

    async def send_message_async(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send message with immediate acknowledgment."""
        try:
            # First, try to get a quick response
            response = await asyncio.wait_for(
                self.client.agent.interact(
                    message=content,
                    channel_id=self.channel_id,
                    context=context
                ),
                timeout=self.quick_timeout
            )

            # Got response within timeout - return it
            return {
                "content": response.response,
                "message_id": response.message_id,
                "complete": True
            }

        except asyncio.TimeoutError:
            # Timeout - return acknowledgment and continue async
            message_id = str(uuid.uuid4())

            # Start async processing
            asyncio.create_task(
                self._process_async(message_id, content, context)
            )

            # Return immediate acknowledgment
            return {
                "content": "I'm thinking about that. Give me a moment...",
                "message_id": message_id,
                "complete": False
            }
```

### 2. Async Processing with Polling

```python
async def _process_async(self, message_id: str, content: str, context: dict):
    """Process message asynchronously and store result."""
    try:
        # Continue processing with longer timeout
        response = await self.client.agent.interact(
            message=content,
            channel_id=self.channel_id,
            context=context,
            timeout=30.0  # 30 second timeout for complex requests
        )

        # Store response for later retrieval
        await self._store_response(message_id, response.response)

        # Optional: Send notification if supported
        await self._notify_response_ready(message_id)

    except Exception as e:
        logger.error(f"Async processing failed: {e}")
        await self._store_response(message_id, "I had trouble with that request.")

async def check_response(self, message_id: str) -> Optional[str]:
    """Check if async response is ready."""
    return await self._get_stored_response(message_id)
```

### 3. Wyoming Bridge Integration

```python
class WyomingEventHandler:
    def __init__(self, config):
        self.ciris_client = CIRISVoiceClient(config)
        self.pending_responses = {}  # Track pending async responses

    async def handle_transcript(self, transcript: str) -> str:
        """Handle voice input with async fallback."""

        # Check if this is a follow-up query
        if self._is_followup_query(transcript):
            return await self._handle_followup(transcript)

        # Send to CIRIS
        response = await self.ciris_client.send_message_async(transcript)

        if response["complete"]:
            # Got immediate response
            return response["content"]
        else:
            # Store pending response info
            self.pending_responses[response["message_id"]] = {
                "original_query": transcript,
                "timestamp": time.time()
            }

            # Return acknowledgment
            return response["content"]

    def _is_followup_query(self, transcript: str) -> bool:
        """Detect if user is asking for a pending response."""
        followup_phrases = [
            "are you done",
            "do you have an answer",
            "what's the response",
            "are you still thinking",
            "any update"
        ]
        transcript_lower = transcript.lower()
        return any(phrase in transcript_lower for phrase in followup_phrases)

    async def _handle_followup(self, transcript: str) -> str:
        """Handle follow-up query for pending responses."""

        # Check most recent pending response
        if not self.pending_responses:
            return "I'm not currently working on anything."

        # Get most recent pending
        recent_id = max(
            self.pending_responses.keys(),
            key=lambda k: self.pending_responses[k]["timestamp"]
        )

        # Check if response is ready
        response = await self.ciris_client.check_response(recent_id)

        if response:
            # Clean up and return response
            del self.pending_responses[recent_id]
            return response
        else:
            # Still processing
            elapsed = time.time() - self.pending_responses[recent_id]["timestamp"]
            if elapsed < 30:
                return "I'm still thinking about that. Just a bit longer..."
            else:
                # Too long - might have failed
                del self.pending_responses[recent_id]
                return "I'm having trouble with that request. Could you try asking again?"
```

### 4. WebSocket Alternative (Better Solution)

```python
class CIRISWebSocketClient:
    """WebSocket-based client for real-time responses."""

    async def stream_interact(self, content: str, voice_callback):
        """Stream response chunks as they become available."""

        async with self.client.agent.stream_interact(
            message=content,
            channel_id=self.channel_id
        ) as stream:

            # Send initial acknowledgment
            await voice_callback("Let me think about that...")

            # Accumulate response
            full_response = []
            async for chunk in stream:
                if chunk.type == "thought_update":
                    # Optional: Provide thinking indicators
                    await voice_callback("Hmm...")
                elif chunk.type == "partial_response":
                    # Stream partial responses
                    full_response.append(chunk.content)
                    if chunk.is_sentence_end:
                        # Speak complete sentences as they arrive
                        sentence = " ".join(full_response)
                        await voice_callback(sentence)
                        full_response = []

            # Final response if any remainder
            if full_response:
                await voice_callback(" ".join(full_response))
```

### 5. Response Storage Options

```python
# Option 1: In-memory cache
class InMemoryResponseCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds

    async def store(self, message_id: str, response: str):
        self.cache[message_id] = {
            "response": response,
            "timestamp": time.time()
        }

        # Schedule cleanup
        asyncio.create_task(self._cleanup_old(message_id))

    async def get(self, message_id: str) -> Optional[str]:
        if message_id in self.cache:
            return self.cache[message_id]["response"]
        return None

    async def _cleanup_old(self, message_id: str):
        await asyncio.sleep(self.ttl)
        self.cache.pop(message_id, None)

# Option 2: Redis for multi-instance
class RedisResponseCache:
    def __init__(self, redis_url: str, ttl_seconds=300):
        self.redis = aioredis.from_url(redis_url)
        self.ttl = ttl_seconds

    async def store(self, message_id: str, response: str):
        await self.redis.setex(
            f"voice_response:{message_id}",
            self.ttl,
            response
        )

    async def get(self, message_id: str) -> Optional[str]:
        response = await self.redis.get(f"voice_response:{message_id}")
        return response.decode() if response else None
```

## Complete Voice Flow Example

```python
# 1. User says: "What's the weather like in Paris next week?"
# 2. CIRIS needs to call weather API (takes 5+ seconds)

async def handle_voice_interaction(transcript: str):
    # Quick check if CIRIS can respond immediately
    try:
        response = await asyncio.wait_for(
            ciris_client.interact(transcript),
            timeout=2.0
        )
        return response.content  # "The weather in Paris next week..."

    except asyncio.TimeoutError:
        # Start async processing
        task_id = await ciris_client.start_async_task(transcript)

        # Return immediate acknowledgment
        return "I'm checking the weather forecast for Paris. Ask me again in a moment."

# 3. User says: "Do you have that weather information?"
async def handle_followup():
    result = await ciris_client.check_pending_tasks()

    if result.ready:
        return result.content  # "The weather in Paris next week will be..."
    else:
        return "I'm still gathering that information. Just a few more seconds..."
```

## Configuration Recommendations

```yaml
# voice_config.yaml
ciris:
  # Quick response for immediate feedback
  quick_timeout_ms: 2000

  # Full timeout for complex requests
  full_timeout_ms: 30000

  # Response caching
  cache:
    type: "memory"  # or "redis"
    ttl_seconds: 300

  # Acknowledgment messages
  messages:
    thinking: "Let me think about that..."
    still_thinking: "I'm still working on that..."
    timeout: "That's taking longer than expected. Please try again."

  # Follow-up detection
  followup_phrases:
    - "are you done"
    - "what's the answer"
    - "do you have a response"
```

## Best Practices

1. **Always acknowledge quickly**: Return something within 2-3 seconds
2. **Use natural acknowledgments**: Make them contextual to the query
3. **Support follow-ups**: Train users to ask for updates
4. **Set expectations**: Tell users if something will take time
5. **Implement timeouts**: Don't leave users waiting indefinitely
6. **Cache responses**: Allow retrieval of recent responses
7. **Handle failures gracefully**: Provide helpful error messages

## Alternative Approaches

### 1. Callback URL Pattern
```python
# Register a callback URL for when response is ready
await ciris_client.interact_async(
    message=transcript,
    callback_url="http://voice-bridge/response-ready",
    callback_token=secure_token
)
```

### 2. Server-Sent Events (SSE)
```python
# Stream updates as they happen
async for event in ciris_client.interact_sse(transcript):
    if event.type == "thinking":
        yield "Still processing..."
    elif event.type == "response":
        yield event.data
```

### 3. Two-Phase Interaction
```python
# Phase 1: Quick assessment
assessment = await ciris_client.assess_complexity(transcript)
if assessment.is_simple:
    return await ciris_client.interact(transcript)
else:
    # Phase 2: Async processing
    return await handle_complex_query(transcript)
```

## Integration with Home Assistant

For Home Assistant specifically, you might want to:

1. Use the conversation agent's `async_process` with a short timeout
2. Store pending responses in a custom integration
3. Expose a service to check pending responses
4. Use notifications when responses are ready

```python
# In Home Assistant custom component
async def async_process(self, text: str) -> str:
    """Process text with CIRIS."""

    # Try quick response
    response = await self.ciris.send_message_async(text)

    if response["complete"]:
        return response["content"]
    else:
        # Store in HA state
        self.hass.states.async_set(
            "ciris.pending_response",
            "thinking",
            {"message_id": response["message_id"]}
        )

        # Return acknowledgment
        return response["content"]
```

This approach ensures voice interactions remain responsive while supporting CIRIS's more complex processing capabilities.
