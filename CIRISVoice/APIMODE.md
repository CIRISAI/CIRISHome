# CIRIS SDK Integration

This Wyoming bridge now uses the official CIRIS SDK for improved reliability and features.

## SDK Features

- **Endpoint**: Automatically uses `/v1/agent/interact`
- **Authentication**: Handles API key or username/password auth
- **Timeout**: Configured for 58 seconds (under HA's 60s limit)
- **Retries**: Disabled for voice (single attempt)
- **Type Safety**: Full request/response validation

## Message Flow

1. Voice input is transcribed to text
2. SDK sends to CIRIS with proper formatting:
   - `message`: The transcribed text
   - `channel_id`: Voice-specific channel (e.g., `voice_wyoming_default`)
   - `context`: Includes source, profile, language, session info
3. SDK waits up to 58 seconds for response
4. Response is converted to speech

## Response Format

The SDK returns a structured response:
```json
{
  "content": "CIRIS's response text",
  "message_id": "unique-id",
  "state": "WORK",
  "processing_time": 15000
}
```

## Error Handling

The SDK handles various error scenarios:
- **Timeout** (58s): "That took too long to process. Please try again."
- **API Error**: "I'm having trouble understanding that request."
- **Network Error**: "I'm experiencing technical difficulties."

All errors are logged with appropriate context for debugging.
