#!/usr/bin/env python3
"""
Minimal Wyoming ASR service to test Home Assistant acceptance.
Based on faster-whisper structure but simplified.
"""
import asyncio
import json
import logging

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class MinimalHandler(AsyncEventHandler):
    """Minimal event handler for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer = bytearray()
        self.is_recording = False

    async def handle_event(self, event: Event) -> bool:
        """Handle a single event from a client."""
        _LOGGER.info(f"Received event type: {event.type}")

        # Describe event - send info
        if event.type == "describe":
            # Create minimal ASR-only info
            info = Info(
                asr=[
                    AsrProgram(
                        name="minimal-asr",
                        description="Minimal ASR for testing",
                        attribution=Attribution(name="Test", url="https://example.com"),
                        installed=True,
                        models=[
                            AsrModel(
                                name="test-model",
                                description="Test model",
                                attribution=Attribution(name="Test", url="https://example.com"),
                                installed=True,
                                languages=["en"],
                            )
                        ],
                    )
                ]
            )

            # Get the event and log what we're sending
            info_event = info.event()
            _LOGGER.info(f"Sending info: {json.dumps(info_event.data, indent=2)}")

            await self.write_event(info_event)
            _LOGGER.info("Info sent successfully")
            return True

        # Handle transcribe request
        if Transcribe.is_type(event.type):
            _LOGGER.info("Transcribe request received")
            return True

        # Handle audio events
        if AudioStart.is_type(event.type):
            _LOGGER.info("Audio start")
            self.is_recording = True
            self.audio_buffer = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            if self.is_recording:
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.info("Audio stop")
            self.is_recording = False

            # Send a simple transcript
            if len(self.audio_buffer) > 0:
                await self.write_event(Transcript(text="Test transcript").event())
                _LOGGER.info("Sent test transcript")

            self.audio_buffer = bytearray()
            return True

        _LOGGER.warning(f"Unhandled event: {event.type}")
        return True


async def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _LOGGER.info("Starting minimal Wyoming server on port 10301")

    # Start server
    server = AsyncServer.from_uri("tcp://0.0.0.0:10301")
    _LOGGER.info("Server created, starting...")

    await server.run(MinimalHandler)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down")
