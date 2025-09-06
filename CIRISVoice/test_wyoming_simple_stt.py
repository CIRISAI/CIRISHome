#!/usr/bin/env python3
"""
Ultra-minimal Wyoming STT server for testing.
Based directly on faster-whisper structure.
"""
import asyncio
import logging

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class SimpleHandler(AsyncEventHandler):
    """Simple event handler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create wyoming info
        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="test-stt",
                    description="Test STT Service",
                    attribution=Attribution(
                        name="Test",
                        url="https://test.com",
                    ),
                    installed=True,
                    models=[
                        AsrModel(
                            name="test-model",
                            description="Test model",
                            attribution=Attribution(
                                name="Test",
                                url="https://test.com",
                            ),
                            installed=True,
                            languages=["en"],
                        )
                    ],
                )
            ],
        )

        # Pre-create event like faster-whisper does
        self.wyoming_info_event = wyoming_info.event()
        self.audio_buffer = bytearray()
        self.is_recording = False

    async def handle_event(self, event: Event) -> bool:
        """Handle a single event from a client."""
        _LOGGER.debug("Received event: %s", event)

        if event.type == "describe":
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            _LOGGER.debug("Transcribe event")
            return True

        if AudioStart.is_type(event.type):
            _LOGGER.debug("Audio start")
            self.is_recording = True
            self.audio_buffer = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            if self.is_recording:
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stop")
            self.is_recording = False
            if len(self.audio_buffer) > 0:
                await self.write_event(Transcript(text="Test transcript").event())
            self.audio_buffer = bytearray()
            return True

        _LOGGER.warning("Unknown event: %s", event)
        return True


async def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.info("Starting test Wyoming STT server on port 10302")

    server = AsyncServer.from_uri("tcp://0.0.0.0:10302")
    await server.run(SimpleHandler)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
