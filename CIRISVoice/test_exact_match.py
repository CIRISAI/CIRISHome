#!/usr/bin/env python3
"""
Exact match of faster-whisper Wyoming structure for testing.
"""
import asyncio
import logging
from functools import partial

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

__version__ = "1.0.0"

_LOGGER = logging.getLogger(__name__)


class TestHandler(AsyncEventHandler):
    """Event handler matching faster-whisper structure."""

    def __init__(self, wyoming_info: Info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.audio_buffer = bytearray()
        self.is_recording = False

    async def handle_event(self, event: Event) -> bool:
        """Handle a single event from a client."""
        if event is None:
            return True

        if event.type == "describe":
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            return True

        if AudioStart.is_type(event.type):
            self.is_recording = True
            self.audio_buffer = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            if self.is_recording:
                chunk = AudioChunk.from_event(event)
                self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            self.is_recording = False
            if len(self.audio_buffer) > 0:
                await self.write_event(Transcript(text="Test transcript").event())
            self.audio_buffer = bytearray()
            return True

        return True


async def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.info("Starting test Wyoming server matching faster-whisper structure")

    # Create wyoming info exactly like faster-whisper
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="test-whisper",
                description="Test Whisper transcription",
                attribution=Attribution(
                    name="Test Author",
                    url="https://github.com/test/",
                ),
                installed=True,
                # Note: faster-whisper uses version here, but our Wyoming doesn't support it
                models=[
                    AsrModel(
                        name="test-model",
                        description="test-model",
                        attribution=Attribution(
                            name="Test",
                            url="https://test.com",
                        ),
                        installed=True,
                        languages=["en"],
                        # Note: faster-whisper uses version here too
                    )
                ],
            )
        ],
    )

    server = AsyncServer.from_uri("tcp://0.0.0.0:10303")
    _LOGGER.info("Ready on port 10303")

    # Use partial like faster-whisper does
    await server.run(
        partial(
            TestHandler,
            wyoming_info,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
