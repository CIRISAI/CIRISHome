#!/usr/bin/env python3
"""Exact mimic of wyoming-faster-whisper for testing."""
import asyncio
import logging
from functools import partial

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class TestHandler(AsyncEventHandler):
    """Exact copy of FasterWhisperEventHandler pattern."""

    def __init__(self, wyoming_info: Info, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            # Just consume audio chunks
            return True

        if AudioStop.is_type(event.type):
            # Send a dummy transcript
            await self.write_event(Transcript(text="Test transcription").event())
            _LOGGER.debug("Completed request")
            return False  # This closes connection after transcription

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True


async def main() -> None:
    """Main entry point."""
    logging.basicConfig(level=logging.DEBUG)

    # Create wyoming info exactly like faster-whisper
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper transcription",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper",
                ),
                installed=True,
                models=[
                    AsrModel(
                        name="tiny-int8",
                        description="Tiny model (39M parameters)",
                        attribution=Attribution(
                            name="Guillaume Klein",
                            url="https://github.com/guillaumekln/faster-whisper",
                        ),
                        installed=True,
                        languages=["en"],
                    )
                ],
            )
        ],
    )

    # Create server
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Test mimic server started on port 10300")

    # Run server with handler
    await server.run(partial(TestHandler, wyoming_info))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
