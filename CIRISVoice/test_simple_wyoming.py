#!/usr/bin/env python3
"""
Ultra-simple Wyoming server that mimics a working implementation.
Based on wyoming-faster-whisper structure.
"""
import asyncio
import logging

from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class Handler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create info that matches working implementations
        self.wyoming_info = Info(
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

    async def handle_event(self, event: Event) -> bool:
        """Handle a single event from a client."""
        if event is None:
            _LOGGER.warning("Received None event")
            return True

        _LOGGER.debug("Received event: %s", event)

        # Describe
        if event.type == "describe":
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent info")
            return True

        # Not handled
        _LOGGER.warning("Unexpected event: %s", event)
        return True


async def main() -> None:
    """Main function."""
    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.info("Starting test Wyoming server")

    # Start server
    server = AsyncServer.from_uri("tcp://0.0.0.0:10301")
    _LOGGER.info("Listening on port 10301")

    await server.run(Handler)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
