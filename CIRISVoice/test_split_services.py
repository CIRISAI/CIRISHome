#!/usr/bin/env python3
"""Test if Home Assistant expects separate STT and TTS services."""
import asyncio
import logging

from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)


class STTOnlyHandler(AsyncEventHandler):
    """STT only service."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="ciris-stt",
                    description="CIRIS Speech to Text",
                    attribution=Attribution(
                        name="CIRIS AI",
                        url="https://ciris.ai",
                    ),
                    installed=True,
                    models=[
                        AsrModel(
                            name="ciris-google",
                            description="Google Cloud Speech Recognition",
                            attribution=Attribution(
                                name="CIRIS AI",
                                url="https://ciris.ai",
                            ),
                            installed=True,
                            languages=["en"],
                        )
                    ],
                )
            ],
        )

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            _LOGGER.info("Sending STT-only info")
            await self.write_event(self.wyoming_info.event())
            return True

        _LOGGER.warning(f"Unexpected event: {event}")
        return True


class TTSOnlyHandler(AsyncEventHandler):
    """TTS only service."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = Info(
            tts=[
                TtsProgram(
                    name="ciris-tts",
                    description="CIRIS Text to Speech",
                    attribution=Attribution(
                        name="CIRIS AI",
                        url="https://ciris.ai",
                    ),
                    installed=True,
                    voices=[
                        TtsVoice(
                            name="en-US-google",
                            description="Google Cloud TTS Voice",
                            attribution=Attribution(
                                name="CIRIS AI",
                                url="https://ciris.ai",
                            ),
                            installed=True,
                            languages=["en"],
                        )
                    ],
                )
            ],
        )

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            _LOGGER.info("Sending TTS-only info")
            await self.write_event(self.wyoming_info.event())
            return True

        _LOGGER.warning(f"Unexpected event: {event}")
        return True


async def main():
    """Run test server."""
    logging.basicConfig(level=logging.INFO)

    import sys

    service_type = sys.argv[1] if len(sys.argv) > 1 else "stt"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 10301

    _LOGGER.info(f"Starting {service_type} service on port {port}")

    handler_class = STTOnlyHandler if service_type == "stt" else TTSOnlyHandler
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{port}")
    await server.run(handler_class)


if __name__ == "__main__":
    asyncio.run(main())
