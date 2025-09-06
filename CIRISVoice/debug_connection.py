#!/usr/bin/env python3
"""Debug what happens after info response."""
import asyncio
import logging

from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DebugHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = Info(
            asr=[
                AsrProgram(
                    name="debug-stt",
                    description="Debug STT",
                    attribution=Attribution(name="Debug", url="https://debug.local"),
                    installed=True,
                    models=[
                        AsrModel(
                            name="debug-model",
                            description="Debug model",
                            languages=["en"],
                            attribution=Attribution(name="Debug", url="https://debug.local"),
                            installed=True,
                        )
                    ],
                )
            ]
        ).event()
        self.event_count = 0

    async def handle_event(self, event: Event) -> bool:
        self.event_count += 1
        logger.info(
            f"Event #{self.event_count}: type={event.type}, data={event.data if hasattr(event, 'data') else 'N/A'}"
        )

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            logger.info("Sent info, waiting for next event...")
            return True

        # Log any other event type
        logger.warning(f"Unexpected event type: {event.type}")
        return True

    async def disconnect(self) -> None:
        logger.info(f"Disconnect called after {self.event_count} events")


async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    logger.info("Debug server listening on port 10300")
    await server.run(DebugHandler)


if __name__ == "__main__":
    asyncio.run(main())
