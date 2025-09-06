#!/usr/bin/env python3
"""Test Wyoming service that keeps connection open."""
import asyncio
import logging

from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersistentHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_count = 0
        # Pre-create info event
        self.wyoming_info_event = Info(
            asr=[
                AsrProgram(
                    name="persistent-stt",
                    description="Test STT that keeps connection open",
                    attribution=Attribution(name="Test", url="https://test.local"),
                    installed=True,
                    models=[
                        AsrModel(
                            name="test-model",
                            description="Test model",
                            languages=["en"],
                            attribution=Attribution(name="Test", url="https://test.local"),
                            installed=True,
                        )
                    ],
                )
            ]
        ).event()

    async def handle_event(self, event: Event) -> bool:
        self.event_count += 1
        logger.info(f"=== EVENT #{self.event_count} ===")
        logger.info(f"Type: {event.type}")

        if Describe.is_type(event.type):
            logger.info("Received Describe, sending Info")
            await self.write_event(self.wyoming_info_event)
            logger.info("Info sent, returning True to keep connection open")
            return True  # Keep connection open

        if Ping.is_type(event.type):
            logger.info("Received Ping, sending Pong")
            await self.write_event(Pong().event())
            return True  # Keep connection open

        logger.info(f"Unknown event type: {event.type}, keeping connection open")
        return True  # Always keep connection open

    async def disconnect(self) -> None:
        logger.info(f"=== DISCONNECT after {self.event_count} events ===")


async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    logger.info("Persistent test server listening on port 10300")
    logger.info("This server will ALWAYS return True to keep connections open")
    await server.run(PersistentHandler)


if __name__ == "__main__":
    asyncio.run(main())
