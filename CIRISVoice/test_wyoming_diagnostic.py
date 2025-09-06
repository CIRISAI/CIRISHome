#!/usr/bin/env python3
"""Diagnostic Wyoming bridge to debug Home Assistant connection issues."""
import asyncio
import json
import logging

from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class DiagnosticHandler(AsyncEventHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_count = 0
        self.connection_start = asyncio.get_event_loop().time()

        # Pre-create info event exactly like faster-whisper
        self.wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="diagnostic-stt",
                    description="Diagnostic STT for testing",
                    attribution=Attribution(name="CIRIS Test", url="https://ciris.ai"),
                    installed=True,
                    models=[
                        AsrModel(
                            name="test-model-v1",
                            description="Test model for diagnostics",
                            languages=["en"],
                            attribution=Attribution(name="CIRIS Test", url="https://ciris.ai"),
                            installed=True,
                        )
                    ],
                )
            ]
        )
        self.wyoming_info_event = self.wyoming_info.event()

        # Log the exact JSON that will be sent
        logger.info("=== PRE-CREATED INFO EVENT ===")
        logger.info(f"Event type: {self.wyoming_info_event.type}")
        logger.info(f"Event data: {json.dumps(self.wyoming_info_event.data, indent=2)}")

    async def write_event(self, event: Event) -> None:
        """Override to log what we're sending."""
        logger.info(f">>> SENDING: {event.type}")
        if hasattr(event, "data"):
            logger.info(f">>> DATA: {json.dumps(event.data, indent=2)}")
        await super().write_event(event)
        logger.info(">>> SENT SUCCESSFULLY")

    async def handle_event(self, event: Event) -> bool:
        self.event_count += 1
        elapsed = asyncio.get_event_loop().time() - self.connection_start

        logger.info(f"=== EVENT #{self.event_count} at {elapsed:.3f}s ===")
        logger.info(f"<<< RECEIVED: {event.type}")
        if hasattr(event, "data"):
            logger.info(f"<<< DATA: {event.data}")

        # Check connection state
        if hasattr(self.writer, "is_closing"):
            logger.info(f"Connection state: is_closing={self.writer.is_closing()}")

        if Describe.is_type(event.type):
            logger.info(">>> Describe detected, sending pre-created Info event")
            try:
                await self.write_event(self.wyoming_info_event)
                logger.info(">>> Returning True to keep connection open")
                return True
            except Exception as e:
                logger.error(f">>> Error sending info: {e}", exc_info=True)
                return False

        if Ping.is_type(event.type):
            logger.info(">>> Ping detected, sending Pong")
            await self.write_event(Pong().event())
            return True

        logger.warning(f">>> Unknown event type: {event.type}")
        return True

    async def disconnect(self) -> None:
        elapsed = asyncio.get_event_loop().time() - self.connection_start
        logger.info(f"=== DISCONNECT after {elapsed:.3f}s and {self.event_count} events ===")

        # Check if writer was closed by remote
        if hasattr(self.writer, "is_closing"):
            logger.info(f"Writer is_closing: {self.writer.is_closing()}")

        # Get exception info if available
        if hasattr(self.writer, "exception"):
            exc = self.writer.exception()
            if exc:
                logger.error(f"Writer exception: {exc}")


async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    logger.info("=== DIAGNOSTIC WYOMING SERVER STARTED ===")
    logger.info("Listening on port 10300")
    logger.info("This server logs everything for debugging")

    def create_handler(reader, writer):
        peer = writer.get_extra_info("peername")
        logger.info(f"=== NEW CONNECTION from {peer} ===")
        return DiagnosticHandler(reader, writer)

    await server.run(create_handler)


if __name__ == "__main__":
    asyncio.run(main())
