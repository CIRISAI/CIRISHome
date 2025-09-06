#!/usr/bin/env python3
"""Minimal Wyoming server to test Home Assistant connection."""
import asyncio
import logging

from wyoming.info import AsrModel, AsrProgram, Describe, Info, TtsProgram, TtsVoice
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MinimalHandler(AsyncEventHandler):
    async def handle_event(self, event):
        logger.info(f"Event: {type(event).__name__}")

        if isinstance(event, Describe):
            return [
                Info(
                    asr=[
                        AsrProgram(
                            name="test-asr",
                            description="Test ASR",
                            attribution="Test",
                            installed=True,
                            models=[
                                AsrModel(
                                    name="test-model",
                                    description="Test model",
                                    attribution="Test",
                                    installed=True,
                                    languages=["en"],
                                )
                            ],
                        )
                    ],
                    tts=[
                        TtsProgram(
                            name="test-tts",
                            description="Test TTS",
                            attribution="Test",
                            installed=True,
                            voices=[
                                TtsVoice(
                                    name="test-voice",
                                    description="Test voice",
                                    attribution="Test",
                                    installed=True,
                                    languages=["en"],
                                )
                            ],
                        )
                    ],
                )
            ]

        if isinstance(event, Ping):
            return [Pong()]

        return []


async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10301")
    logger.info("Starting minimal Wyoming server on port 10301")
    await server.run(lambda r, w: MinimalHandler(r, w))


if __name__ == "__main__":
    asyncio.run(main())
