#!/usr/bin/env python3
"""Test script for voice interaction with CIRIS"""

import asyncio
import logging

from wyoming.asr import Transcript
from wyoming.client import AsyncClient


async def test_text_interaction():
    async with AsyncClient.from_uri("tcp://localhost:10300") as client:
        test_phrases = [
            "Hello CIRIS",
            "What's the weather like?",
            "Turn on the living room lights",
            "Why should I trust you with my home?",
            "Explain your ethical principles",
        ]
        for phrase in test_phrases:
            print(f"\nYou: {phrase}")
            await client.send_event(Transcript(text=phrase))
            response = await client.receive_event()
            if hasattr(response, "text"):
                print(f"CIRIS: {response.text}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_text_interaction())
