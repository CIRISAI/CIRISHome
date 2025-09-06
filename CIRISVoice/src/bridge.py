import asyncio
import logging
import os
import sys

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

try:
    from wyoming.info import TtsProgram, TtsVoice
except ImportError:
    # Older wyoming versions don't have TTS types
    TtsProgram = None
    TtsVoice = None
from wyoming.ping import Ping, Pong

from .ciris_sdk_client import CIRISClient
from .config import Config
from .stt_service import create_stt_service
from .tts_service import create_tts_service

__version__ = "1.0.11"

logger = logging.getLogger(__name__)


class CIRISWyomingHandler(AsyncEventHandler):
    def __init__(self, reader, writer, config: Config = None):
        super().__init__(reader, writer)
        self.config = config or Config.from_yaml("config.yaml")
        self.stt_service = create_stt_service(self.config.stt)
        self.tts_service = create_tts_service(self.config.tts)
        self.ciris_client = CIRISClient(self.config.ciris)
        self.audio_buffer = bytearray()
        self.is_recording = False
        self._initialized = False
        self._connection_info = writer.get_extra_info("peername")
        self.last_transcript = None  # Store last transcript for Synthesize event
        self._expecting_synthesize = False  # Track if we're waiting for Synthesize
        self._ciris_response = None  # Store CIRIS response for TTS
        logger.info(f"Handler created for connection from {self._connection_info}")

        # Pre-create wyoming info like faster-whisper does
        self.wyoming_info = self._get_info()

        # Create event and try to remove empty arrays
        info_event = self.wyoming_info.event()

        # Manually create event without empty arrays if possible
        if hasattr(info_event, "data") and isinstance(info_event.data, dict):
            # Only include non-empty arrays
            cleaned_data = {k: v for k, v in info_event.data.items() if not (isinstance(v, list) and len(v) == 0)}

            # Create custom event with only ASR data
            from wyoming.event import Event

            self.wyoming_info_event = Event(type="info", data=cleaned_data)
        else:
            self.wyoming_info_event = info_event

    @property
    def closed(self):
        """Check if connection is closed."""
        return self.writer.is_closing() if hasattr(self.writer, "is_closing") else False

    async def disconnect(self):
        """Handle disconnection cleanup."""
        logger.info("=== DISCONNECT ===")
        logger.info(f"Connection closed from {self._connection_info}")
        logger.info(f"Total events received: {getattr(self, '_event_count', 0)}")
        logger.info(f"Handler initialized: {self._initialized}")

        # Log if we were expecting a Synthesize event
        if self._expecting_synthesize:
            logger.warning("Connection closed while waiting for Synthesize event!")
            logger.warning("This suggests Home Assistant is not sending TTS to this bridge.")
            logger.warning("Check Home Assistant voice pipeline configuration.")

        # Log if connection was closed by remote
        if hasattr(self.writer, "is_closing"):
            logger.info(f"Writer is_closing: {self.writer.is_closing()}")

        if self._initialized and hasattr(self.ciris_client, "voice_client"):
            try:
                await self.ciris_client.voice_client.close()
            except Exception as e:
                logger.error(f"Error closing CIRIS client: {e}")

    async def _ensure_initialized(self):
        """Initialize CIRIS client on first use."""
        if not self._initialized:
            try:
                await self.ciris_client.voice_client.initialize()
                self._initialized = True
                logger.info("CIRIS client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize CIRIS client: {e}")
                raise

    async def handle_event(self, event) -> bool:
        import time

        start_time = time.time()

        # Comprehensive event logging
        logger.info(f"=== EVENT #{getattr(self, '_event_count', 0) + 1} ===")
        self._event_count = getattr(self, "_event_count", 0) + 1

        # Log all event attributes
        event_attrs = {}
        for attr in dir(event):
            if not attr.startswith("_"):
                try:
                    value = getattr(event, attr)
                    if not callable(value):
                        event_attrs[attr] = str(value)
                except Exception:
                    pass

        logger.info(f"Event type: {type(event).__name__}")
        logger.info(f"Event attributes: {event_attrs}")

        # Check if we have raw event data
        if hasattr(event, "__dict__"):
            logger.info(f"Event dict: {event.__dict__}")

        # Log event type in detail
        if hasattr(event, "type"):
            logger.info(f"Event.type field: {event.type}")
        if hasattr(event, "data"):
            logger.info(f"Event.data field: {event.data}")

        # Handle None event
        if event is None:
            logger.warning("Received None event")
            return True

        # Check for Describe event using string comparison as backup
        if event.type == "describe" or (hasattr(event, "is_type") and Describe.is_type(event.type)):
            logger.info("Describe event detected, sending info response")

            # Debug what we're sending
            import json

            logger.info(f"Info event type: {self.wyoming_info_event.type}")
            if hasattr(self.wyoming_info_event, "data"):
                logger.info(f"Info data: {json.dumps(self.wyoming_info_event.data, indent=2)}")

            # Check if connection is still open
            if self.closed:
                logger.error("Connection already closed before sending info!")
                return False

            try:
                logger.info("About to write event...")
                await self.write_event(self.wyoming_info_event)
                logger.info("Info sent successfully")
                await asyncio.sleep(0.1)  # Give HA time to process
                logger.info("After sleep, connection still open")
            except ConnectionResetError:
                logger.error("Connection reset by peer while sending info")
                return False
            except Exception as e:
                logger.error(f"Error sending info: {type(e).__name__}: {e}")
                return False

            return True

        # Handle Ping events for health checks
        if Ping.is_type(event.type):
            logger.debug("Received ping, sending pong")
            await self.write_event(Pong().event())
            return True

        # Skip CIRIS initialization for pure STT mode
        # if not self._initialized:
        #     try:
        #         await self._ensure_initialized()
        #     except Exception as e:
        #         logger.error(f"Initialization failed: {e}", exc_info=True)
        #         # Continue anyway - Wyoming might just be probing

        if Transcribe.is_type(event.type):
            # Wyoming might send language preference
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                logger.debug(f"Language set to {transcribe.language}")
            return True

        if AudioStart.is_type(event.type):
            logger.debug("Audio recording started")
            self.is_recording = True
            self.audio_buffer = bytearray()
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self.is_recording:
                self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            logger.debug("Audio recording stopped")
            self.is_recording = False
            if len(self.audio_buffer) > 0:
                try:
                    # Don't send any events until we have actual content
                    # Home Assistant might close connection on early Transcript

                    # Track total processing time
                    total_start = asyncio.get_event_loop().time()

                    # Transcribe audio
                    stt_start = asyncio.get_event_loop().time()
                    text = await self.stt_service.transcribe(bytes(self.audio_buffer))
                    stt_time = asyncio.get_event_loop().time() - stt_start

                    if text:
                        logger.info(f"Transcribed in {stt_time:.1f}s: {text}")

                        # Pure STT mode - just send the transcript
                        await self.write_event(Transcript(text=text).event())
                        logger.info(f"Sent transcript: {text}")
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    # Don't try to send error messages if connection is broken
                    if not isinstance(e, (ConnectionResetError, BrokenPipeError)):
                        try:
                            await self.write_event(
                                Synthesize(text="I encountered an error processing your request.").event()
                            )
                        except (ConnectionResetError, BrokenPipeError):
                            logger.warning("Connection lost while sending error message")
            self.audio_buffer = bytearray()
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            logger.info(f"Received Synthesize event with text: {synthesize.text[:50]}...")

            try:
                # Check if this is from our transcript and we have CIRIS response ready
                if self._expecting_synthesize and self._ciris_response and synthesize.text == self.last_transcript:
                    logger.info("Using stored CIRIS response for TTS")

                    # Start audio stream immediately
                    await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())

                    # Stream silence while processing
                    silence_chunk = bytes(4800)  # 100ms of silence at 24kHz
                    stop_silence = asyncio.Event()

                    async def stream_silence():
                        while not stop_silence.is_set() and not self.closed:
                            try:
                                await self.write_event(
                                    AudioChunk(audio=silence_chunk, rate=24000, width=2, channels=1).event()
                                )
                                await asyncio.sleep(0.05)
                            except (ConnectionResetError, BrokenPipeError):
                                break

                    silence_task = asyncio.create_task(stream_silence())

                    try:
                        # Use the pre-calculated CIRIS response
                        response_text = self._ciris_response
                        logger.info(f"Synthesizing pre-calculated CIRIS response: {response_text[:50]}...")

                        # Stop silence immediately
                        stop_silence.set()
                        await silence_task

                        # Synthesize CIRIS response
                        audio_data = await self.tts_service.synthesize(response_text)

                        # Send audio chunks
                        chunk_size = 4800
                        for i in range(0, len(audio_data), chunk_size):
                            if self.closed:
                                break
                            chunk = audio_data[i : i + chunk_size]
                            await self.write_event(AudioChunk(audio=chunk, rate=24000, width=2, channels=1).event())

                        await self.write_event(AudioStop().event())
                        logger.info("Successfully sent CIRIS response as audio")

                    finally:
                        stop_silence.set()
                        if not silence_task.done():
                            silence_task.cancel()
                            try:
                                await silence_task
                            except asyncio.CancelledError:
                                pass
                else:
                    # Direct TTS synthesis (not from transcript)
                    logger.info("Direct TTS synthesis requested")
                    audio_data = await self.tts_service.synthesize(synthesize.text)

                    await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())

                    # Send audio chunks
                    chunk_size = 4800
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i : i + chunk_size]
                        await self.write_event(AudioChunk(audio=chunk, rate=24000, width=2, channels=1).event())

                    await self.write_event(AudioStop().event())

            except Exception as e:
                logger.error(f"TTS/CIRIS error: {e}")
                # Send empty audio to close the stream
                await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

            # Clear transcript and response after use
            self.last_transcript = None
            self._ciris_response = None
            self._expecting_synthesize = False
            return True

        # Log unknown event types
        logger.warning(f"Unhandled event type: {type(event).__name__}")
        return True  # Keep connection alive even for unknown events

    def _get_info(self):
        # Create info with ASR always
        info_dict = {
            "asr": [
                AsrProgram(
                    name="ciris",
                    description=f"CIRIS STT using {self.config.stt.provider}",
                    attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                    installed=True,
                    version=__version__,
                    models=[
                        AsrModel(
                            name="ciris-stt-v1",
                            description=f"{self.config.stt.provider} speech recognition",
                            languages=["en"],
                            attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                            installed=True,
                            version=__version__,
                        )
                    ],
                )
            ]
        }

        # Add TTS info if types are available
        if TtsProgram is not None and TtsVoice is not None:
            info_dict["tts"] = [
                TtsProgram(
                    name="ciris",
                    description=f"CIRIS TTS using {self.config.tts.provider}",
                    attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                    installed=True,
                    version=__version__,
                    voices=[
                        TtsVoice(
                            name="en-US-Standard",
                            description="English (US) voice",
                            attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                            installed=True,
                            languages=["en"],
                            version=__version__,
                        )
                    ],
                )
            ]

        return Info(**info_dict)


async def main():
    logging.basicConfig(
        level=logging.INFO,  # Changed to INFO for clearer output
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("=== CIRIS WYOMING BRIDGE STARTING ===")

    # Pre-flight checks
    logger.info("Running pre-flight diagnostics...")

    # Check if we're in Home Assistant addon environment

    is_addon = os.path.exists("/data/options.json")
    logger.info(f"Running as Home Assistant addon: {is_addon}")

    if is_addon:
        try:
            import json

            with open("/data/options.json", "r") as f:
                options = json.load(f)
                logger.info(f"Addon options: {json.dumps(options, indent=2)}")
        except Exception as e:
            logger.warning(f"Could not read addon options: {e}")

    # Check environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Process ID: {os.getpid()}")

    # Load configuration
    config = Config.from_yaml("config.yaml")
    logger.info("Configuration loaded successfully")

    # Test Wyoming info generation
    try:
        # Create a minimal test to verify info generation
        test_info = Info(
            asr=[
                AsrProgram(
                    name="ciris-stt",
                    description=f"CIRIS STT using {config.stt.provider}",
                    attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                    installed=True,
                    models=[
                        AsrModel(
                            name="ciris-stt-v1",
                            description=f"{config.stt.provider} speech recognition",
                            languages=["en_US", "en"],
                            attribution=Attribution(name="CIRIS AI", url="https://ciris.ai"),
                            installed=True,
                        )
                    ],
                )
            ]
        )
        logger.info("Wyoming info test successful")
        logger.info(f"Info event type: {test_info.event().type}")
        logger.info(f"Info event data keys: {list(test_info.event().data.keys())}")
    except Exception as e:
        logger.error(f"Failed to generate Wyoming info: {e}")

    # Create handler factory with debugging
    def handler_factory(reader, writer):
        peer = writer.get_extra_info("peername")
        logger.info("=== NEW CONNECTION ===")
        logger.info(f"Connection from: {peer}")
        logger.info(f"Socket info: {writer.get_extra_info('socket')}")
        return CIRISWyomingHandler(reader, writer, config)

    # Create and run server
    server = AsyncServer.from_uri(f"tcp://{config.wyoming.host}:{config.wyoming.port}")
    logger.info(f"Starting CIRIS Wyoming bridge on {config.wyoming.host}:{config.wyoming.port}")
    logger.info(f"Using {config.ciris.timeout}s timeout for CIRIS interactions")
    logger.info(f"STT: {config.stt.provider}, TTS: {config.tts.provider}")
    logger.info("=== READY FOR CONNECTIONS ===")

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        logger.info("Server stopped")


if __name__ == "__main__":
    import signal

    # Handle SIGTERM gracefully
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    asyncio.run(main())
