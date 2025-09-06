#!/usr/bin/env python3
"""
CIRISHome - Multi-modal AI Development Platform.

Startup script for CIRISHome multi-modal capabilities development.
Uses CIRISAgent as the core engine with home automation modules.
"""

import argparse
import os
import sys
from pathlib import Path

# Add external CIRIS engine to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "ciris-engine"))


def main():
    """Run CIRISHome platform."""
    parser = argparse.ArgumentParser(
        description="CIRISHome Multi-modal AI Platform"
    )

    # Adapter selection
    parser.add_argument(
        "--adapter",
        choices=["ha", "api", "cli"],
        default="ha",
        help="Adapter to use (ha=Home Assistant, api=API server, cli=CLI)",
    )

    # Home Assistant specific
    parser.add_argument(
        "--ha-url",
        default=os.getenv("HOME_ASSISTANT_URL", "http://localhost:8123"),
        help="Home Assistant URL",
    )
    parser.add_argument(
        "--ha-token",
        default=os.getenv("HOME_ASSISTANT_TOKEN"),
        help="Home Assistant Long-Lived Access Token",
    )

    # Development options
    parser.add_argument(
        "--dev-mode", action="store_true", help="Run in development mode"
    )
    parser.add_argument(
        "--mock-ha",
        action="store_true",
        help="Use mock Home Assistant for testing",
    )

    # Multi-modal options
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable vision processing pipeline",
    )
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Enable audio processing pipeline",
    )

    # Standard CIRIS options
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for API server mode"
    )
    parser.add_argument(
        "--mock-llm", action="store_true", help="Use mock LLM for testing"
    )

    args = parser.parse_args()

    # Set environment variables for modules
    if args.ha_url:
        os.environ["HOME_ASSISTANT_URL"] = args.ha_url
    if args.ha_token:
        os.environ["HOME_ASSISTANT_TOKEN"] = args.ha_token

    # Enable development features
    if args.dev_mode:
        os.environ["DEV_MODE"] = "true"
        os.environ["I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY"] = "true"
        print("Development mode enabled - home automation capabilities active")

    if args.mock_ha:
        os.environ["MOCK_HOME_ASSISTANT"] = "true"
        print("Mock Home Assistant enabled")

    # Enable multi-modal processing
    if args.enable_vision:
        os.environ["ENABLE_VISION_PIPELINE"] = "true"
        print("Vision processing enabled")

    if args.enable_audio:
        os.environ["ENABLE_AUDIO_PIPELINE"] = "true"
        print("Audio processing enabled")

    # Import and run CIRIS engine with our modules
    try:
        print("Starting CIRISHome multi-modal AI platform...")
        print(f"Adapter: {args.adapter}")
        print(f"Home Assistant: {args.ha_url}")

        # Set module path for our home modules
        modules_path = str(Path(__file__).parent / "modules")
        os.environ["CIRIS_MODULES_PATH"] = modules_path

        # Import and delegate to CIRISAgent main
        from ciris_engine.main import main as ciris_main

        # Convert args for CIRIS engine
        ciris_args = ["--adapter", args.adapter, "--port", str(args.port)]

        if args.mock_llm:
            ciris_args.append("--mock-llm")

        # Override sys.argv for CIRIS engine
        original_argv = sys.argv
        sys.argv = ["main.py"] + ciris_args

        try:
            ciris_main()
        finally:
            sys.argv = original_argv

    except KeyboardInterrupt:
        print("\nCIRISHome shutdown requested")
    except Exception as e:
        print(f"CIRISHome startup failed: {e}")
        if args.dev_mode:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
