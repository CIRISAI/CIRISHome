"""CIRIS Home CLI - Simple setup for Home Assistant AI integration.

Usage:
    ciris-home              # Run interactive setup
    ciris-home discover     # Discover HA instances
    ciris-home auth         # Re-authenticate with HA
    ciris-home jetson       # Configure Jetson LLM
    ciris-home start        # Start CIRIS agent
    ciris-home status       # Show current status
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .discovery import DiscoveredHA, discover_all
from .installer import (
    CIRISConfig,
    check_docker,
    discover_jetson,
    get_config_dir,
    list_ollama_models,
    load_config,
    save_config,
)
from .oauth import OAuthTokens, authenticate_with_browser, authenticate_with_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print CIRIS Home banner."""
    print(
        """
╔═══════════════════════════════════════════════════════════╗
║                     CIRIS Home v{}                      ║
║         Multi-Modal AI Home Automation Platform          ║
╚═══════════════════════════════════════════════════════════╝
""".format(
            __version__.ljust(6)
        )
    )


def select_from_list(items: list, prompt: str) -> Optional[int]:
    """Interactive selection from a list.

    Returns:
        Selected index, or None if cancelled
    """
    if not items:
        return None

    for i, item in enumerate(items, 1):
        print(f"  [{i}] {item}")
    print(f"  [0] Cancel")
    print()

    while True:
        try:
            choice = input(f"{prompt}: ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return idx
            print("Invalid selection, try again.")
        except (ValueError, EOFError):
            return None


async def cmd_discover() -> Optional[DiscoveredHA]:
    """Discover Home Assistant instances."""
    print("\nSearching for Home Assistant on your network...")

    instances = await discover_all()

    if not instances:
        print("\nNo Home Assistant instances found.")
        print("\nTroubleshooting:")
        print("  - Ensure HA is running and on the same network")
        print("  - Try: export HOME_ASSISTANT_URL=http://your-ha-ip:8123")
        return None

    print(f"\nFound {len(instances)} Home Assistant instance(s):\n")

    idx = select_from_list(
        [f"{ha.name} - {ha.url} ({ha.source})" for ha in instances],
        "Select Home Assistant instance",
    )

    if idx is None:
        return None

    return instances[idx]


async def cmd_auth(
    ha_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[OAuthTokens]:
    """Authenticate with Home Assistant.

    If username/password provided, uses programmatic login.
    Otherwise opens browser for OAuth.
    """
    print(f"\nAuthenticating with {ha_url}...")

    if username and password:
        print("Using programmatic login...")
        tokens = await authenticate_with_credentials(ha_url, username, password)
        if tokens:
            print("Authentication successful!")
        else:
            print("Authentication failed!")
        return tokens
    else:
        return await authenticate_with_browser(ha_url)


def cmd_jetson(config: Optional[CIRISConfig] = None) -> Optional[str]:
    """Configure Jetson Nano LLM."""
    print("\nSearching for Jetson Nano...")

    jetson_ip = discover_jetson()

    if jetson_ip:
        print(f"Found Jetson at: {jetson_ip}")
    else:
        jetson_ip = input(
            "Jetson not found. Enter IP address (or press Enter to skip): "
        ).strip()
        if not jetson_ip:
            return None

    # List available models
    print(f"\nChecking available models on {jetson_ip}...")
    models = list_ollama_models(jetson_ip)

    if models:
        print(f"Found {len(models)} model(s):\n")
        for model in models:
            print(f"  - {model}")

        # Check for Gemma
        gemma_models = [m for m in models if "gemma" in m.lower()]
        if gemma_models:
            print(f"\nRecommended Gemma model: {gemma_models[0]}")
    else:
        print("No models found. You may need to pull a model first:")
        print("  ssh jetson.local")
        print("  ollama pull gemma3:4b-it-q4_K_M")

    return jetson_ip


def cmd_status():
    """Show current configuration status."""
    print("\nCIRIS Home Status")
    print("=" * 40)

    config = load_config()

    if not config:
        print("Status: NOT CONFIGURED")
        print("\nRun 'ciris-home' to start setup.")
        return

    print(f"Home Assistant: {config.ha_url or 'Not configured'}")
    print(f"HA Token: {'Set' if config.ha_token else 'Not set'}")
    print(f"Jetson IP: {config.jetson_ip or 'Not configured'}")
    print(f"LLM Model: {config.llm_model}")

    # Check Docker
    if check_docker():
        print(f"Docker: Available")
    else:
        print(f"Docker: Not found")

    # Check if agent is running
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ciris", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"Agent: Running ({result.stdout.strip()})")
        else:
            print("Agent: Not running")
    except Exception:
        pass


async def run_interactive_setup():
    """Run the full interactive setup wizard."""
    print_banner()

    # Check for existing config
    existing_config = load_config()
    if existing_config and existing_config.ha_token:
        print("Existing configuration found!")
        print(f"  Home Assistant: {existing_config.ha_url}")
        print(f"  Jetson: {existing_config.jetson_ip or 'Not configured'}")
        print()
        reconfigure = input("Reconfigure? [y/N]: ").strip().lower()
        if reconfigure != "y":
            print("\nUse 'ciris-home status' to see current config.")
            print("Use 'ciris-home start' to start the agent.")
            return

    # Step 1: Discover Home Assistant
    print("\n" + "=" * 50)
    print("Step 1: Discover Home Assistant")
    print("=" * 50)

    ha = await cmd_discover()
    if not ha:
        print("\nSetup cancelled.")
        return

    # Step 2: Authenticate
    print("\n" + "=" * 50)
    print("Step 2: Authenticate with Home Assistant")
    print("=" * 50)

    # Check for credentials in environment
    import os

    username = os.getenv("HA_USERNAME")
    password = os.getenv("HA_PASSWORD")

    tokens = await cmd_auth(ha.url, username, password)
    if not tokens:
        print("\nAuthentication failed. Setup cancelled.")
        return

    # Step 3: Configure Jetson (optional)
    print("\n" + "=" * 50)
    print("Step 3: Configure Jetson Nano LLM (optional)")
    print("=" * 50)

    jetson_ip = cmd_jetson()

    # Step 4: Save configuration
    print("\n" + "=" * 50)
    print("Step 4: Save Configuration")
    print("=" * 50)

    config = CIRISConfig(
        ha_url=ha.url,
        ha_token=tokens.access_token,
        ha_refresh_token=tokens.refresh_token,
        jetson_ip=jetson_ip or "",
        jetson_port=11434,
        llm_model="gemma3:4b-it-q4_K_M",  # Default to Gemma 4 Q4
    )

    if save_config(config):
        print(f"\nConfiguration saved to: {get_config_dir() / 'config.json'}")
    else:
        print("\nFailed to save configuration!")
        return

    # Step 5: Docker deployment (optional)
    print("\n" + "=" * 50)
    print("Step 5: Deploy CIRIS Agent")
    print("=" * 50)

    if not check_docker():
        print("\nDocker not found. To deploy the CIRIS agent:")
        print("  1. Install Docker: https://docs.docker.com/get-docker/")
        print("  2. Run: ciris-home start")
        return

    deploy = input("\nDeploy CIRIS agent container now? [Y/n]: ").strip().lower()
    if deploy != "n":
        print("\nStarting CIRIS agent deployment...")
        print("This will pull the latest CIRIS agent image and start the container.")
        # TODO: Implement actual deployment
        print("\nDeploy functionality coming soon!")
        print("For now, use docker-compose manually:")
        print("  cd /path/to/CIRISHome")
        print("  docker-compose up -d")

    # Done!
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print(
        f"""
Your CIRIS Home is configured:
  Home Assistant: {ha.url}
  Jetson LLM: {jetson_ip or 'Not configured'}

Next steps:
  ciris-home status   - Check status
  ciris-home start    - Start the agent
  ciris-home --help   - See all commands
"""
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CIRIS Home - Multi-Modal AI Home Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  (none)      Run interactive setup wizard
  discover    Find Home Assistant instances
  auth        Re-authenticate with Home Assistant
  jetson      Configure Jetson Nano LLM
  start       Start CIRIS web UI and agent
  serve       Start web UI server only
  status      Show current status

Examples:
  ciris-home                              # Interactive setup
  ciris-home discover                     # Find HA instances
  ciris-home auth -u user -p pass         # Programmatic auth
  ciris-home status                       # Check status

Environment Variables:
  HA_USERNAME    Home Assistant username (for auto-auth)
  HA_PASSWORD    Home Assistant password (for auto-auth)
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ciris-home {__version__}",
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["discover", "auth", "jetson", "start", "serve", "status"],
        help="Command to run",
    )

    parser.add_argument(
        "--ha-url",
        help="Home Assistant URL (for auth command)",
    )

    parser.add_argument(
        "-u",
        "--username",
        help="Home Assistant username (for programmatic auth)",
    )

    parser.add_argument(
        "-p",
        "--password",
        help="Home Assistant password (for programmatic auth)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Port for web server (default: 8099)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run command
    try:
        if args.command == "discover":
            asyncio.run(cmd_discover())

        elif args.command == "auth":
            config = load_config()
            ha_url = args.ha_url or (config.ha_url if config else None)
            if not ha_url:
                print("No Home Assistant URL. Run 'ciris-home' first or use --ha-url.")
                sys.exit(1)
            tokens = asyncio.run(cmd_auth(ha_url, args.username, args.password))
            if tokens and config:
                # Update saved config with new tokens
                config.ha_token = tokens.access_token
                config.ha_refresh_token = tokens.refresh_token
                save_config(config)
                print(f"Configuration updated at: {get_config_dir() / 'config.json'}")

        elif args.command == "jetson":
            cmd_jetson()

        elif args.command == "status":
            cmd_status()

        elif args.command == "start":
            print("Starting CIRIS agent...")
            config = load_config()
            if not config:
                print("No configuration found. Run 'ciris-home' first.")
                sys.exit(1)

            from .webserver import run_server

            print(f"\nStarting CIRIS Web UI server...")
            print(f"Home Assistant: {config.ha_url}")
            asyncio.run(run_server(config))

        elif args.command == "serve":
            config = load_config()
            if not config:
                print("No configuration found. Run 'ciris-home' first.")
                sys.exit(1)

            from .webserver import run_server

            asyncio.run(run_server(config, port=args.port or 8099))

        else:
            # No command - run interactive setup
            asyncio.run(run_interactive_setup())

    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
