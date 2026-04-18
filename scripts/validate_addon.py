#!/usr/bin/env python3
"""Validate CIRIS Agent addon config.yaml.

This script is used by pre-commit to ensure:
1. The slug never changes (prevents breaking upgrades)
2. Version format is valid semver
3. Required fields are present
"""

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
CONFIG_FILE = REPO_ROOT / "ciris-agent" / "config.yaml"

# CRITICAL: This slug must NEVER change
REQUIRED_SLUG = "ciris-agent"

REQUIRED_FIELDS = [
    "name",
    "version",
    "slug",
    "description",
    "arch",
    "ingress",
]

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def error(msg: str) -> None:
    """Print error and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg: str) -> None:
    """Print warning."""
    print(f"WARNING: {msg}", file=sys.stderr)


def validate_config() -> None:
    """Validate the addon config.yaml."""
    if not CONFIG_FILE.exists():
        error(f"Config file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            error(f"Missing required field: {field}")

    # CRITICAL: Validate slug never changes
    if config["slug"] != REQUIRED_SLUG:
        error(
            f"SLUG MUST NOT CHANGE!\n"
            f"  Expected: {REQUIRED_SLUG}\n"
            f"  Found: {config['slug']}\n"
            f"\n"
            f"Changing the slug breaks upgrades for all users.\n"
            f"The addon will appear as a completely new addon."
        )

    # Validate version format
    version = str(config["version"])
    if not SEMVER_PATTERN.match(version):
        error(
            f"Invalid version format: {version}\n"
            f"Version must be semver format: X.Y.Z (e.g., 5.0.0)"
        )

    # Validate architectures
    valid_archs = {"aarch64", "amd64", "armv7", "armhf", "i386"}
    for arch in config.get("arch", []):
        if arch not in valid_archs:
            warn(f"Unknown architecture: {arch}")

    print(f"✓ Config valid: {config['name']} v{config['version']}")


if __name__ == "__main__":
    validate_config()
