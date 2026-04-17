"""Tests for CIRIS Agent addon configuration.

These tests ensure the addon configuration is valid and
the upgrade mechanism works reliably.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
ADDON_DIR = REPO_ROOT / "ciris-agent"
CONFIG_FILE = ADDON_DIR / "config.yaml"

# CRITICAL: This must never change
REQUIRED_SLUG = "ciris-agent"


@pytest.fixture
def config() -> dict:
    """Load the addon config.yaml."""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


class TestAddonConfig:
    """Test suite for addon configuration."""

    def test_config_exists(self):
        """Config.yaml must exist."""
        assert CONFIG_FILE.exists(), f"Config file not found: {CONFIG_FILE}"

    def test_slug_is_correct(self, config):
        """Slug must be 'ciris-agent' - NEVER CHANGE THIS.

        Changing the slug breaks upgrades for all users.
        """
        assert config["slug"] == REQUIRED_SLUG, (
            f"CRITICAL: Slug must be '{REQUIRED_SLUG}', not '{config['slug']}'. "
            "Changing the slug breaks upgrades for all existing users!"
        )

    def test_version_is_semver(self, config):
        """Version must be valid semver format."""
        version = str(config["version"])
        pattern = re.compile(r"^\d+\.\d+\.\d+$")
        assert pattern.match(version), (
            f"Version '{version}' is not valid semver. "
            "Use format X.Y.Z (e.g., 5.0.0)"
        )

    def test_required_fields_present(self, config):
        """All required fields must be present."""
        required = ["name", "version", "slug", "description", "arch", "ingress"]
        for field in required:
            assert field in config, f"Missing required field: {field}"

    def test_architectures_valid(self, config):
        """Architecture list must contain valid values."""
        valid_archs = {"aarch64", "amd64", "armv7", "armhf", "i386"}
        for arch in config.get("arch", []):
            assert arch in valid_archs, f"Invalid architecture: {arch}"

    def test_ingress_enabled(self, config):
        """Ingress must be enabled for web UI addon."""
        assert config.get("ingress") is True, "Ingress must be enabled"

    def test_ingress_port_set(self, config):
        """Ingress port must be configured."""
        assert "ingress_port" in config, "ingress_port must be set"
        assert isinstance(config["ingress_port"], int), "ingress_port must be int"


class TestAddonStructure:
    """Test addon directory structure."""

    def test_dockerfile_exists(self):
        """Dockerfile must exist."""
        dockerfile = ADDON_DIR / "Dockerfile"
        assert dockerfile.exists(), f"Dockerfile not found: {dockerfile}"

    def test_run_script_exists(self):
        """run.sh must exist."""
        run_sh = ADDON_DIR / "run.sh"
        assert run_sh.exists(), f"run.sh not found: {run_sh}"

    def test_server_exists(self):
        """Server code must exist."""
        server = ADDON_DIR / "rootfs" / "app" / "server.py"
        assert server.exists(), f"server.py not found: {server}"

    def test_web_assets_exist(self):
        """Web assets must exist."""
        www = ADDON_DIR / "www"
        assert www.exists(), f"www directory not found: {www}"
        assert (www / "index.html").exists(), "index.html not found"


class TestRepositoryConfig:
    """Test repository configuration."""

    def test_repository_json_exists(self):
        """repository.json must exist at repo root."""
        repo_json = REPO_ROOT / "repository.json"
        assert repo_json.exists(), f"repository.json not found: {repo_json}"

    def test_repository_json_valid(self):
        """repository.json must be valid."""
        import json

        repo_json = REPO_ROOT / "repository.json"
        with open(repo_json) as f:
            data = json.load(f)

        assert "name" in data, "repository.json missing 'name'"
        assert "url" in data, "repository.json missing 'url'"
        assert "maintainer" in data, "repository.json missing 'maintainer'"
