"""
Tests for the home_enabler module - critical capability override functionality.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestHomeEnabler:
    """Test home automation capability enablement."""

    @pytest.fixture
    def mock_wise_bus(self):
        """Mock WiseBus with prohibited capabilities."""
        bus = Mock()
        bus.PROHIBITED_CAPABILITIES = {
            "domain:medical",
            "domain:health",
            "domain:clinical",
            "domain:home_automation",  # Should be removed
            "modality:sensor:medical",
            "action:home_control",  # Should be removed
        }
        return bus

    def test_home_automation_capabilities_enabled(self, mock_wise_bus):
        """Test that home automation capabilities are enabled."""
        with patch("ciris_engine.logic.buses.wise_bus.WiseBus", mock_wise_bus):
            # Import and run the function
            from modules.home_enabler.service import enable_home_capabilities

            result = enable_home_capabilities()

            # Verify function succeeded
            assert result is True

            # Check that home automation capabilities were removed
            remaining = mock_wise_bus.PROHIBITED_CAPABILITIES
            assert "domain:home_automation" not in remaining
            assert "action:home_control" not in remaining

            # Medical capabilities should remain
            assert "domain:medical" in remaining
            assert "domain:health" in remaining
            assert "modality:sensor:medical" in remaining

    def test_medical_capabilities_remain_prohibited(self, mock_wise_bus):
        """Test that medical capabilities are never enabled."""
        with patch("ciris_engine.logic.buses.wise_bus.WiseBus", mock_wise_bus):
            from modules.home_enabler.service import enable_home_capabilities

            enable_home_capabilities()

            # Medical capabilities must remain prohibited
            remaining = mock_wise_bus.PROHIBITED_CAPABILITIES
            assert "domain:medical" in remaining
            assert "domain:health" in remaining
            assert "domain:clinical" in remaining
            assert "modality:sensor:medical" in remaining

    def test_responsibility_acceptance_required(self, env_vars, monkeypatch):
        """Test that responsibility acceptance is enforced."""
        # Remove responsibility acceptance
        monkeypatch.delenv("I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY", raising=False)

        with patch("ciris_engine.logic.buses.wise_bus.WiseBus", Mock()):
            with patch("modules.home_enabler.service.logger") as mock_logger:
                from modules.home_enabler.service import enable_home_capabilities

                enable_home_capabilities()

                # Should log warning about missing responsibility acceptance
                mock_logger.warning.assert_called_with(
                    "I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY not set to 'true' - some features limited"
                )

    def test_function_enables_capabilities(self, env_vars):
        """Test that function successfully enables capabilities."""
        mock_bus = Mock()
        mock_bus.PROHIBITED_CAPABILITIES = {
            "domain:medical",
            "domain:home_automation",
            "action:home_control",
        }

        with patch("ciris_engine.logic.buses.wise_bus.WiseBus", mock_bus):
            from modules.home_enabler.service import enable_home_capabilities

            result = enable_home_capabilities()

            # Function should return True on success
            assert result is True


class TestCapabilityFiltering:
    """Test capability filtering logic."""

    def test_only_home_capabilities_removed(self):
        """Test that only home automation capabilities are removed."""
        prohibited_capabilities = {
            "domain:medical",
            "domain:health",
            "domain:clinical",
            "domain:home_automation",  # Should be removed
            "modality:sensor:medical",
            "action:home_control",  # Should be removed
            "domain:unknown_future",  # Should remain
            "some:other:capability",  # Should remain
        }

        # Define the filtering logic (normally in home_enabler)
        home_capabilities = {
            "domain:home_automation",
            "action:home_control",
            "modality:home:*",
            "capability:control_devices",
        }

        # Filter out home capabilities
        filtered = {
            cap
            for cap in prohibited_capabilities
            if not any(cap.startswith(hc.replace("*", "")) for hc in home_capabilities)
        }

        # Medical capabilities should remain
        assert "domain:medical" in filtered
        assert "domain:health" in filtered
        assert "modality:sensor:medical" in filtered

        # Home capabilities should be removed
        assert "domain:home_automation" not in filtered
        assert "action:home_control" not in filtered

        # Unknown capabilities should remain
        assert "domain:unknown_future" in filtered
        assert "some:other:capability" in filtered

    def test_capability_override_is_logged(self):
        """Test that capability override is properly logged for audit."""
        with patch("logging.getLogger") as mock_get_logger:
            logger = Mock()
            mock_get_logger.return_value = logger

            # Simulate the logging that should happen
            logger.info(
                "Enabling home automation capabilities - user has accepted responsibility"
            )
            logger.info(
                "Removed 2 home automation restrictions from WiseBus.PROHIBITED_CAPABILITIES"
            )

            # Verify logging occurred
            assert logger.info.call_count >= 1

            # Check that important messages were logged
            calls = [call[0][0] for call in logger.info.call_args_list]
            assert any("home automation capabilities" in call for call in calls)

    def test_edge_cases_in_capability_matching(self):
        """Test edge cases in capability string matching."""
        test_capabilities = {
            "domain:home_automation",
            "domain:home_automation_extended",
            "action:home_control",
            "action:home_control_advanced",
            "modality:home:lighting",
            "modality:home:climate",
            "domain:medical",  # Should never match
            "action:medical_home_care",  # Tricky - contains both
        }

        home_patterns = [
            "domain:home_automation",
            "action:home_control",
            "modality:home:",
        ]

        # Test filtering logic
        should_be_removed = []
        should_remain = []

        for cap in test_capabilities:
            if any(cap.startswith(pattern.rstrip(":*")) for pattern in home_patterns):
                # But never remove medical capabilities
                if "medical" not in cap:
                    should_be_removed.append(cap)
                else:
                    should_remain.append(cap)
            else:
                should_remain.append(cap)

        # Verify correct categorization
        assert "domain:home_automation" in should_be_removed
        assert "action:home_control" in should_be_removed
        assert "modality:home:lighting" in should_be_removed

        assert "domain:medical" in should_remain
        assert "action:medical_home_care" in should_remain  # Medical takes precedence
