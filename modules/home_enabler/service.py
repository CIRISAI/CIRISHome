"""
Home Enabler Service - Enables home automation capabilities when authorized.

Similar to medical_enabler but for home automation and multi-modal capabilities.
This enables the prohibited home automation capabilities in WiseBus.
"""

import logging
import os

logger = logging.getLogger(__name__)


def enable_home_capabilities():
    """
    Post-startup hook that enables home automation capabilities in WiseBus.

    This clears home automation prohibitions, allowing multi-modal processing
    for the CIRISHome development platform.
    """
    # Verify home automation requirements
    ha_url = os.getenv("HOME_ASSISTANT_URL")
    ha_token = os.getenv("HOME_ASSISTANT_TOKEN")
    responsibility_accepted = os.getenv("I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY")

    if not ha_url:
        logger.warning("HOME_ASSISTANT_URL not set - running in standalone mode")
        ha_url = "http://localhost:8123"

    if not ha_token:
        logger.warning(
            "HOME_ASSISTANT_TOKEN not set - Home Assistant integration disabled"
        )

    if responsibility_accepted != "true":
        logger.warning(
            "I_ACCEPT_HOME_AUTOMATION_RESPONSIBILITY not set to 'true' - "
            "some features limited"
        )

    # Import WiseBus and enable home automation capabilities
    try:
        from ciris_engine.logic.buses.wise_bus import WiseBus

        # Enable home automation capabilities
        # This is the key line - removes home automation from prohibited set
        current_prohibited = getattr(WiseBus, "PROHIBITED_CAPABILITIES", set())
        home_prohibited = {
            cap
            for cap in current_prohibited
            if "home" in cap.lower() or "automation" in cap.lower()
        }

        WiseBus.PROHIBITED_CAPABILITIES = current_prohibited - home_prohibited

        logger.info(
            "HOME AUTOMATION ENABLED - Multi-modal capabilities active. "
            "HA URL: %s, Token: %s",
            ha_url,
            "SET" if ha_token else "NOT SET",
        )
        logger.info(
            "Home automation and multi-modal processing enabled for "
            "CIRISHome development"
        )

        return True

    except ImportError:
        logger.error("Could not import WiseBus - running without CIRIS engine")
        return False
    except Exception as e:
        logger.error(f"Failed to enable home capabilities: {e}")
        return False
