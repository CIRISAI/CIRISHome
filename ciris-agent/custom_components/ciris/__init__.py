"""The CIRIS AI Assistant integration."""

import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .agent import CIRISAgent
from .ciris_ha_client import CIRISClient
from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_TIMEOUT,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up CIRIS from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Initialize the CIRIS client
    api_url = entry.data[CONF_API_URL]
    api_key = entry.data.get(CONF_API_KEY)
    timeout = entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

    if not api_key:
        api_key = "admin:ciris_admin_password"  # pragma: allowlist secret

    client = CIRISClient(
        base_url=api_url,
        api_key=api_key,
        timeout=float(timeout),
        max_retries=1,
    )

    # Create agent with context from options
    context_profile = dict(entry.options) if entry.options else None
    agent = CIRISAgent(hass, entry, client, context_profile=context_profile)

    # Store runtime data
    hass.data[DOMAIN][entry.entry_id] = {
        "client": client,
        "agent": agent,
    }

    # Register the conversation agent
    conversation.async_set_agent(hass, entry, agent)

    # Listen for options updates
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    _LOGGER.info("CIRIS integration set up successfully")
    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    _LOGGER.info("CIRIS options updated, reloading integration")
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    runtime_data = hass.data[DOMAIN].get(entry.entry_id)

    if runtime_data:
        agent = runtime_data.get("agent")
        if agent:
            conversation.async_unset_agent(hass, entry)
            await agent.async_close()

    # Remove runtime data
    hass.data[DOMAIN].pop(entry.entry_id, None)

    _LOGGER.info("CIRIS integration unloaded")
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    if entry.version == 1:
        _LOGGER.info("Migrating CIRIS config entry from version 1 to 2")
        hass.config_entries.async_update_entry(entry, version=2)
        _LOGGER.info("Migration to version 2 complete")

    return True
