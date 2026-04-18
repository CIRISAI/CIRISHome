"""The CIRIS AI Assistant integration with context profile sub-entries."""
import logging
from dataclasses import dataclass
from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .agent import CIRISAgent
from .ciris_ha_client import CIRISClient
from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_CHANNEL,
    CONF_TIMEOUT,
    DEFAULT_CHANNEL,
    DEFAULT_TIMEOUT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class CIRISRuntimeData:
    """Runtime data for CIRIS integration."""

    client: CIRISClient
    agents: dict[str, CIRISAgent]  # subentry_id -> agent


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up CIRIS from a config entry (parent entry - API connection)."""
    hass.data.setdefault(DOMAIN, {})

    # Initialize the CIRIS client for this API connection
    api_url = entry.data[CONF_API_URL]
    api_key = entry.data.get(CONF_API_KEY)
    timeout = entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)

    if not api_key:
        api_key = "admin:ciris_admin_password"

    client = CIRISClient(
        base_url=api_url,
        api_key=api_key,
        timeout=float(timeout),
        max_retries=1,
    )

    # Store runtime data
    runtime_data = CIRISRuntimeData(client=client, agents={})
    hass.data[DOMAIN][entry.entry_id] = runtime_data

    # If there are no sub-entries yet, create a default agent for the parent entry
    # This provides backwards compatibility
    if not entry.subentries:
        _LOGGER.info("No context profiles configured, using default agent")
        agent = CIRISAgent(hass, entry, client, context_profile=None)
        runtime_data.agents["default"] = agent
        conversation.async_set_agent(hass, entry, agent)
    else:
        # Set up agents for each sub-entry (context profile)
        for subentry in entry.subentries.values():
            await _setup_subentry_agent(hass, entry, subentry, runtime_data)

    _LOGGER.info(
        "CIRIS integration set up with %d context profile(s)",
        len(runtime_data.agents),
    )

    return True


async def _setup_subentry_agent(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry: ConfigEntry,
    runtime_data: CIRISRuntimeData,
) -> None:
    """Set up a conversation agent for a context profile sub-entry."""
    profile_name = subentry.title
    _LOGGER.info("Setting up CIRIS agent for context profile: %s", profile_name)

    # Create agent with context profile
    agent = CIRISAgent(
        hass,
        entry,
        runtime_data.client,
        context_profile=subentry.data,
    )
    runtime_data.agents[subentry.entry_id] = agent

    # Register the conversation agent for this sub-entry
    conversation.async_set_agent(hass, subentry, agent)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    runtime_data: CIRISRuntimeData = hass.data[DOMAIN].get(entry.entry_id)

    if runtime_data:
        # Unregister all agents
        for agent_id, agent in runtime_data.agents.items():
            if agent_id == "default":
                conversation.async_unset_agent(hass, entry)
            await agent.async_close()

        # Unregister sub-entry agents
        for subentry in entry.subentries.values():
            conversation.async_unset_agent(hass, subentry)

    # Remove runtime data
    hass.data[DOMAIN].pop(entry.entry_id, None)

    _LOGGER.info("CIRIS integration unloaded")
    return True


async def async_setup_subentry(
    hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigEntry
) -> bool:
    """Set up a CIRIS context profile sub-entry."""
    runtime_data: CIRISRuntimeData = hass.data[DOMAIN].get(entry.entry_id)

    if not runtime_data:
        _LOGGER.error("Cannot set up sub-entry: parent entry not found")
        return False

    # If we had a default agent, remove it now that we have sub-entries
    if "default" in runtime_data.agents:
        _LOGGER.info("Removing default agent, using context profiles instead")
        conversation.async_unset_agent(hass, entry)
        default_agent = runtime_data.agents.pop("default")
        await default_agent.async_close()

    await _setup_subentry_agent(hass, entry, subentry, runtime_data)
    return True


async def async_unload_subentry(
    hass: HomeAssistant, entry: ConfigEntry, subentry: ConfigEntry
) -> bool:
    """Unload a CIRIS context profile sub-entry."""
    runtime_data: CIRISRuntimeData = hass.data[DOMAIN].get(entry.entry_id)

    if not runtime_data:
        return True

    # Unregister and clean up the agent
    if subentry.entry_id in runtime_data.agents:
        agent = runtime_data.agents.pop(subentry.entry_id)
        conversation.async_unset_agent(hass, subentry)
        await agent.async_close()

    # If no more sub-entries, restore default agent
    if not runtime_data.agents and not entry.subentries:
        _LOGGER.info("No context profiles remaining, restoring default agent")
        agent = CIRISAgent(hass, entry, runtime_data.client, context_profile=None)
        runtime_data.agents["default"] = agent
        conversation.async_set_agent(hass, entry, agent)

    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    if entry.version == 1:
        _LOGGER.info("Migrating CIRIS config entry from version 1 to 2")
        # Version 1 -> 2: Add sub-entry support
        # No data migration needed, just update version
        hass.config_entries.async_update_entry(entry, version=2)
        _LOGGER.info("Migration to version 2 complete")

    return True
