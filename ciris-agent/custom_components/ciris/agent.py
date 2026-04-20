"""CIRIS conversation agent with context profile support."""

import logging
import re
from typing import Any, Literal, Optional

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.util import ulid

from .ciris_ha_client import CIRISClient
from .ciris_sdk.exceptions import CIRISError, CIRISTimeoutError
from .const import (
    CONF_CHANNEL,
    CONF_CUSTOM_INSTRUCTIONS,
    CONF_LANGUAGE,
    CONF_PROFILE_NAME,
    CONF_RESPONSE_STYLE,
    CONF_ROOM_TYPE,
    CONF_SAFETY_LEVEL,
    DEFAULT_CHANNEL,
    DEFAULT_LANGUAGE,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_SAFETY_LEVEL,
    LANGUAGES,
    RESPONSE_STYLES,
    ROOM_TYPES,
)

_LOGGER = logging.getLogger(__name__)


class CIRISAgent(conversation.AbstractConversationAgent):
    """CIRIS conversation agent with context profile support."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: CIRISClient,
        context_profile: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            hass: Home Assistant instance
            entry: Parent config entry (API connection)
            client: Shared CIRIS client
            context_profile: Optional context profile from sub-entry
        """
        self.hass = hass
        self.entry = entry
        self._client = client
        self._client_initialized = False

        # Context profile settings
        self.context_profile = context_profile or {}
        self.profile_name = self.context_profile.get(CONF_PROFILE_NAME, "Default")
        self.room_type = self.context_profile.get(CONF_ROOM_TYPE, "shared_space")
        self.language = self.context_profile.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        self.safety_level = self.context_profile.get(
            CONF_SAFETY_LEVEL, DEFAULT_SAFETY_LEVEL
        )
        self.response_style = self.context_profile.get(
            CONF_RESPONSE_STYLE, DEFAULT_RESPONSE_STYLE
        )
        self.custom_instructions = self.context_profile.get(
            CONF_CUSTOM_INSTRUCTIONS, ""
        )
        self.channel = entry.data.get(CONF_CHANNEL, DEFAULT_CHANNEL)

        _LOGGER.info(
            "CIRIS Agent initialized - Profile: %s, Room: %s, Safety: %s, Style: %s",
            self.profile_name,
            self.room_type,
            self.safety_level,
            self.response_style,
        )

    def _build_system_instructions(self) -> str:
        """Build system instructions based on context profile."""
        instructions = []

        # Base HA integration instructions
        instructions.append(
            "You are integrated with Home Assistant. "
            "You can control devices by mentioning their names in your response. "
            "For example: 'I'll turn on the kitchen light for you' or "
            "'Let me switch off the bedroom fan'. "
            "The system will automatically execute these commands."
        )

        # Room type context
        room_desc = ROOM_TYPES.get(self.room_type, "")
        if room_desc:
            instructions.append(f"Context: {room_desc}")

        # Safety level instructions
        if self.safety_level == "kids_safe":
            instructions.append(
                "IMPORTANT: This is a child-safe environment. "
                "Use simple, age-appropriate language. "
                "Never discuss adult topics, violence, or anything scary. "
                "Be encouraging and educational."
            )
        elif self.safety_level == "elderly_safe":
            instructions.append(
                "IMPORTANT: You are assisting an elderly person. "
                "Speak clearly with simple language and short sentences. "
                "Be patient and respectful. If health is mentioned, "
                "suggest consulting their doctor."
            )
        elif self.safety_level == "family_friendly":
            instructions.append(
                "Keep all responses appropriate for a family environment."
            )

        # Response style
        style_desc = RESPONSE_STYLES.get(self.response_style, "")
        if style_desc and self.response_style != "normal":
            instructions.append(f"Response style: {style_desc}")

        # Language preference
        if self.language != "en":
            lang_name = LANGUAGES.get(self.language, self.language)
            instructions.append(f"Respond in {lang_name}.")

        # Custom instructions
        if self.custom_instructions:
            instructions.append(self.custom_instructions)

        # Final instruction
        instructions.append(
            "Please SPEAK naturally and include any actions in your response."
        )

        return " ".join(instructions)

    async def _ensure_client(self) -> CIRISClient:
        """Ensure the CIRIS client is initialized."""
        if not self._client_initialized:
            await self._client.__aenter__()

            # Handle username:password auth
            if (
                self._client._transport.api_key
                and ":" in self._client._transport.api_key
            ):
                username, password = self._client._transport.api_key.split(":", 1)
                _LOGGER.info("Using username/password auth for user: %s", username)

                try:
                    token = await self._client.auth.login(username, password)
                    _LOGGER.info("Successfully logged in to CIRIS")
                    self._client._transport.set_api_key(
                        token.access_token, persist=False
                    )
                except Exception as e:
                    _LOGGER.error("Failed to login to CIRIS: %s", e)
                    raise

            self._client_initialized = True

        return self._client

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    @property
    def attribution(self) -> str | None:
        """Return attribution information."""
        if self.profile_name != "Default":
            return f"Powered by CIRIS AI ({self.profile_name})"
        return "Powered by CIRIS AI"

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence from the user."""
        _LOGGER.info(
            "CIRIS [%s]: Processing input: '%s'",
            self.profile_name,
            user_input.text,
        )
        intent_response = intent.IntentResponse(language=user_input.language)

        try:
            client = await self._ensure_client()

            # Check if CIRIS is available
            try:
                status = await client.agent.get_status()
                _LOGGER.info(
                    "CIRIS status: %s (state: %s)",
                    status.name,
                    status.cognitive_state,
                )
            except Exception as e:
                _LOGGER.error("Failed to get CIRIS status: %s", e)
                msg = "I'm having trouble connecting to CIRIS. "
                msg += "Please check the configuration."
                intent_response.async_set_speech(msg)
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=user_input.conversation_id or ulid.ulid(),
                )

            # Get available devices for context
            device_info = await self._get_device_info()

            # Build context for CIRIS with profile settings
            conv_id = user_input.conversation_id or "default"
            context = {
                "source": "homeassistant",
                "channel_id": f"{self.channel}_{self.profile_name}_{conv_id}",
                "input_method": "voice" if user_input.conversation_id else "text",
                "language": self.language,
                "hass_context": {
                    "user_id": (
                        user_input.context.user_id if user_input.context else None
                    ),
                    "parent_id": (
                        user_input.context.parent_id if user_input.context else None
                    ),
                },
                "available_devices": device_info,
                "instructions": self._build_system_instructions(),
                # Context profile metadata
                "context_profile": {
                    "name": self.profile_name,
                    "room_type": self.room_type,
                    "safety_level": self.safety_level,
                    "response_style": self.response_style,
                },
            }

            # Send to CIRIS using the SDK
            try:
                profile = self.profile_name
                enhanced_message = (
                    f"{user_input.text}\n\n"
                    f"[Received via Home Assistant ({profile}), "
                    f"please SPEAK to service this request, thank you!]"
                )

                _LOGGER.debug(
                    "CIRIS: Sending message with context profile: %s", context
                )
                response = await client.agent.interact(
                    message=enhanced_message, context=context
                )

                response_text = response.response
                _LOGGER.info(
                    "CIRIS responded in %dms: '%s'",
                    response.processing_time_ms,
                    response_text[:100],
                )

                # Check if CIRIS wants to control devices
                device_controlled = await self._process_device_control(
                    response_text, user_input
                )

                # Set the speech response
                intent_response.async_set_speech(response_text)

                # If we controlled a device, add a card
                if device_controlled:
                    intent_response.async_set_card(
                        f"CIRIS ({self.profile_name})", response_text
                    )

            except CIRISTimeoutError:
                _LOGGER.warning("CIRIS timeout")
                intent_response.async_set_speech(
                    "CIRIS is taking too long to respond. Please try again."
                )
            except CIRISError as e:
                _LOGGER.error("CIRIS error: %s", e)
                intent_response.async_set_speech(
                    "I encountered an error processing your request."
                )

        except Exception as e:
            _LOGGER.error("Error processing with CIRIS: %s", e, exc_info=True)
            intent_response.async_set_speech(
                "I encountered an error. Please try again later."
            )

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id or ulid.ulid(),
        )

    async def _get_device_info(self) -> dict:
        """Get information about available devices."""
        device_info: dict[str, list] = {
            "lights": [],
            "switches": [],
            "fans": [],
            "covers": [],
            "climate": [],
        }

        try:
            states = self.hass.states.async_all()

            for state in states:
                entity_id = state.entity_id
                domain = entity_id.split(".")[0]

                if domain in device_info:
                    device_info[domain].append(
                        {
                            "entity_id": entity_id,
                            "name": state.attributes.get("friendly_name", entity_id),
                            "state": state.state,
                        }
                    )

            _LOGGER.debug(
                "Found devices: %d lights, %d switches, %d fans, %d covers",
                len(device_info["lights"]),
                len(device_info["switches"]),
                len(device_info["fans"]),
                len(device_info["covers"]),
            )

        except Exception as e:
            _LOGGER.error("Error getting device info: %s", e)

        return device_info

    async def _process_device_control(
        self,
        response_text: str,
        user_input: conversation.ConversationInput,
    ) -> bool:
        """Process device control commands from CIRIS response."""
        controlled_any = False

        # Turn on/off pattern
        turn_pattern = (
            r"(turn(?:ing)?|switch(?:ing)?)\s+(on|off)\s+(?:the\s+)?(.+?)(?:\.|,|$)"
        )
        matches = re.finditer(turn_pattern, response_text.lower(), re.IGNORECASE)

        for match in matches:
            action = match.group(2)
            target = match.group(3).strip()

            _LOGGER.info("Detected device control: %s %s", action, target)

            entity_id = await self._find_entity(target)
            if entity_id:
                try:
                    domain = entity_id.split(".")[0]
                    service = "turn_on" if action == "on" else "turn_off"
                    await self.hass.services.async_call(
                        domain,
                        service,
                        {"entity_id": entity_id},
                        context=user_input.context,
                    )
                    controlled_any = True
                except Exception as e:
                    _LOGGER.error("Error controlling device %s: %s", entity_id, e)

        # Toggle pattern
        toggle_pattern = r"toggle\s+(?:the\s+)?(.+?)(?:\.|,|$)"
        toggle_matches = re.finditer(
            toggle_pattern, response_text.lower(), re.IGNORECASE
        )

        for match in toggle_matches:
            target = match.group(1).strip()
            _LOGGER.info("Detected toggle: %s", target)

            entity_id = await self._find_entity(target)
            if entity_id:
                try:
                    await self.hass.services.async_call(
                        entity_id.split(".")[0],
                        "toggle",
                        {"entity_id": entity_id},
                        context=user_input.context,
                    )
                    controlled_any = True
                except Exception as e:
                    _LOGGER.error("Error toggling device %s: %s", entity_id, e)

        return controlled_any

    async def _find_entity(self, name: str) -> Optional[str]:
        """Find entity ID by friendly name."""
        name_lower = name.lower()
        states = self.hass.states.async_all()

        for state in states:
            entity_id = state.entity_id
            domain = entity_id.split(".")[0]

            if domain in ["light", "switch", "fan", "cover", "climate"]:
                friendly_name = state.attributes.get("friendly_name", "").lower()

                if name_lower == friendly_name or name_lower in friendly_name:
                    _LOGGER.info("Found entity %s for '%s'", entity_id, name)
                    return entity_id

                entity_name = entity_id.split(".")[1].replace("_", " ")
                if name_lower == entity_name or name_lower in entity_name:
                    _LOGGER.info("Found entity %s for '%s'", entity_id, name)
                    return entity_id

        _LOGGER.warning("Could not find entity for '%s'", name)
        return None

    async def async_close(self) -> None:
        """Close the agent."""
        # Note: Don't close the shared client here, it's managed by __init__.py
        self._client_initialized = False
        _LOGGER.info("CIRIS Agent [%s] closed", self.profile_name)
