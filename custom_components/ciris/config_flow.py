"""Config flow for CIRIS integration with sub-entries for context profiles."""
import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry, ConfigFlowResult
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_CHANNEL,
    CONF_CUSTOM_INSTRUCTIONS,
    CONF_LANGUAGE,
    CONF_PROFILE_NAME,
    CONF_RESPONSE_STYLE,
    CONF_ROOM_TYPE,
    CONF_SAFETY_LEVEL,
    CONF_TIMEOUT,
    CONF_WAKE_WORD,
    CONTEXT_PRESETS,
    DEFAULT_API_URL,
    DEFAULT_CHANNEL,
    DEFAULT_LANGUAGE,
    DEFAULT_RESPONSE_STYLE,
    DEFAULT_SAFETY_LEVEL,
    DEFAULT_TIMEOUT,
    DOMAIN,
    LANGUAGES,
    RESPONSE_STYLES,
    ROOM_TYPES,
    SAFETY_LEVELS,
)

from .ciris_ha_client import CIRISClient
from .ciris_sdk.exceptions import CIRISError, CIRISTimeoutError

_LOGGER = logging.getLogger(__name__)


class CIRISConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for CIRIS (parent entry - API connection)."""

    VERSION = 2  # Bumped for sub-entry support

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry):
        """Get the options flow for this handler."""
        return CIRISOptionsFlow(config_entry)

    @staticmethod
    @callback
    def async_get_subentry_flow(config_entry: ConfigEntry, subentry_type: str):
        """Get the subentry flow for this handler."""
        return CIRISSubentryFlow()

    @classmethod
    @callback
    def async_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[config_entries.SubentryFlowHandler]]:
        """Return subentry types supported by this handler."""
        return {"context_profile": CIRISSubentryFlow}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - API connection setup."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate the API connection
            try:
                await self._test_connection(
                    user_input[CONF_API_URL],
                    user_input.get(CONF_API_KEY),
                    user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                )
            except CIRISTimeoutError:
                errors["base"] = "timeout"
            except CIRISError as e:
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title=user_input.get(CONF_NAME, "CIRIS"),
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_NAME, default="CIRIS"): str,
                    vol.Required(CONF_API_URL, default=DEFAULT_API_URL): str,
                    vol.Optional(CONF_API_KEY): str,
                    vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.All(
                        vol.Coerce(int), vol.Range(min=5, max=300)
                    ),
                    vol.Optional(CONF_CHANNEL, default=DEFAULT_CHANNEL): str,
                }
            ),
            errors=errors,
        )

    async def _test_connection(
        self, api_url: str, api_key: str | None, timeout: int
    ) -> None:
        """Test the API connection."""
        if not api_key:
            api_key = "admin:ciris_admin_password"

        client = CIRISClient(
            base_url=api_url,
            api_key=api_key,
            timeout=float(timeout),
            max_retries=0,
        )

        try:
            async with client:
                if client._transport.api_key and ":" in client._transport.api_key:
                    username, password = client._transport.api_key.split(":", 1)
                    _LOGGER.info("Testing connection with username: %s", username)

                    try:
                        token = await client.auth.login(username, password)
                        _LOGGER.info("Successfully authenticated with CIRIS")
                        client._transport.set_api_key(token.access_token, persist=False)
                    except Exception as e:
                        _LOGGER.error("Failed to authenticate: %s", e)
                        raise

                status = await client.agent.get_status()
                _LOGGER.info(
                    "Connected to CIRIS: %s (state: %s)",
                    status.name,
                    status.cognitive_state,
                )

        except Exception as e:
            _LOGGER.error("Connection test failed: %s", e)
            raise


class CIRISOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for CIRIS."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=self.config_entry.data.get(
                            CONF_TIMEOUT, DEFAULT_TIMEOUT
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                    vol.Optional(
                        CONF_CHANNEL,
                        default=self.config_entry.data.get(
                            CONF_CHANNEL, DEFAULT_CHANNEL
                        ),
                    ): str,
                }
            ),
        )


class CIRISSubentryFlow(config_entries.SubentryFlowHandler):
    """Handle subentry flow for context profiles."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the context profile setup."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Apply preset if selected
            preset = user_input.get("preset")
            if preset and preset in CONTEXT_PRESETS:
                preset_data = CONTEXT_PRESETS[preset].copy()
                # Merge preset with user input (user input takes precedence)
                for key, value in preset_data.items():
                    if key not in user_input or not user_input[key]:
                        user_input[key] = value

            # Validate profile name
            profile_name = user_input.get(CONF_PROFILE_NAME, "").strip()
            if not profile_name:
                errors[CONF_PROFILE_NAME] = "required"
            else:
                return self.async_create_entry(
                    title=profile_name,
                    data=user_input,
                )

        # Build preset options
        preset_options = [
            {"value": "", "label": "Custom (configure manually)"},
            {"value": "kids_room", "label": "Kids Room (safe, playful)"},
            {"value": "elderly_care", "label": "Elderly Care (patient, clear)"},
            {"value": "adult_room", "label": "Adult Room (unrestricted)"},
            {"value": "shared_space", "label": "Shared Space (family-friendly)"},
            {"value": "office", "label": "Office (professional)"},
        ]

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_PROFILE_NAME): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.TEXT)
                    ),
                    vol.Optional("preset", default=""): SelectSelector(
                        SelectSelectorConfig(
                            options=preset_options,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_ROOM_TYPE, default="shared_space"
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=list(ROOM_TYPES.keys()),
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_LANGUAGE, default=DEFAULT_LANGUAGE
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": k, "label": v} for k, v in LANGUAGES.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_SAFETY_LEVEL, default=DEFAULT_SAFETY_LEVEL
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=list(SAFETY_LEVELS.keys()),
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(
                        CONF_RESPONSE_STYLE, default=DEFAULT_RESPONSE_STYLE
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=list(RESPONSE_STYLES.keys()),
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_WAKE_WORD): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.TEXT)
                    ),
                    vol.Optional(CONF_CUSTOM_INSTRUCTIONS, default=""): TextSelector(
                        TextSelectorConfig(
                            type=TextSelectorType.TEXT,
                            multiline=True,
                        )
                    ),
                }
            ),
            errors=errors,
        )
