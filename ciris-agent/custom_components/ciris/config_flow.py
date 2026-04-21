"""Config flow for CIRIS integration."""

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

from .ciris_ha_client import CIRISClient
from .ciris_sdk.exceptions import CIRISError, CIRISTimeoutError
from .const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_CHANNEL,
    CONF_CUSTOM_INSTRUCTIONS,
    CONF_LANGUAGE,
    CONF_RESPONSE_STYLE,
    CONF_ROOM_TYPE,
    CONF_SAFETY_LEVEL,
    CONF_TIMEOUT,
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

_LOGGER = logging.getLogger(__name__)


class CIRISConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for CIRIS."""

    VERSION = 2

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry):
        """Get the options flow for this handler."""
        return CIRISOptionsFlow(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - API connection setup."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Check if this is auto-config from addon (skip validation)
            # The addon sets api_url to the known-good internal URL
            skip_validation = user_input.get(CONF_API_URL, "").startswith(
                "http://local-ciris_agent:"
            )

            if skip_validation:
                _LOGGER.info("Auto-config detected, skipping connection validation")
            else:
                # Validate the API connection for manual setup
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

            if not errors:
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
            api_key = "admin:ciris_admin_password"  # pragma: allowlist secret

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

        # Get current values
        current_data = {**self.config_entry.data, **self.config_entry.options}

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TIMEOUT,
                        default=current_data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                    vol.Optional(
                        CONF_CHANNEL,
                        default=current_data.get(CONF_CHANNEL, DEFAULT_CHANNEL),
                    ): str,
                    vol.Optional(
                        CONF_ROOM_TYPE,
                        default=current_data.get(CONF_ROOM_TYPE, "shared_space"),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": k, "label": v}
                                for k, v in ROOM_TYPES.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_LANGUAGE,
                        default=current_data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": k, "label": v} for k, v in LANGUAGES.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SAFETY_LEVEL,
                        default=current_data.get(CONF_SAFETY_LEVEL, DEFAULT_SAFETY_LEVEL),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": k, "label": v}
                                for k, v in SAFETY_LEVELS.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_RESPONSE_STYLE,
                        default=current_data.get(CONF_RESPONSE_STYLE, DEFAULT_RESPONSE_STYLE),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": k, "label": v}
                                for k, v in RESPONSE_STYLES.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_CUSTOM_INSTRUCTIONS,
                        default=current_data.get(CONF_CUSTOM_INSTRUCTIONS, ""),
                    ): TextSelector(
                        TextSelectorConfig(
                            type=TextSelectorType.TEXT,
                            multiline=True,
                        )
                    ),
                }
            ),
        )
