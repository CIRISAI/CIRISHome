"""Type stubs for pytest-homeassistant."""

from typing import Any
from unittest.mock import Mock

def aioclient_mock() -> Mock: ...
def hass() -> Mock: ...
