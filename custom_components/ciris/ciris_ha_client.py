"""
CIRIS SDK Client wrapper for Home Assistant.

This wrapper modifies the SDK to avoid blocking operations in the event loop.
"""
import logging
from typing import Optional

# Import the CIRIS SDK
from .ciris_sdk.client import CIRISClient as SDKCIRISClient
from .ciris_sdk.exceptions import CIRISError, CIRISTimeoutError

_LOGGER = logging.getLogger(__name__)


class CIRISClient(SDKCIRISClient):
    """CIRIS client wrapper that's safe for Home Assistant's event loop."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 0,
        **kwargs
    ):
        """Initialize client with HA-safe defaults."""
        # Initialize parent but disable auth store to avoid file I/O
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            use_auth_store=False,  # Disable auth store to avoid file I/O
            rate_limit=False,  # Disable rate limiting for now
            **kwargs
        )
    
    async def __aenter__(self):
        """Enter async context with SSL workaround."""
        # Create client with SSL verification disabled to avoid blocking
        import httpx
        _LOGGER.info(f"Creating httpx client with base URL: {self._transport.base_url}")
        self._transport._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._transport.timeout),
            verify=False,  # Disable SSL verification to avoid blocking
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context."""
        if self._transport._client:
            await self._transport._client.aclose()
            self._transport._client = None