"""Home Assistant discovery via mDNS/Zeroconf.

Discovers Home Assistant instances on the local network using:
1. mDNS service discovery (_home-assistant._tcp.local.)
2. Hostname probing (homeassistant.local, hass.local)
3. Environment variable fallback
"""

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded zeroconf
_zeroconf_available: Optional[bool] = None


def _check_zeroconf() -> bool:
    """Check if zeroconf is available."""
    global _zeroconf_available
    if _zeroconf_available is None:
        try:
            import zeroconf  # noqa: F401

            _zeroconf_available = True
        except ImportError:
            _zeroconf_available = False
    return _zeroconf_available


@dataclass
class DiscoveredHA:
    """A discovered Home Assistant instance."""

    name: str
    url: str
    ip: str
    port: int
    source: str  # 'mdns', 'probe', 'env'

    def __str__(self) -> str:
        return f"{self.name} ({self.url})"


class HADiscoveryListener:
    """Zeroconf listener for Home Assistant mDNS service."""

    def __init__(self) -> None:
        self.instances: List[DiscoveredHA] = []

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        """Handle discovered HA service."""
        info = zc.get_service_info(type_, name)
        if not info:
            return

        addresses = info.parsed_addresses()
        if not addresses:
            return

        ip = addresses[0]
        port = info.port
        server = getattr(info, "server", None)
        hostname = server.rstrip(".") if server else ip

        instance = DiscoveredHA(
            name=hostname,
            url=f"http://{ip}:{port}",
            ip=ip,
            port=port,
            source="mdns",
        )
        self.instances.append(instance)
        logger.info(f"[mDNS] Found: {instance}")

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        pass

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        pass


async def discover_via_mdns(timeout: float = 3.0) -> List[DiscoveredHA]:
    """Discover Home Assistant via mDNS service browsing."""
    if not _check_zeroconf():
        logger.info("Zeroconf not available, skipping mDNS discovery")
        return []

    try:
        from zeroconf import ServiceBrowser, Zeroconf

        listener = HADiscoveryListener()
        zc = Zeroconf()

        browser = ServiceBrowser(zc, "_home-assistant._tcp.local.", listener)
        await asyncio.sleep(timeout)

        browser.cancel()
        zc.close()

        return listener.instances
    except Exception as e:
        logger.warning(f"mDNS discovery failed: {e}")
        return []


async def probe_hostname(hostname: str, port: int = 8123) -> Optional[DiscoveredHA]:
    """Probe a hostname to check if it's a Home Assistant instance."""
    import aiohttp

    # Resolve .local hostnames
    ip = hostname
    if hostname.endswith(".local"):
        try:
            ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            # Try mDNS resolution if socket fails
            if _check_zeroconf():
                ip = await _resolve_mdns(hostname)
            if ip == hostname:
                return None

    url = f"http://{ip}:{port}"

    try:
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{url}/api/", allow_redirects=False) as resp:
                # HA returns 401 without auth - that's what we expect
                if resp.status in (200, 401):
                    return DiscoveredHA(
                        name=hostname,
                        url=url,
                        ip=ip,
                        port=port,
                        source="probe",
                    )
    except Exception:
        pass

    return None


async def _resolve_mdns(hostname: str) -> str:
    """Resolve a .local hostname via mDNS."""
    if not _check_zeroconf():
        return hostname

    try:
        from zeroconf import ServiceInfo, Zeroconf

        zc = Zeroconf()
        name = hostname.rstrip(".")

        # Use ServiceInfo to resolve
        info = ServiceInfo(
            "_ciris-probe._tcp.local.",
            f"probe._ciris-probe._tcp.local.",
            server=f"{name}.",
            port=0,
        )

        if info.request(zc, 2000):
            addresses = info.parsed_addresses()
            if addresses:
                zc.close()
                return addresses[0]

        zc.close()
    except Exception:
        pass

    return hostname


async def discover_via_probe() -> List[DiscoveredHA]:
    """Discover Home Assistant by probing common hostnames."""
    hostnames = [
        ("homeassistant.local", 8123),
        ("hass.local", 8123),
        ("homeassistant", 8123),
    ]

    tasks = [probe_hostname(h, p) for h, p in hostnames]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    instances = []
    for result in results:
        if isinstance(result, DiscoveredHA):
            instances.append(result)

    return instances


def discover_from_env() -> List[DiscoveredHA]:
    """Check environment variables for HA configuration."""
    url = os.getenv("HOME_ASSISTANT_URL")
    if url:
        # Parse URL to extract IP/port
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return [
            DiscoveredHA(
                name="Home Assistant (from env)",
                url=url.rstrip("/"),
                ip=parsed.hostname or "localhost",
                port=parsed.port or 8123,
                source="env",
            )
        ]
    return []


async def discover_all(timeout: float = 5.0) -> List[DiscoveredHA]:
    """Discover all Home Assistant instances.

    Tries in order:
    1. mDNS service discovery
    2. Hostname probing
    3. Environment variable

    Returns all unique instances found.
    """
    instances: List[DiscoveredHA] = []
    seen_urls: set = set()

    def add_unique(ha: DiscoveredHA) -> None:
        if ha.url not in seen_urls:
            seen_urls.add(ha.url)
            instances.append(ha)

    # Try mDNS first
    print("Searching for Home Assistant via mDNS...")
    mdns_results = await discover_via_mdns(timeout)
    for ha in mdns_results:
        add_unique(ha)

    # Try hostname probing if nothing found
    if not instances:
        print("Probing common hostnames...")
        probe_results = await discover_via_probe()
        for ha in probe_results:
            add_unique(ha)

    # Check env as fallback
    if not instances:
        env_results = discover_from_env()
        for ha in env_results:
            add_unique(ha)

    return instances
