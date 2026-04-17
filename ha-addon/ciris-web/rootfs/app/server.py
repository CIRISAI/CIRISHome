"""CIRIS Web UI Server for Home Assistant Add-on.

Serves the CIRIS web UI with HA Supervisor integration.
Uses SUPERVISOR_TOKEN for HA API access.
"""

import asyncio
import json
import logging
import mimetypes
import os
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add WebAssembly MIME type
mimetypes.add_type("application/wasm", ".wasm")

# HA Supervisor environment
SUPERVISOR_TOKEN = os.getenv("SUPERVISOR_TOKEN", "")
HASSIO_API = "http://supervisor"
HA_API = "http://supervisor/core/api"

# Web assets directory
WEB_DIR = Path(os.getenv("CIRIS_WEB_ASSETS", "/app/web"))


class CIRISAddonServer:
    """Web server for CIRIS Add-on."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8099):
        self.host = host
        self.port = port
        self._app = None
        self._runner = None
        self._client = None

    async def start(self):
        """Start the web server."""
        if not WEB_DIR.exists():
            raise RuntimeError(f"Web assets not found at {WEB_DIR}")

        self._client = ClientSession(
            timeout=ClientTimeout(total=30),
            headers=(
                {"Authorization": f"Bearer {SUPERVISOR_TOKEN}"}
                if SUPERVISOR_TOKEN
                else {}
            ),
        )

        self._app = web.Application()
        self._app.router.add_route("*", "/api/{path:.*}", self._proxy_to_ha)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/ingress-config", self._handle_ingress_config)
        self._app.router.add_get("/{path:.*}", self._handle_static)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"CIRIS Web UI running on http://{self.host}:{self.port}")
        logger.info(f"Serving from: {WEB_DIR}")
        logger.info(f"Supervisor token: {'present' if SUPERVISOR_TOKEN else 'missing'}")

    async def stop(self):
        """Stop the server."""
        if self._client:
            await self._client.close()
        if self._runner:
            await self._runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        # Test HA connection
        ha_status = "unknown"
        if self._client and SUPERVISOR_TOKEN:
            try:
                async with self._client.get(f"{HA_API}/") as resp:
                    if resp.status == 200:
                        ha_status = "connected"
                    else:
                        ha_status = f"error:{resp.status}"
            except Exception as e:
                ha_status = f"error:{e}"

        return web.json_response(
            {
                "status": "ok",
                "ha_status": ha_status,
                "supervisor_token": "present" if SUPERVISOR_TOKEN else "missing",
            }
        )

    async def _handle_ingress_config(self, request: web.Request) -> web.Response:
        """Provide config for the web app via ingress."""
        # Get ingress path from headers
        ingress_path = request.headers.get("X-Ingress-Path", "")

        return web.json_response(
            {
                "ingress_path": ingress_path,
                "ha_api": f"{ingress_path}/api",
                "ws_api": f"{ingress_path}/api/websocket",
            }
        )

    async def _handle_static(self, request: web.Request) -> web.Response:
        """Serve static files."""
        path = request.match_info.get("path", "")

        if not path or path == "/":
            path = "index.html"

        file_path = WEB_DIR / path

        # Security check
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(WEB_DIR.resolve())):
                return web.Response(status=403, text="Forbidden")
        except Exception:
            return web.Response(status=400, text="Bad request")

        if not file_path.exists():
            # SPA fallback
            file_path = WEB_DIR / "index.html"
            if not file_path.exists():
                return web.Response(status=404, text="Not found")

        if file_path.is_dir():
            file_path = file_path / "index.html"
            if not file_path.exists():
                return web.Response(status=404, text="Not found")

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            content = file_path.read_bytes()
            headers = {}
            if any(str(file_path).endswith(ext) for ext in [".js", ".wasm", ".css"]):
                headers["Cache-Control"] = "public, max-age=86400"

            return web.Response(
                body=content, content_type=content_type, headers=headers
            )
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return web.Response(status=500, text="Internal server error")

    async def _proxy_to_ha(self, request: web.Request) -> web.Response:
        """Proxy API requests to Home Assistant via Supervisor."""
        if not self._client:
            return web.Response(status=503, text="Client not initialized")

        path = request.match_info.get("path", "")
        target_url = f"{HA_API}/{path}"

        # Forward headers
        headers = dict(request.headers)
        headers.pop("Host", None)
        if SUPERVISOR_TOKEN:
            headers["Authorization"] = f"Bearer {SUPERVISOR_TOKEN}"

        body = None
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.read()

        try:
            async with self._client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as resp:
                response_body = await resp.read()
                response_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower()
                    not in ("content-encoding", "transfer-encoding", "content-length")
                }
                return web.Response(
                    status=resp.status, body=response_body, headers=response_headers
                )
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return web.json_response({"error": str(e)}, status=502)


async def main():
    """Run the server."""
    server = CIRISAddonServer(host="0.0.0.0", port=8099)
    await server.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
