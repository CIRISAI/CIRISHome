"""CIRIS Web UI Server.

Serves the CIRIS web UI and proxies requests to Home Assistant.
Can be run standalone or as part of ciris-home CLI.
"""

import asyncio
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

from aiohttp import ClientSession, ClientTimeout, web

from .installer import CIRISConfig, load_config

logger = logging.getLogger(__name__)

# Add WebAssembly MIME type
mimetypes.add_type("application/wasm", ".wasm")


def get_web_assets_dir() -> Optional[Path]:
    """Find the web assets directory.

    Checks in order:
    1. CIRIS_WEB_ASSETS env var
    2. Bundled assets in package (for pip install)
    3. Development build output
    """
    # Check env var
    env_path = os.getenv("CIRIS_WEB_ASSETS")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # Check for bundled assets (pip install)
    package_dir = Path(__file__).parent
    bundled = package_dir / "web"
    if bundled.exists() and (bundled / "index.html").exists():
        return bundled

    # Check for development build
    dev_paths = [
        Path.cwd() / "mobile-web/webApp/build/dist/wasmJs/developmentExecutable",
        Path.cwd() / "mobile-web/webApp/build/dist/wasmJs/productionExecutable",
        package_dir.parent
        / "mobile-web/webApp/build/dist/wasmJs/developmentExecutable",
    ]

    for path in dev_paths:
        if path.exists() and (path / "index.html").exists():
            return path

    return None


class CIRISWebServer:
    """Web server for CIRIS UI with HA proxy."""

    def __init__(
        self,
        config: CIRISConfig,
        host: str = "0.0.0.0",  # nosec B104 - Docker container binding
        port: int = 8099,
        web_dir: Optional[Path] = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.web_dir = web_dir or get_web_assets_dir()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._client: Optional[ClientSession] = None

    async def start(self) -> None:
        """Start the web server."""
        if not self.web_dir or not self.web_dir.exists():
            raise RuntimeError(
                "Web assets not found. Build the KMP web app first:\n"
                "  cd mobile-web && ./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution"
            )

        self._client = ClientSession(timeout=ClientTimeout(total=30))

        self._app = web.Application()
        self._app.router.add_route("*", "/api/{path:.*}", self._proxy_to_ha)
        self._app.router.add_get("/setup.html", self._handle_setup)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/{path:.*}", self._handle_static)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"CIRIS Web UI running at http://{self.host}:{self.port}")
        logger.info(f"Serving from: {self.web_dir}")

    async def stop(self) -> None:
        """Stop the web server."""
        if self._client:
            await self._client.close()
        if self._runner:
            await self._runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "ok",
                "ha_url": self.config.ha_url,
                "has_token": bool(self.config.ha_token),
            }
        )

    async def _handle_setup(self, request: web.Request) -> web.Response:
        """Handle setup page with token injection."""
        # If token/baseUrl in query params, redirect to app
        token = request.query.get("token")
        base_url = request.query.get("baseUrl") or request.query.get("base_url")

        if not token and not base_url:
            # Auto-inject from config
            token = self.config.ha_token
            base_url = self.config.ha_url

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CIRIS Setup</title>
    <style>
        body {{
            background-color: #0A0A0F;
            color: #fff;
            font-family: system-ui, -apple-system, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{ max-width: 500px; text-align: center; }}
        h1 {{ color: #6B9AFF; }}
        .success {{ color: #4CAF50; }}
        .info {{ color: #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CIRIS Setup</h1>
        <div id="status">
            <p class="info">Configuring CIRIS...</p>
        </div>
    </div>
    <script>
        localStorage.setItem('ciris_access_token', '{token or ""}');
        localStorage.setItem('ciris_base_url', '{base_url or ""}');

        document.getElementById('status').innerHTML = `
            <p class="success">Setup complete!</p>
            <p>Connected to: {base_url or "Not configured"}</p>
            <p><a href="/" style="color: #6B9AFF;">Go to CIRIS</a></p>
        `;

        setTimeout(() => window.location.href = '/', 1500);
    </script>
</body>
</html>"""

        return web.Response(text=html, content_type="text/html")

    async def _handle_static(self, request: web.Request) -> web.Response:
        """Serve static files from web assets directory."""
        path = request.match_info.get("path", "")

        if not path or path == "/":
            path = "index.html"

        file_path = self.web_dir / path

        # Security: prevent directory traversal
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(self.web_dir.resolve())):
                return web.Response(status=403, text="Forbidden")
        except Exception:
            return web.Response(status=400, text="Bad request")

        if not file_path.exists():
            # SPA fallback - serve index.html for non-existent paths
            file_path = self.web_dir / "index.html"
            if not file_path.exists():
                return web.Response(status=404, text="Not found")

        if file_path.is_dir():
            file_path = file_path / "index.html"
            if not file_path.exists():
                return web.Response(status=404, text="Not found")

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = "application/octet-stream"

        # Read and return file
        try:
            content = file_path.read_bytes()

            # Add cache headers for static assets
            headers = {}
            if any(str(file_path).endswith(ext) for ext in [".js", ".wasm", ".css"]):
                headers["Cache-Control"] = "public, max-age=86400"

            return web.Response(
                body=content,
                content_type=content_type,
                headers=headers,
            )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return web.Response(status=500, text="Internal server error")

    async def _proxy_to_ha(self, request: web.Request) -> web.Response:
        """Proxy API requests to Home Assistant."""
        if not self._client:
            return web.Response(status=503, text="Client not initialized")

        path = request.match_info.get("path", "")
        target_url = f"{self.config.ha_url}/api/{path}"

        # Forward request headers, add auth
        headers = dict(request.headers)
        headers.pop("Host", None)
        if self.config.ha_token:
            headers["Authorization"] = f"Bearer {self.config.ha_token}"

        # Forward body for POST/PUT/PATCH
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

                # Filter response headers
                response_headers = {}
                for key, value in resp.headers.items():
                    if key.lower() not in (
                        "content-encoding",
                        "transfer-encoding",
                        "content-length",
                    ):
                        response_headers[key] = value

                return web.Response(
                    status=resp.status,
                    body=response_body,
                    headers=response_headers,
                )
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=502,
            )


async def run_server(
    config: Optional[CIRISConfig] = None,
    host: str = "0.0.0.0",  # nosec B104 - intentional for Docker container
    port: int = 8099,
) -> None:
    """Run the web server until interrupted.

    Args:
        config: CIRIS configuration. If None, loads from disk.
        host: Host to bind to
        port: Port to listen on
    """
    if config is None:
        config = load_config()
        if config is None:
            raise RuntimeError(
                "No configuration found. Run 'ciris-home' first to set up."
            )

    server = CIRISWebServer(config, host=host, port=port)
    await server.start()

    print(f"\nCIRIS Web UI: http://localhost:{port}")
    print(f"Setup URL: http://localhost:{port}/setup.html")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


def main():
    """CLI entry point for standalone server."""
    import argparse

    parser = argparse.ArgumentParser(description="CIRIS Web UI Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to"  # nosec B104
    )
    parser.add_argument("--port", type=int, default=8099, help="Port to listen on")
    args = parser.parse_args()

    try:
        asyncio.run(run_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
