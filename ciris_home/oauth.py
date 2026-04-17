"""Home Assistant OAuth2 authentication.

Supports two authentication methods:
1. Browser-based OAuth (IndieAuth) - for interactive use
2. Programmatic login - for CLI automation with username/password
"""

import asyncio
import base64
import hashlib
import logging
import secrets
import webbrowser
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class OAuthTokens:
    """OAuth2 tokens from Home Assistant."""

    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    ha_url: str

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "ha_url": self.ha_url,
        }


async def authenticate_with_credentials(
    ha_url: str,
    username: str,
    password: str,
    client_id: str = "http://127.0.0.1:8099",
) -> Optional[OAuthTokens]:
    """Authenticate with Home Assistant using username/password.

    Uses the internal login_flow API:
    1. POST /auth/login_flow - Create login flow
    2. POST /auth/login_flow/{flow_id} - Submit credentials
    3. POST /auth/token - Exchange code for tokens

    Args:
        ha_url: Home Assistant base URL
        username: HA username
        password: HA password
        client_id: OAuth client ID

    Returns:
        OAuthTokens if successful, None otherwise
    """
    base = ha_url.rstrip("/")

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Step 1: Create login flow
            flow_data = {
                "client_id": client_id,
                "handler": ["homeassistant", None],
                "redirect_uri": f"{client_id}/oauth/callback",
            }

            async with session.post(
                f"{base}/auth/login_flow",
                json=flow_data,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(
                        f"Failed to create login flow: {resp.status} - {error}"
                    )
                    return None
                flow_response = await resp.json()

            flow_id = flow_response.get("flow_id")
            if not flow_id:
                logger.error("No flow_id in response")
                return None

            # Step 2: Submit credentials
            creds_data = {
                "username": username,
                "password": password,
                "client_id": client_id,
            }

            async with session.post(
                f"{base}/auth/login_flow/{flow_id}",
                json=creds_data,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Login failed: {resp.status} - {error}")
                    return None
                login_response = await resp.json()

            # Check for errors (wrong password, etc.)
            if login_response.get("type") == "form":
                errors = login_response.get("errors", {})
                if errors:
                    logger.error(f"Login errors: {errors}")
                    return None

            if login_response.get("type") != "create_entry":
                logger.error(f"Unexpected response type: {login_response.get('type')}")
                return None

            auth_code = login_response.get("result")
            if not auth_code:
                logger.error("No authorization code in response")
                return None

            # Step 3: Exchange code for tokens
            token_data = {
                "grant_type": "authorization_code",
                "code": auth_code,
                "client_id": client_id,
            }

            async with session.post(
                f"{base}/auth/token",
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token exchange failed: {resp.status} - {error}")
                    return None
                token_response = await resp.json()

            return OAuthTokens(
                access_token=token_response["access_token"],
                refresh_token=token_response.get("refresh_token", ""),
                token_type=token_response.get("token_type", "Bearer"),
                expires_in=token_response.get("expires_in", 1800),
                ha_url=ha_url,
            )

    except aiohttp.ClientError as e:
        logger.error(f"Connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None


class OAuthCallbackServer:
    """Local HTTP server to receive OAuth callback."""

    def __init__(self, port: int = 8099):
        self.port = port
        self.code: Optional[str] = None
        self.state: Optional[str] = None
        self.error: Optional[str] = None
        self._event = asyncio.Event()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None

    async def start(self) -> None:
        """Start the callback server."""
        self._app = web.Application()
        self._app.router.add_get("/oauth/callback", self._handle_callback)
        self._app.router.add_get("/", self._handle_root)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        logger.info(f"OAuth callback server listening on http://127.0.0.1:{self.port}")

    async def stop(self) -> None:
        """Stop the callback server."""
        if self._runner:
            await self._runner.cleanup()

    async def _handle_root(self, request: web.Request) -> web.Response:
        """Handle root request - redirect to callback info."""
        return web.Response(
            text="CIRIS Home OAuth Server - Waiting for Home Assistant callback...",
            content_type="text/plain",
        )

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handle OAuth callback from Home Assistant."""
        self.code = request.query.get("code")
        self.state = request.query.get("state")
        self.error = request.query.get("error")

        if self.error:
            html = f"""
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>Authentication Failed</h1>
            <p>Error: {self.error}</p>
            <p>You can close this window.</p>
            </body></html>
            """
        elif self.code:
            html = """
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>Authentication Successful!</h1>
            <p>CIRIS Home is now connected to Home Assistant.</p>
            <p>You can close this window and return to the terminal.</p>
            </body></html>
            """
        else:
            html = """
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>Invalid Callback</h1>
            <p>No authorization code received.</p>
            </body></html>
            """

        self._event.set()
        return web.Response(text=html, content_type="text/html")

    async def wait_for_callback(self, timeout: float = 300) -> Optional[str]:
        """Wait for the OAuth callback.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Authorization code if successful, None otherwise
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return self.code
        except asyncio.TimeoutError:
            logger.warning("OAuth callback timeout")
            return None


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge.

    Returns:
        (code_verifier, code_challenge) tuple
    """
    code_verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def build_auth_url(
    ha_url: str,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    """Build the Home Assistant authorization URL.

    Args:
        ha_url: Home Assistant base URL
        client_id: OAuth client ID (must match redirect_uri host)
        redirect_uri: OAuth callback URL
        state: CSRF protection state
        code_challenge: PKCE code challenge

    Returns:
        Full authorization URL
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{ha_url.rstrip('/')}/auth/authorize?{urlencode(params)}"


async def exchange_code_for_tokens(
    ha_url: str,
    code: str,
    client_id: str,
    code_verifier: str,
) -> Optional[OAuthTokens]:
    """Exchange authorization code for access tokens.

    Args:
        ha_url: Home Assistant base URL
        code: Authorization code from callback
        client_id: OAuth client ID
        code_verifier: PKCE code verifier

    Returns:
        OAuthTokens if successful, None otherwise
    """
    token_url = f"{ha_url.rstrip('/')}/auth/token"

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    return OAuthTokens(
                        access_token=token_data["access_token"],
                        refresh_token=token_data.get("refresh_token", ""),
                        token_type=token_data.get("token_type", "Bearer"),
                        expires_in=token_data.get("expires_in", 1800),
                        ha_url=ha_url,
                    )
                else:
                    error = await response.text()
                    logger.error(f"Token exchange failed: {response.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"Token exchange error: {e}")
        return None


async def authenticate_with_browser(
    ha_url: str,
    callback_port: int = 8099,
) -> Optional[OAuthTokens]:
    """Complete OAuth flow using system browser.

    Opens the user's browser for HA authentication, waits for callback,
    and exchanges the code for tokens.

    Args:
        ha_url: Home Assistant base URL
        callback_port: Port for local callback server

    Returns:
        OAuthTokens if successful, None otherwise
    """
    # Generate PKCE challenge
    code_verifier, code_challenge = generate_pkce()
    state = secrets.token_urlsafe(16)

    # Client ID must match callback server host for IndieAuth
    client_id = f"http://127.0.0.1:{callback_port}"
    redirect_uri = f"{client_id}/oauth/callback"

    # Build auth URL
    auth_url = build_auth_url(
        ha_url=ha_url,
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )

    # Start callback server
    server = OAuthCallbackServer(port=callback_port)
    await server.start()

    try:
        print("\nOpening browser for Home Assistant authentication...")
        print(f"If browser doesn't open, visit: {auth_url}\n")

        # Open browser
        webbrowser.open(auth_url)

        # Wait for callback
        print("Waiting for authentication (5 minute timeout)...")
        code = await server.wait_for_callback(timeout=300)

        if not code:
            print("Authentication timed out or failed.")
            return None

        if server.state != state:
            print("State mismatch - possible CSRF attack!")
            return None

        # Exchange code for tokens
        print("Exchanging authorization code for tokens...")
        tokens = await exchange_code_for_tokens(
            ha_url=ha_url,
            code=code,
            client_id=client_id,
            code_verifier=code_verifier,
        )

        if tokens:
            print("Authentication successful!")
        else:
            print("Failed to obtain tokens.")

        return tokens

    finally:
        await server.stop()
