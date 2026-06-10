"""HTTP proxy used by sandboxed agents to call host-provided tools."""

from __future__ import annotations

import logging
import socket
from collections.abc import Mapping, Sequence
from typing import cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from harnyx_miner_sdk.json_types import JsonValue
from harnyx_miner_sdk.sandbox_headers import PLATFORM_TOKEN_HEADER, SESSION_ID_HEADER

logger = logging.getLogger(__name__)


class ToolInvocationError(RuntimeError):
    """Raised when a tool invocation fails."""


DEFAULT_TOKEN_HEADER = PLATFORM_TOKEN_HEADER
DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS = 120.0
PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS = 300.0


class ToolProxy:
    """Thin wrapper around a host container tool execution endpoint."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        session_id: str,
        endpoint: str = "/v1/tools/execute",
        timeout: float = DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided")
        if not token:
            raise ValueError("token must be provided")
        if not session_id:
            raise ValueError("session_id must be provided")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._endpoint = endpoint
        self._timeout = timeout
        self._owns_client = client is None
        resolved_base_url = base_url.rstrip("/")
        if client is None:
            resolved_base_url = _resolve_base_url_host(resolved_base_url)
        self._client = client or httpx.AsyncClient(base_url=resolved_base_url, timeout=timeout)
        self._token = token
        self._session_id = session_id

    async def invoke(
        self,
        method: str,
        *,
        args: Sequence[JsonValue] | None = None,
        kwargs: Mapping[str, JsonValue] | None = None,
    ) -> JsonValue:
        """Invoke a host-managed tool."""
        payload = {
            "tool": method,
            "args": list(args or []),
            "kwargs": dict(kwargs or {}),
        }
        headers = {
            DEFAULT_TOKEN_HEADER: self._token,
            SESSION_ID_HEADER: self._session_id,
        }
        endpoint = f"{self._client.base_url}{self._endpoint}"
        logger.info("[ToolProxy] POST %s %s", endpoint, _describe_payload(payload))
        try:
            response = await self._client.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - exercised via integration
            status = exc.response.status_code
            detail = _summarize_error_response(exc.response)
            logger.warning(
                "[ToolProxy] tool invocation failed (status=%s, reason=%s)",
                status,
                detail,
            )
            raise ToolInvocationError(
                f"tool invocation failed with {status}: {detail}",
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - exercised in tests
            raise ToolInvocationError(f"tool invocation failed: {exc}") from exc
        return cast(JsonValue, response.json())

    def close(self) -> None:
        raise RuntimeError("ToolProxy is async-only; use 'await tool_proxy.aclose()'")

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._owns_client:
            await self._client.aclose()

    def __enter__(self) -> ToolProxy:
        raise RuntimeError("ToolProxy is async-only; use 'async with ToolProxy(...)'")

    def __exit__(self, *exc_info: object) -> None:
        raise RuntimeError("ToolProxy is async-only; use 'async with ToolProxy(...)'")

    async def __aenter__(self) -> ToolProxy:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _summarize_error_response(response: httpx.Response) -> str:
    """Return a short string summarizing the server error payload."""
    try:
        data = response.json()
    except ValueError:
        data = response.text
    if isinstance(data, dict) and "detail" in data:
        summary = data["detail"]
    else:
        summary = data
    text = str(summary)
    return text if len(text) <= 500 else text[:500] + "…"


def _describe_payload(payload: Mapping[str, object]) -> str:
    tool = payload.get("tool")
    args = payload.get("args")
    kwargs = payload.get("kwargs")
    arg_count = len(args) if isinstance(args, Sequence) else "?"
    kwargs_keys: object
    if isinstance(kwargs, Mapping):
        kwargs_keys = sorted(kwargs.keys())
    else:
        kwargs_keys = "?"
    return f"tool={tool!r} args={arg_count} kwargs_keys={kwargs_keys}"


def _resolve_base_url_host(base_url: str) -> str:
    parsed = urlsplit(base_url)
    host = parsed.hostname
    if not host or _is_ip_literal(host):
        return base_url

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        resolved_host = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4][0]
    except OSError:
        return base_url

    if not isinstance(resolved_host, str) or not resolved_host:
        return base_url

    if ":" in resolved_host and not resolved_host.startswith("["):
        resolved_host = f"[{resolved_host}]"
    netloc = resolved_host if parsed.port is None else f"{resolved_host}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _is_ip_literal(host: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET, host)
        return True
    except OSError:
        pass

    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        return False


__all__ = [
    "DEFAULT_TOOL_PROXY_TIMEOUT_SECONDS",
    "DEFAULT_TOKEN_HEADER",
    "PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS",
    "ToolInvocationError",
    "ToolProxy",
]
