"""HTTP proxy used by sandboxed agents to call host-provided tools."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from caster_sandbox.json_types import JsonValue

logger = logging.getLogger(__name__)


class ToolInvocationError(RuntimeError):
    """Raised when a tool invocation fails."""


DEFAULT_TOKEN_HEADER = "x-caster-token"  # noqa: S105


class ToolProxy:
    """Thin wrapper around the validator's tool execution endpoint."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        session_id: str,
        endpoint: str = "/rpc/tools/execute",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        token_header: str = DEFAULT_TOKEN_HEADER,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be provided")
        if not token:
            raise ValueError("token must be provided")
        if not session_id:
            raise ValueError("session_id must be provided")
        self._endpoint = endpoint
        self._timeout = timeout
        self._token_header = token_header
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
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
            "session_id": self._session_id,
            "token": self._token,
            "tool": method,
            "args": list(args or []),
            "kwargs": dict(kwargs or {}),
        }
        headers = {self._token_header: self._token}
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


__all__ = [
    "ToolInvocationError",
    "ToolProxy",
]

