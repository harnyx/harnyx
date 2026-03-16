"""HTTP proxy used by sandboxed agents to call host-provided tools."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from harnyx_miner_sdk.tools import proxy as _proxy

DEFAULT_TOKEN_HEADER = _proxy.DEFAULT_TOKEN_HEADER
ToolInvocationError = _proxy.ToolInvocationError
ToolProxy = _proxy.ToolProxy

_ACTIVE_PROXY: ToolProxy | None = None


@contextmanager
def bind_tool_proxy(proxy: ToolProxy) -> Iterator[None]:
    """Bind the provided ToolProxy for the duration of the context."""
    global _ACTIVE_PROXY
    if _ACTIVE_PROXY is not None:
        raise RuntimeError("a ToolProxy is already bound")
    _ACTIVE_PROXY = proxy
    try:
        yield
    finally:
        _ACTIVE_PROXY = None


def reset_tool_proxy() -> None:
    """Clear any bound ToolProxy."""
    global _ACTIVE_PROXY
    _ACTIVE_PROXY = None


def current_tool_proxy() -> ToolProxy:
    """Return the currently bound ToolProxy."""
    proxy = _ACTIVE_PROXY
    if proxy is None:
        raise RuntimeError("no ToolProxy bound in this context")
    return proxy


__all__ = [
    "ToolInvocationError",
    "ToolProxy",
    "bind_tool_proxy",
    "reset_tool_proxy",
    "current_tool_proxy",
]
