"""HTTP proxy used by sandboxed agents to call host-provided tools."""

from __future__ import annotations

from harnyx_miner_sdk.tools import proxy as _proxy

DEFAULT_TOKEN_HEADER = _proxy.DEFAULT_TOKEN_HEADER
ToolInvocationError = _proxy.ToolInvocationError
ToolProxy = _proxy.ToolProxy


__all__ = [
    "ToolInvocationError",
    "ToolProxy",
]
