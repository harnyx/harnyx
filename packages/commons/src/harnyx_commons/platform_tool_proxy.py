"""Shared platform-tool-proxy protocol constants."""

from __future__ import annotations

from harnyx_commons.tools.types import ToolName
from harnyx_miner_sdk.tools.proxy import PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS

PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS = 350.0
PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS = 60.0
PLATFORM_TOOL_PROXY_LLM_CHAT_DEFAULT_TIMEOUT_SECONDS = 120.0
PLATFORM_TOOL_PROXY_PROVIDER_TIMEOUT_HEADROOM_SECONDS = 10.0
PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_HEADROOM_SECONDS = 30.0
PLATFORM_TOOL_PROXY_PLATFORM_BACKEND_TIMEOUT_SECONDS = 360.0
PLATFORM_TOOL_PROXY_DEFAULT_MAX_EXECUTION_TIMEOUT_SECONDS = (
    PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS - PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_HEADROOM_SECONDS
)
PLATFORM_TOOL_PROXY_STALE_CALL_TIMEOUT_MARGIN_SECONDS = 120.0
PLATFORM_TOOL_PROXY_DEFAULT_TIMEOUT_SECONDS_BY_TOOL: dict[ToolName, float] = {
    "search_web": PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS,
    "search_ai": PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS,
    "fetch_page": PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS,
    "llm_chat": PLATFORM_TOOL_PROXY_LLM_CHAT_DEFAULT_TIMEOUT_SECONDS,
}


def platform_tool_proxy_default_timeout_seconds(tool: ToolName) -> float:
    return PLATFORM_TOOL_PROXY_DEFAULT_TIMEOUT_SECONDS_BY_TOOL[tool]


def platform_tool_proxy_provider_timeout_seconds(effective_tool_timeout_seconds: float) -> float:
    return effective_tool_timeout_seconds + PLATFORM_TOOL_PROXY_PROVIDER_TIMEOUT_HEADROOM_SECONDS


def platform_tool_proxy_live_call_outer_envelope_seconds(max_execution_timeout_seconds: float) -> float:
    return max(
        platform_tool_proxy_provider_timeout_seconds(max_execution_timeout_seconds),
        PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS,
        PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS,
        PLATFORM_TOOL_PROXY_PLATFORM_BACKEND_TIMEOUT_SECONDS,
    )


def platform_tool_proxy_minimum_stale_call_timeout_seconds(max_execution_timeout_seconds: float) -> float:
    return (
        platform_tool_proxy_live_call_outer_envelope_seconds(max_execution_timeout_seconds)
        + PLATFORM_TOOL_PROXY_STALE_CALL_TIMEOUT_MARGIN_SECONDS
    )


__all__ = [
    "PLATFORM_TOOL_PROXY_DEFAULT_TIMEOUT_SECONDS_BY_TOOL",
    "PLATFORM_TOOL_PROXY_DEFAULT_MAX_EXECUTION_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_LLM_CHAT_DEFAULT_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_PLATFORM_BACKEND_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_PROVIDER_TIMEOUT_HEADROOM_SECONDS",
    "PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_HEADROOM_SECONDS",
    "PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS",
    "PLATFORM_TOOL_PROXY_STALE_CALL_TIMEOUT_MARGIN_SECONDS",
    "platform_tool_proxy_default_timeout_seconds",
    "platform_tool_proxy_live_call_outer_envelope_seconds",
    "platform_tool_proxy_minimum_stale_call_timeout_seconds",
    "platform_tool_proxy_provider_timeout_seconds",
]
