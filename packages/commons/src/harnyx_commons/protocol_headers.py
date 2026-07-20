"""Shared protocol header constants for service and SDK boundaries."""

from __future__ import annotations

from harnyx_miner_sdk.sandbox_headers import (
    HOST_CONTAINER_URL_HEADER,
    PLATFORM_TOKEN_HEADER,
    SESSION_ID_HEADER,
    read_host_container_url_header,
    read_platform_token_header,
    read_session_id_header,
)

PLATFORM_TOOL_PROXY_TOKEN_HEADER = "x-platform-tool-proxy-token"  # noqa: S105


__all__ = [
    "HOST_CONTAINER_URL_HEADER",
    "PLATFORM_TOKEN_HEADER",
    "PLATFORM_TOOL_PROXY_TOKEN_HEADER",
    "SESSION_ID_HEADER",
    "read_host_container_url_header",
    "read_platform_token_header",
    "read_session_id_header",
]
