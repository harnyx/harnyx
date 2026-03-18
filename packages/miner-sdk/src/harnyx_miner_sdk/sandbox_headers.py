"""HTTP header names used for host↔sandbox requests."""

from __future__ import annotations

from collections.abc import Mapping

SESSION_ID_HEADER = "x-session-id"
HOST_CONTAINER_URL_HEADER = "x-host-container-url"
PLATFORM_TOKEN_HEADER = "x-platform-token"  # noqa: S105


def _read_header(headers: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = (headers.get(name) or "").strip()
        if value:
            return value
    return ""


def read_session_id_header(headers: Mapping[str, str]) -> str:
    return _read_header(headers, SESSION_ID_HEADER)


def read_host_container_url_header(headers: Mapping[str, str]) -> str:
    return _read_header(headers, HOST_CONTAINER_URL_HEADER)


def read_platform_token_header(headers: Mapping[str, str]) -> str:
    return _read_header(headers, PLATFORM_TOKEN_HEADER)


__all__ = [
    "HOST_CONTAINER_URL_HEADER",
    "PLATFORM_TOKEN_HEADER",
    "SESSION_ID_HEADER",
    "read_host_container_url_header",
    "read_platform_token_header",
    "read_session_id_header",
]
