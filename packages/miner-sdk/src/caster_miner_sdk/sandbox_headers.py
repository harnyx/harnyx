"""HTTP header names used for host↔sandbox requests."""

from __future__ import annotations

from collections.abc import Mapping

CASTER_SESSION_ID_HEADER = "x-caster-session-id"
SESSION_ID_HEADER = "x-session-id"
CASTER_HOST_CONTAINER_URL_HEADER = "x-caster-host-container-url"
HOST_CONTAINER_URL_HEADER = "x-host-container-url"

# Backward-compatible aliases.
SANDBOX_SESSION_ID_HEADER = CASTER_SESSION_ID_HEADER
SANDBOX_HOST_CONTAINER_URL_HEADER = CASTER_HOST_CONTAINER_URL_HEADER


def _read_header(headers: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = (headers.get(name) or "").strip()
        if value:
            return value
    return ""


def read_session_id_header(headers: Mapping[str, str]) -> str:
    return _read_header(headers, CASTER_SESSION_ID_HEADER, SESSION_ID_HEADER)


def read_host_container_url_header(headers: Mapping[str, str]) -> str:
    return _read_header(headers, CASTER_HOST_CONTAINER_URL_HEADER, HOST_CONTAINER_URL_HEADER)


__all__ = [
    "CASTER_HOST_CONTAINER_URL_HEADER",
    "CASTER_SESSION_ID_HEADER",
    "HOST_CONTAINER_URL_HEADER",
    "SANDBOX_HOST_CONTAINER_URL_HEADER",
    "SANDBOX_SESSION_ID_HEADER",
    "SESSION_ID_HEADER",
    "read_host_container_url_header",
    "read_session_id_header",
]
