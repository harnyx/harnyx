"""Shared protocol header constants for service and SDK boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from caster_miner_sdk.sandbox_headers import (
    CASTER_HOST_CONTAINER_URL_HEADER,
    CASTER_SESSION_ID_HEADER,
    CASTER_TOKEN_HEADER,
    HOST_CONTAINER_URL_HEADER,
    PLATFORM_TOKEN_HEADER,
    SESSION_ID_HEADER,
    read_host_container_url_header,
    read_platform_token_header,
    read_session_id_header,
)

CASTER_INTERNAL_SECRET_HEADER = "x-caster-internal-secret"  # noqa: S105
INTERNAL_SECRET_HEADER = "x-internal-secret"  # noqa: S105


def read_internal_secret_header(headers: Mapping[str, str]) -> str:
    for name in (CASTER_INTERNAL_SECRET_HEADER, INTERNAL_SECRET_HEADER):
        value = (headers.get(name) or "").strip()
        if value:
            return value
    return ""


__all__ = [
    "CASTER_INTERNAL_SECRET_HEADER",
    "CASTER_HOST_CONTAINER_URL_HEADER",
    "CASTER_SESSION_ID_HEADER",
    "CASTER_TOKEN_HEADER",
    "HOST_CONTAINER_URL_HEADER",
    "INTERNAL_SECRET_HEADER",
    "PLATFORM_TOKEN_HEADER",
    "SESSION_ID_HEADER",
    "read_host_container_url_header",
    "read_internal_secret_header",
    "read_platform_token_header",
    "read_session_id_header",
]
