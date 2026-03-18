"""Shared protocol header constants for service and SDK boundaries."""

from __future__ import annotations

from collections.abc import Mapping

from harnyx_miner_sdk.sandbox_headers import (
    HOST_CONTAINER_URL_HEADER,
    PLATFORM_TOKEN_HEADER,
    SESSION_ID_HEADER,
    read_host_container_url_header,
    read_platform_token_header,
    read_session_id_header,
)

INTERNAL_SECRET_HEADER = "x-internal-secret"  # noqa: S105


def read_internal_secret_header(headers: Mapping[str, str]) -> str:
    return (headers.get(INTERNAL_SECRET_HEADER) or "").strip()


__all__ = [
    "HOST_CONTAINER_URL_HEADER",
    "INTERNAL_SECRET_HEADER",
    "PLATFORM_TOKEN_HEADER",
    "SESSION_ID_HEADER",
    "read_host_container_url_header",
    "read_internal_secret_header",
    "read_platform_token_header",
    "read_session_id_header",
]
