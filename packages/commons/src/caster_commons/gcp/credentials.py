"""GCP credential helpers shared across providers and observability."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, cast

from google.oauth2.service_account import Credentials as ServiceAccountCredentials

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def decode_service_account_b64(blob: str, *, source: str) -> str:
    try:
        decoded = base64.b64decode(blob.encode("utf-8"), validate=True)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - invalid config
        raise ValueError(f"{source} is invalid") from exc
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - invalid config
        raise ValueError(f"{source} must decode to UTF-8 JSON") from exc


def load_service_account_info(serialized: str, *, source: str) -> dict[str, Any]:
    try:
        data = json.loads(serialized)
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid config
        raise ValueError(f"{source} is not valid JSON") from exc
    if not isinstance(data, dict):  # pragma: no cover - invalid config
        raise ValueError(f"{source} must be a JSON object")
    return data


def service_account_credentials_from_b64(
    blob: str,
    *,
    source: str,
    scopes: tuple[str, ...] = (_CLOUD_PLATFORM_SCOPE,),
) -> ServiceAccountCredentials:
    serialized = decode_service_account_b64(blob, source=source)
    info = load_service_account_info(serialized, source=source)
    return cast(
        ServiceAccountCredentials,
        ServiceAccountCredentials.from_service_account_info(  # type: ignore[no-untyped-call]
            info,
            scopes=scopes,
        ),
    )


__all__ = [
    "decode_service_account_b64",
    "load_service_account_info",
    "service_account_credentials_from_b64",
    "ServiceAccountCredentials",
]
