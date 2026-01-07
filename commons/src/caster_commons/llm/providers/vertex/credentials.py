"""Credential handling helpers for Vertex provider."""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path
from typing import Any

from google.oauth2.service_account import Credentials as ServiceAccountCredentials

from caster_commons.gcp.credentials import decode_service_account_b64, load_service_account_info

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_SERVICE_ACCOUNT_B64_SOURCE = "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64"


def prepare_credentials(
    credentials_path: str | None,
    service_account_b64: str | None,
) -> tuple[ServiceAccountCredentials | None, str | None]:
    """Load credentials from file or base64 payload, returning creds and temp path."""
    credentials_info: dict[str, Any] | None = None
    credential_path: str | None = None

    if credentials_path:
        credential_path = credentials_path
    elif service_account_b64:
        serialized = decode_service_account_b64(service_account_b64, source=_SERVICE_ACCOUNT_B64_SOURCE)
        credentials_info = load_service_account_info(serialized, source=_SERVICE_ACCOUNT_B64_SOURCE)
        credential_path = _persist_credentials(serialized)
    else:
        credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if credential_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

    if credentials_info is not None:
        return (
            ServiceAccountCredentials.from_service_account_info(  # type: ignore[no-untyped-call]
                credentials_info,
                scopes=(_CLOUD_PLATFORM_SCOPE,),
            ),
            credential_path,
        )
    if credential_path:
        return (
            ServiceAccountCredentials.from_service_account_file(  # type: ignore[no-untyped-call]
                credential_path,
                scopes=(_CLOUD_PLATFORM_SCOPE,),
            ),
            credential_path,
        )
    return None, None


def cleanup_credentials_file(path: str | None, logger: Any | None = None) -> None:
    """Best-effort removal of temporary credential file."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - cleanup best effort
        if logger:
            logger.debug("Failed to remove temporary Vertex credentials: %s", exc)


def _persist_credentials(serialized: str) -> str:
    with tempfile.NamedTemporaryFile(
        prefix="vertex-sa-",
        suffix=".json",
        delete=False,
    ) as temp_file:
        data = serialized.encode("utf-8")
        temp_file.write(data)
        temp_file.flush()
        path = temp_file.name
        Path(path).chmod(0o600)
        atexit.register(cleanup_credentials_file, path)
        return path


__all__ = ["prepare_credentials", "cleanup_credentials_file", "ServiceAccountCredentials"]
