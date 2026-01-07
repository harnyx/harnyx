"""Client for registering validator endpoints with the platform."""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import bittensor as bt
import httpx

from caster_commons.bittensor import build_canonical_request

logger = logging.getLogger("caster_validator.platform.registration")


class RegistrationError(RuntimeError):
    """Raised when validator registration fails."""


def _log_platform_resolution(platform_base_url: str) -> None:
    parsed = urlsplit(platform_base_url)
    host = parsed.hostname
    if host is None:
        logger.warning(
            "platform base url missing hostname",
            extra={"data": {"platform_base_url": platform_base_url}},
        )
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = sorted({info[4][0] for info in resolved})
        logger.info(
            "platform base url resolved",
            extra={"data": {"platform_host": host, "platform_port": port, "resolved_addrs": addresses}},
        )
    except OSError as exc:
        logger.warning(
            "platform base url resolution failed",
            extra={
                "data": {
                    "platform_host": host,
                    "platform_port": port,
                    "error_type": type(exc).__name__,
                    "errno": exc.errno,
                    "error": str(exc),
                }
            },
        )


@dataclass
class PlatformRegistrationClient:
    platform_base_url: str
    hotkey: bt.Keypair
    timeout_seconds: float = 10.0

    def _signed_header(self, method: str, path_qs: str, body: bytes) -> str:
        canonical = build_canonical_request(method, path_qs, body)
        signature = self.hotkey.sign(canonical)
        return f'Bittensor ss58="{self.hotkey.ss58_address}",sig="{signature.hex()}"'

    def register(self, validator_public_base_url: str) -> None:
        path = "/v1/validators/register"
        body = f'{{"base_url":"{validator_public_base_url}"}}'.encode()
        headers = {
            "Authorization": self._signed_header("POST", path, body),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            with httpx.Client(base_url=self.platform_base_url, timeout=self.timeout_seconds) as client:
                response = client.post(path, content=body, headers=headers)
                response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network path
            raise RegistrationError(
                f"platform registration failed: POST {self.platform_base_url.rstrip('/')}{path}: {exc}"
            ) from exc


def register_with_retry(
    client: PlatformRegistrationClient,
    public_url: str,
    *,
    attempts: int = 3,
    delay_seconds: float = 2.0,
) -> None:
    logger.info(
        "platform registration starting",
        extra={
            "data": {
                "platform_base_url": client.platform_base_url,
                "validator_public_base_url": public_url,
                "attempts": attempts,
                "delay_seconds": delay_seconds,
            }
        },
    )
    _log_platform_resolution(client.platform_base_url)
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            logger.info(
                "platform registration attempt",
                extra={"data": {"attempt": attempt, "attempts": attempts}},
            )
            client.register(public_url)
            logger.info(
                "platform registration succeeded",
                extra={"data": {"attempt": attempt, "attempts": attempts}},
            )
            return
        except Exception as exc:  # pragma: no cover - network path
            cause = exc.__cause__
            last_error = exc
            logger.warning(
                "platform registration attempt failed",
                extra={
                    "data": {
                        "attempt": attempt,
                        "attempts": attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "cause_type": type(cause).__name__ if cause else None,
                        "cause_errno": cause.errno if isinstance(cause, OSError) else None,
                    }
                },
            )
            time.sleep(delay_seconds)
    raise RegistrationError(str(last_error) if last_error else "platform registration failed")


__all__ = ["PlatformRegistrationClient", "register_with_retry", "RegistrationError"]
