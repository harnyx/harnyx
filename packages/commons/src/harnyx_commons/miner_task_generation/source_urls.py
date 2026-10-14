"""Shared HTTP source authority validation without downloading source content."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import SplitResult, urlsplit


class SourceUrlError(ValueError):
    """A citation authority cannot be established as public HTTP(S)."""


def validated_url(url: str) -> SplitResult:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise SourceUrlError("source URL must use HTTP(S)")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise SourceUrlError("source URL authority is invalid or contains credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SourceUrlError("source URL port is invalid") from exc
    if port is not None and not 1 <= port <= 65535:
        raise SourceUrlError("source URL port is invalid")
    return parsed


def public_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SourceUrlError(f"DNS resolution failed: {exc}") from exc
    addresses = tuple(sorted({str(item[4][0]) for item in infos}))
    if not addresses:
        raise SourceUrlError("DNS resolution returned no addresses")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise SourceUrlError("DNS returned an invalid address") from exc
        if not parsed.is_global:
            raise SourceUrlError(f"DNS returned non-public address {address}")
    return addresses
