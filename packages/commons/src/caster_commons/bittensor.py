"""Shared helpers for Bittensor request signing and header parsing."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import bittensor as bt

_CANONICAL_SEPARATOR = "\n"
_HEADER_PATTERN = re.compile(
    r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-fA-F]+)"$'
)


@dataclass(frozen=True)
class ParsedAuthorizationHeader:
    """Parsed components of a Bittensor Authorization header."""

    ss58: str
    signature_hex: str


def build_canonical_request(method: str, path_qs: str, body: bytes) -> bytes:
    """Return the canonical byte string used for sr25519 signatures."""

    normalized_method = (method or "GET").upper()
    normalized_path = path_qs or "/"
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = _CANONICAL_SEPARATOR.join(
        (normalized_method, normalized_path, body_hash)
    )
    return canonical.encode("utf-8")


def parse_bittensor_header(header_value: str) -> ParsedAuthorizationHeader:
    """Parse the Bittensor Authorization header into its components."""

    match = _HEADER_PATTERN.match(header_value.strip())
    if not match:
        raise ValueError("invalid Bittensor Authorization header")

    ss58 = match.group("ss58")
    signature_hex = match.group("sig")
    if not ss58 or not signature_hex:
        raise ValueError("missing ss58 or signature in Authorization header")

    return ParsedAuthorizationHeader(ss58=ss58, signature_hex=signature_hex)


def decode_auth_signature(signature_hex: str) -> bytes:
    """Decode the hex-encoded sr25519 signature."""

    try:
        return bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise ValueError("signature must be hex-encoded") from exc


class VerificationError(Exception):
    """Raised when Bittensor signature verification fails."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def verify_signed_request(
    *,
    method: str,
    path_qs: str,
    body: bytes,
    authorization_header: str | None,
    allowed_ss58: Iterable[str] | None = None,
    parse_header: Callable[[str], ParsedAuthorizationHeader] = parse_bittensor_header,
) -> ParsedAuthorizationHeader:
    """Validate a Bittensor-signed request and return the parsed header."""

    if not authorization_header:
        raise VerificationError("missing_authorization", "Authorization header is required")

    try:
        parsed = parse_header(authorization_header)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError("invalid_authorization_header", "Authorization header is invalid") from exc

    if allowed_ss58 is not None:
        allowed_set = set(allowed_ss58)
        if parsed.ss58 not in allowed_set:
            raise VerificationError("caller_not_allowed", "caller not allowed")

    canonical = build_canonical_request(method, path_qs, body)
    try:
        signature = decode_auth_signature(parsed.signature_hex)
    except ValueError as exc:
        raise VerificationError("invalid_signature_hex", "Signature must be hex-encoded") from exc

    if len(signature) != 64:
        raise VerificationError("invalid_signature_length", "Signature must be 64 bytes")

    try:
        keypair = bt.Keypair(ss58_address=parsed.ss58)
    except Exception as exc:
        raise VerificationError("invalid_ss58", "Hotkey address is invalid") from exc

    if not keypair.verify(canonical, signature):
        raise VerificationError("invalid_signature", "Signature verification failed")

    return parsed


__all__ = [
    "ParsedAuthorizationHeader",
    "build_canonical_request",
    "decode_auth_signature",
    "parse_bittensor_header",
    "VerificationError",
    "verify_signed_request",
]
