"""Authorization header parsing utilities for validator control-plane RPC."""

from __future__ import annotations

from dataclasses import dataclass

from caster_commons.bittensor import ParsedAuthorizationHeader as _Parsed
from caster_commons.bittensor import parse_bittensor_header as _parse


@dataclass(frozen=True, slots=True)
class ParsedAuthorizationHeader(_Parsed):
    """Alias dataclass re-exported for validator imports."""


def parse_bittensor_header(header_value: str) -> ParsedAuthorizationHeader:
    parsed = _parse(header_value)
    return ParsedAuthorizationHeader(
        ss58=parsed.ss58,
        signature_hex=parsed.signature_hex.lower(),
    )


__all__ = ["ParsedAuthorizationHeader", "parse_bittensor_header"]
