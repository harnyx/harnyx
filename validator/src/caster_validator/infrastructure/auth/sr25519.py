"""sr25519 signature verification for inbound control-plane RPC."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from caster_commons.bittensor import verify_signed_request
from caster_validator.infrastructure.auth.header import parse_bittensor_header


@dataclass(slots=True)
class BittensorSr25519InboundVerifier:
    """Validates sr25519-signed requests from the platform."""

    allowed_ss58: frozenset[str]

    @classmethod
    def from_allowed(cls, allowed: Iterable[str]) -> BittensorSr25519InboundVerifier:
        return cls(allowed_ss58=frozenset(allowed))

    def verify(
        self,
        *,
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        parsed = verify_signed_request(
            method=method,
            path_qs=path_qs,
            body=body,
            authorization_header=authorization_header,
            allowed_ss58=self.allowed_ss58,
            parse_header=parse_bittensor_header,
        )
        return parsed.ss58


__all__ = ["BittensorSr25519InboundVerifier"]
