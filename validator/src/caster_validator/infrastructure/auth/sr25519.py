"""sr25519 signature verification for inbound control-plane RPC."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import bittensor as bt

from caster_commons.bittensor import VerificationError, verify_signed_request
from caster_validator.infrastructure.auth.header import parse_bittensor_header

logger = logging.getLogger("caster_validator.auth")


@dataclass(slots=True)
class BittensorSr25519InboundVerifier:
    """Validates sr25519-signed requests from the platform."""

    netuid: int
    network: str
    owner_coldkey_ss58: str

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
            allowed_ss58=None,
            parse_header=parse_bittensor_header,
        )
        subtensor = bt.Subtensor(network=self.network)
        try:
            owner = subtensor.get_hotkey_owner(parsed.ss58)
        finally:
            try:
                subtensor.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("subtensor close failed during inbound auth check")

        if owner is None:
            raise VerificationError("unknown_hotkey", "hotkey owner not found on chain")
        if str(owner) != self.owner_coldkey_ss58:
            raise VerificationError("not_owner", "caller hotkey is not owned by subnet owner coldkey")
        return parsed.ss58


__all__ = ["BittensorSr25519InboundVerifier"]
