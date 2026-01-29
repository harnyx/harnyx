"""sr25519 signature verification for inbound control-plane RPC."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

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
    owner_cache_ttl_seconds: float = 300.0
    _owner_cache: dict[str, tuple[float, str]] = field(default_factory=dict)

    def _resolve_owner_coldkey(self, hotkey_ss58: str) -> str | None:
        now = time.monotonic()
        cached = self._owner_cache.get(hotkey_ss58)
        if cached is not None:
            expires_at, owner = cached
            if now <= expires_at:
                return owner
            del self._owner_cache[hotkey_ss58]

        subtensor = bt.Subtensor(network=self.network)
        try:
            owner = subtensor.get_hotkey_owner(hotkey_ss58)
        finally:
            try:
                subtensor.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("subtensor close failed during inbound auth check")

        if owner is None:
            return None

        resolved = str(owner)
        if len(self._owner_cache) >= 1024:
            self._owner_cache.clear()
        self._owner_cache[hotkey_ss58] = (now + self.owner_cache_ttl_seconds, resolved)
        return resolved

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
        owner_coldkey = self._resolve_owner_coldkey(parsed.ss58)
        if owner_coldkey is None:
            raise VerificationError("unknown_hotkey", "hotkey owner not found on chain")
        if owner_coldkey != self.owner_coldkey_ss58:
            raise VerificationError("not_owner", "caller hotkey is not owned by subnet owner coldkey")
        return parsed.ss58


__all__ = ["BittensorSr25519InboundVerifier"]
