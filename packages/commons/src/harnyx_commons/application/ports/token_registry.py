"""Port describing runtime access token metadata."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class TokenRegistryPort(Protocol):
    """Tracks hashed tokens and verifies callers."""

    def register(self, session_id: UUID, raw_token: str) -> str:
        """Store the token for ``session_id`` and return its hashed representation."""

    def verify(self, session_id: UUID, presented_token: str) -> bool:
        """Return ``True`` when ``presented_token`` matches the stored token."""

    def get_hash(self, session_id: UUID) -> str | None:
        """Return the stored token hash for ``session_id``."""

    def revoke(self, session_id: UUID) -> None:
        """Remove the stored token metadata, if present."""


__all__ = ["TokenRegistryPort"]
