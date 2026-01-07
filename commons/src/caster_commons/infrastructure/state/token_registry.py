"""In-memory implementation of the token registry port."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import blake2b
from threading import Lock
from uuid import UUID

from caster_commons.application.ports.token_registry import TokenRegistryPort


def _default_hash(data: str) -> str:
    digest = blake2b(data.encode("utf-8"), digest_size=32)
    return digest.hexdigest()


class InMemoryTokenRegistry(TokenRegistryPort):
    """Stores hashed tokens in memory for the lifetime of the process."""

    def __init__(self, *, hash_fn: Callable[[str], str] | None = None) -> None:
        self._hash = hash_fn or _default_hash
        self._tokens: dict[UUID, str] = {}
        self._lock = Lock()

    def register(self, session_id: UUID, raw_token: str) -> str:
        token_hash = self._hash(raw_token)
        with self._lock:
            self._tokens[session_id] = token_hash
        return token_hash

    def verify(self, session_id: UUID, presented_token: str) -> bool:
        presented_hash = self._hash(presented_token)
        with self._lock:
            stored = self._tokens.get(session_id)
        return stored is not None and stored == presented_hash

    def get_hash(self, session_id: UUID) -> str | None:
        with self._lock:
            return self._tokens.get(session_id)

    def revoke(self, session_id: UUID) -> None:
        with self._lock:
            self._tokens.pop(session_id, None)


__all__ = ["InMemoryTokenRegistry"]
