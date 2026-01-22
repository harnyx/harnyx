"""Per-token concurrency guards shared across services."""

from __future__ import annotations

import threading
from collections import defaultdict

from caster_commons.errors import ConcurrencyLimitError


class TokenSemaphore:
    """Lightweight counting semaphore for access tokens."""

    def __init__(self, max_parallel_calls: int = 1) -> None:
        if max_parallel_calls <= 0:
            raise ValueError("max_parallel_calls must be positive")
        self._max_parallel_calls = max_parallel_calls
        self._in_flight: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def acquire(self, token: str) -> None:
        """Reserve a permit for the supplied token or raise when exhausted."""
        with self._lock:
            current = self._in_flight[token]
            if current >= self._max_parallel_calls:
                raise ConcurrencyLimitError(
                    f"token {token!r} exceeds {self._max_parallel_calls} concurrent calls",
                )
            self._in_flight[token] = current + 1

    def release(self, token: str) -> None:
        """Release a previously acquired permit."""
        with self._lock:
            current = self._in_flight.get(token)
            if current is None or current == 0:
                raise RuntimeError(f"token {token!r} has no active permits to release")
            if current == 1:
                del self._in_flight[token]
            else:
                self._in_flight[token] = current - 1

    def in_flight(self, token: str) -> int:
        """Return the number of active calls for a token."""
        with self._lock:
            return self._in_flight.get(token, 0)


__all__ = ["TokenSemaphore"]
