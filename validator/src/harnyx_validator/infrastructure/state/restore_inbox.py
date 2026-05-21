"""In-memory inbox for validator-owned restore attempts."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock
from uuid import UUID


@dataclass(frozen=True, slots=True)
class _QueuedRestore:
    batch_id: UUID
    available_at: float


class InMemoryRestoreInbox:
    def __init__(self) -> None:
        self._queue: deque[_QueuedRestore] = deque()
        self._queued: set[UUID] = set()
        self._in_flight: set[UUID] = set()
        self._lock = Lock()
        self._not_empty = Condition(self._lock)

    def put_once(self, batch_id: UUID) -> bool:
        with self._not_empty:
            if batch_id in self._queued or batch_id in self._in_flight:
                return False
            self._queue.append(_QueuedRestore(batch_id=batch_id, available_at=time.monotonic()))
            self._queued.add(batch_id)
            self._not_empty.notify()
            return True

    def get(
        self,
        *,
        timeout: float | None = None,
        stop_event: Event | None = None,
    ) -> UUID | None:
        with self._not_empty:
            remaining = timeout
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                now = time.monotonic()
                if self._queue:
                    queued = min(self._queue, key=lambda item: item.available_at)
                    wait_for = queued.available_at - now
                    if wait_for <= 0:
                        self._queue.remove(queued)
                        self._queued.discard(queued.batch_id)
                        self._in_flight.add(queued.batch_id)
                        return queued.batch_id
                    effective_timeout = wait_for if remaining is None else min(remaining, wait_for)
                else:
                    effective_timeout = remaining
                if effective_timeout is not None and effective_timeout <= 0:
                    return None
                start = time.monotonic()
                self._not_empty.wait(effective_timeout)
                if remaining is not None:
                    remaining = max(0.0, remaining - (time.monotonic() - start))

    def mark_done(self, batch_id: UUID) -> None:
        with self._not_empty:
            self._in_flight.discard(batch_id)

    def release_for_retry(self, batch_id: UUID, *, after_seconds: float) -> bool:
        with self._not_empty:
            self._in_flight.discard(batch_id)
            if batch_id in self._queued:
                return False
            self._queue.append(
                _QueuedRestore(
                    batch_id=batch_id,
                    available_at=time.monotonic() + max(0.0, after_seconds),
                )
            )
            self._queued.add(batch_id)
            self._not_empty.notify()
            return True

    def wake(self) -> None:
        with self._not_empty:
            self._not_empty.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._queued) + len(self._in_flight)


__all__ = ["InMemoryRestoreInbox"]
