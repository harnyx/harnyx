"""In-memory FIFO inbox for platform-supplied evaluation batches."""

from __future__ import annotations

import time
from collections import deque
from threading import Condition, Event, Lock

from caster_validator.application.dto.evaluation import EvaluationBatchSpec


class InMemoryBatchInbox:
    """Thread-safe queue for pending evaluation batches."""

    def __init__(self) -> None:
        self._queue: deque[EvaluationBatchSpec] = deque()
        self._lock = Lock()
        self._not_empty = Condition(self._lock)

    def put(self, batch: EvaluationBatchSpec) -> None:
        with self._not_empty:
            self._queue.append(batch)
            self._not_empty.notify()

    def next(self) -> EvaluationBatchSpec | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def get(
        self,
        *,
        timeout: float | None = None,
        stop_event: Event | None = None,
    ) -> EvaluationBatchSpec | None:
        with self._not_empty:
            remaining = timeout
            while not self._queue:
                if stop_event is not None and stop_event.is_set():
                    return None
                if remaining is None:
                    self._not_empty.wait()
                    continue
                if remaining <= 0:
                    return None
                start = time.monotonic()
                self._not_empty.wait(remaining)
                elapsed = time.monotonic() - start
                remaining = max(0.0, remaining - elapsed)
            return self._queue.popleft()

    def peek(self) -> EvaluationBatchSpec | None:
        with self._lock:
            return self._queue[0] if self._queue else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def wake(self) -> None:
        with self._not_empty:
            self._not_empty.notify_all()


__all__ = ["InMemoryBatchInbox"]
