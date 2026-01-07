"""Base worker abstraction with shared threading lifecycle."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseWorker(ABC):
    """Abstract background worker with start/stop lifecycle.

    Subclasses must implement ``_tick()`` which is called repeatedly until
    ``stop()`` is invoked. Optionally set ``poll_interval`` to sleep between
    ticks (useful for polling workers vs queue-blocking workers).
    """

    worker_name: ClassVar[str] = "base-worker"
    logger_name: ClassVar[str] = "caster.worker"
    default_poll_interval: ClassVar[float | None] = None  # None = no sleep (blocking workers)

    def __init__(self, *, poll_interval: float | None = None) -> None:
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logger = logging.getLogger(self.logger_name)

    def start(self) -> None:
        """Start the background worker thread (idempotent)."""

        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name=self.worker_name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for termination."""

        if not self._thread:
            return

        self._stop.set()
        self._on_stop_requested()
        self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        """Return True if the worker thread is alive."""

        return bool(self._thread and self._thread.is_alive())

    @property
    def poll_interval(self) -> float | None:
        """Return the poll interval (instance override or class default)."""

        return self._poll_interval if self._poll_interval is not None else self.default_poll_interval

    def _run_loop(self) -> None:
        """Main loop that calls tick() until stopped."""

        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # pragma: no cover - unexpected
                self._logger.exception("worker tick failed")
                self._on_error()

            # Sleep if this is a polling worker
            interval = self.poll_interval
            if interval is not None:
                time.sleep(interval)

    @abstractmethod
    def _tick(self) -> None:
        """Execute one iteration of the worker's task.

        For blocking workers (poll_interval=None), this method should block
        until work is available or stop is signaled.

        For polling workers, this method should check for work and return
        quickly; the base class handles sleeping between ticks.
        """

    def _on_stop_requested(self) -> None:  # noqa: B027
        """Hook called when stop is requested (before join).

        Override to wake any blocking operations (e.g., queue.get()).
        """

    def _on_error(self) -> None:  # noqa: B027
        """Hook called when tick() raises an exception.

        Override for custom error handling (e.g., updating status).
        """


__all__ = ["BaseWorker"]

