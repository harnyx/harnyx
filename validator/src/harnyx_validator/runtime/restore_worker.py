"""Background worker for validator-owned restore attempts."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from uuid import UUID

from harnyx_validator.application.restore_batch import RestoreEvaluationBatch, RestoreStartDecision
from harnyx_validator.infrastructure.state.restore_inbox import InMemoryRestoreInbox

logger = logging.getLogger("harnyx_validator.restore_worker")

_DEFAULT_RESTORE_RETRY_DELAY_SECONDS = 10.0
_DEFAULT_RESTORE_MAX_ATTEMPTS = 3


@dataclass(slots=True)
class RestoreWorker:
    restore_service: RestoreEvaluationBatch
    restore_inbox: InMemoryRestoreInbox
    retry_delay_seconds: float = _DEFAULT_RESTORE_RETRY_DELAY_SECONDS
    max_attempts: int = _DEFAULT_RESTORE_MAX_ATTEMPTS
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _attempts_by_batch: dict[UUID, int] = field(default_factory=dict, init=False, repr=False)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="validator-restore-worker")

    async def stop(self, *, timeout: float = 5.0) -> None:
        task = self._task
        if task is None:
            return
        self._stop.set()
        self.restore_inbox.wake()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        finally:
            self._task = None

    @property
    def running(self) -> bool:
        task = self._task
        return bool(task is not None and not task.done())

    def request_restore(self, decision: RestoreStartDecision) -> bool:
        if not decision.should_start_restore:
            return False
        queued = self.restore_inbox.put_once(decision.batch_id)
        if queued:
            logger.info(
                "validator restore attempt queued",
                extra={"data": {"batch_id": str(decision.batch_id)}},
            )
        return queued

    async def _run(self) -> None:
        while not self._stop.is_set():
            batch_id = await asyncio.to_thread(
                self.restore_inbox.get,
                timeout=self.retry_delay_seconds,
                stop_event=self._stop,
            )
            if batch_id is None:
                continue
            try:
                await asyncio.to_thread(self.restore_service.restore, batch_id)
            except Exception as exc:
                attempts = self._attempts_by_batch.get(batch_id, 0) + 1
                self._attempts_by_batch[batch_id] = attempts
                if attempts >= self.max_attempts:
                    logger.exception(
                        "validator restore failed after retry budget",
                        extra={"data": {"batch_id": str(batch_id), "attempts": attempts}},
                    )
                    self.restore_service.mark_restore_failed(batch_id, error=exc)
                    self.restore_inbox.mark_done(batch_id)
                    self._attempts_by_batch.pop(batch_id, None)
                    continue
                logger.exception(
                    "validator restore failed; retry scheduled",
                    extra={"data": {"batch_id": str(batch_id), "attempts": attempts}},
                )
                self.restore_inbox.release_for_retry(batch_id, after_seconds=self.retry_delay_seconds)
                continue
            self.restore_inbox.mark_done(batch_id)
            self._attempts_by_batch.pop(batch_id, None)


def create_restore_worker(
    *,
    restore_service: RestoreEvaluationBatch,
    restore_inbox: InMemoryRestoreInbox,
) -> RestoreWorker:
    return RestoreWorker(restore_service=restore_service, restore_inbox=restore_inbox)


__all__ = ["RestoreWorker", "create_restore_worker"]
