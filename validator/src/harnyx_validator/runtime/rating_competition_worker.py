"""Independent comparison execution with completed-result delivery retries."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from harnyx_commons.rating_competition import RatingJudgment, RatingWork
from harnyx_validator.application.ports.platform import PlatformPort, RatingJudgmentDeliveryRejectedError
from harnyx_validator.application.rating_competition import RatingCompetitionService
from harnyx_validator.infrastructure.observability.sentry import capture_exception

MAX_RETAINED_COMPARISONS = 10
INITIAL_JUDGING_RETRY_SECONDS = 30
MAX_JUDGING_RETRY_SECONDS = 300

logger = logging.getLogger(__name__)


class RatingCompetitionWorker:
    def __init__(
        self, platform: PlatformPort, service: RatingCompetitionService, *, poll_interval_seconds: float = 5.0
    ) -> None:
        self._platform = platform
        self._service = service
        self._poll_interval_seconds = poll_interval_seconds
        self._active: dict[UUID, asyncio.Task[RatingJudgment]] = {}
        self._completed: dict[UUID, RatingJudgment] = {}
        self._delivery_failed: set[UUID] = set()
        self._submissions: dict[UUID, asyncio.Task[None]] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._stop.is_set()

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout)
            finally:
                self._task = None
                pending = (*self._active.values(), *self._submissions.values())
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self._active.clear()
                self._submissions.clear()

    async def tick(self) -> None:
        for comparison_id, task in tuple(self._active.items()):
            if not task.done():
                continue
            del self._active[comparison_id]
            self._completed[comparison_id] = task.result()
        for comparison_id, task in tuple(self._submissions.items()):
            if not task.done():
                continue
            del self._submissions[comparison_id]
            try:
                task.result()
            except RatingJudgmentDeliveryRejectedError as exc:
                capture_exception(exc)
                del self._completed[comparison_id]
                self._delivery_failed.add(comparison_id)
                logger.exception(
                    "rating result delivery permanently rejected; automatic judging and delivery stopped",
                    extra={"comparison_id": str(comparison_id)},
                )
            except Exception as exc:
                capture_exception(exc)
                logger.exception(
                    "rating result delivery failed; retaining completed result",
                    extra={"comparison_id": str(comparison_id)},
                )
            else:
                del self._completed[comparison_id]
        for comparison_id, result in self._completed.items():
            if comparison_id not in self._submissions:
                self._submissions[comparison_id] = asyncio.create_task(self._platform.submit_rating_judgment(result))
        after = None
        while (available := MAX_RETAINED_COMPARISONS - len(self._active) - len(self._completed)) > 0:
            page = await self._platform.poll_rating_comparisons(after=after, limit=available)
            for work in page.items:
                if (
                    work.comparison_id not in self._active
                    and work.comparison_id not in self._completed
                    and work.comparison_id not in self._delivery_failed
                ):
                    self._active[work.comparison_id] = asyncio.create_task(self._judge_with_retries(work))
            if page.next_after is None:
                break
            if page.next_after == after:
                raise ValueError("rating work pagination did not advance")
            after = page.next_after

    async def _judge_with_retries(self, work: RatingWork) -> RatingJudgment:
        delay = INITIAL_JUDGING_RETRY_SECONDS
        while True:
            try:
                return await self._service.judge(work)
            except Exception as exc:
                capture_exception(exc)
                logger.exception(
                    "rating comparison judging failed; retaining work for retry",
                    extra={"comparison_id": str(work.comparison_id), "retry_delay_seconds": delay},
                )
            # The task keeps its capacity slot while waiting; other tasks remain independent.
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_JUDGING_RETRY_SECONDS)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception as exc:
                capture_exception(exc)
                logger.exception("rating comparison polling failed")
            try:
                await asyncio.wait_for(self._stop.wait(), self._poll_interval_seconds)
            except TimeoutError:
                pass
