"""Background worker for platform-owned miner-task assignments."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from harnyx_validator.application.assigned_work import AssignedArtifactWork
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptTerminalEffect,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
)
from harnyx_validator.application.ports.platform import PlatformPort, PlatformTaskAttemptIdentity
from harnyx_validator.infrastructure.observability.sentry import capture_exception

logger = logging.getLogger("harnyx_validator.platform_work_worker")
_PLATFORM_WORK_DISPATCH_START_LEASE_SECONDS = 300.0
"""Bounded dispatch-start lease.

This applies only after an assigned artifact has started and a specific
assignment is dispatchable but still queued.
"""

ArtifactAssignmentExecutor = Callable[
    [
        UUID,
        AssignedArtifactWork,
        asyncio.Event,
        asyncio.Queue[PlatformOwnedTaskResult],
    ],
    Coroutine[Any, Any, None],
]
_AssignmentKey = tuple[UUID, UUID, UUID, int]
_ArtifactGroupKey = tuple[UUID, UUID]


class _ArtifactGroupState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"


class _AssignmentState(StrEnum):
    STARTUP_RESERVED = "startup_reserved"
    DISPATCHABLE_QUEUED = "dispatchable_queued"
    STARTED = "started"


@dataclass(slots=True)
class _AssignmentRecord:
    assignment: MinerTaskWorkAssignment
    reserved_at: float
    state: _AssignmentState
    dispatchable_at: float | None = None
    session_id: UUID | None = None
    started_at: datetime | None = None


@dataclass(slots=True)
class _ClaimedAssignment:
    group: _AssignedArtifactGroup
    key: _AssignmentKey
    record: _AssignmentRecord
    closed: bool = False

    @property
    def assignment(self) -> MinerTaskWorkAssignment:
        return self.record.assignment

    def mark_started(self, validator_session_id: UUID, *, started_at: datetime | None = None) -> None:
        if started_at is None:
            started_at = datetime.now(UTC)
        self.group._finish_starting_claim(self, validator_session_id, started_at=started_at)

    def fail_before_start(self, result: PlatformOwnedTaskResult) -> None:
        self.group._fail_starting_claim(self, result)


@dataclass(slots=True)
class _AssignedArtifactGroup:
    artifact_id: UUID
    state: _ArtifactGroupState
    assignment_queue: asyncio.Queue[MinerTaskWorkAssignment]
    close_requested: asyncio.Event
    result_queue: asyncio.Queue[PlatformOwnedTaskResult]
    assignment_records: dict[_AssignmentKey, _AssignmentRecord]
    starting_records: dict[_AssignmentKey, _AssignmentRecord]
    monotonic_clock: Callable[[], float]
    task: asyncio.Task[None] | None = None
    dispatch_ready: bool = False
    deferred_delivery_failure: PlatformOwnedTaskResult | None = None

    def put_nowait(self, assignment: MinerTaskWorkAssignment) -> bool:
        key = _assignment_key(assignment)
        if key in self.assignment_records or key in self.starting_records:
            return False
        now = self.monotonic_clock()
        state = _AssignmentState.DISPATCHABLE_QUEUED if self.dispatch_ready else _AssignmentState.STARTUP_RESERVED
        self.assignment_records[key] = _AssignmentRecord(
            assignment=assignment,
            reserved_at=now,
            state=state,
            dispatchable_at=now if self.dispatch_ready else None,
        )
        self.assignment_queue.put_nowait(assignment)
        return True

    async def take_for_startup(self) -> MinerTaskWorkAssignment:
        return await self.assignment_queue.get()

    def take_nowait_for_startup(self) -> MinerTaskWorkAssignment:
        return self.assignment_queue.get_nowait()

    def drain_for_setup_failure(self) -> tuple[MinerTaskWorkAssignment, ...]:
        return _drain_assignment_queue(self.assignment_queue)

    def mark_dispatch_ready(self) -> None:
        if self.dispatch_ready:
            return
        self.dispatch_ready = True
        now = self.monotonic_clock()
        for record in self.assignment_records.values():
            if record.state is _AssignmentState.STARTUP_RESERVED:
                record.state = _AssignmentState.DISPATCHABLE_QUEUED
                record.dispatchable_at = now

    def claim_initial_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> _ClaimedAssignment | None:
        return self._claim_for_dispatch(assignment)

    async def claim_for_dispatch(self) -> _ClaimedAssignment:
        while True:
            assignment = await self.assignment_queue.get()
            claimed = self._claim_for_dispatch(assignment)
            if claimed is not None:
                return claimed

    def claim_nowait_for_dispatch(self) -> _ClaimedAssignment:
        while True:
            assignment = self.assignment_queue.get_nowait()
            claimed = self._claim_for_dispatch(assignment)
            if claimed is not None:
                return claimed

    def release_expired_queued(self, *, now: float, dispatch_start_lease_seconds: float) -> int:
        expired_keys = frozenset(
            key
            for key, record in self.assignment_records.items()
            if record.state is _AssignmentState.DISPATCHABLE_QUEUED
            and record.dispatchable_at is not None
            and now - record.dispatchable_at > dispatch_start_lease_seconds
        )
        if not expired_keys:
            return 0
        removed_keys = _remove_assignments_from_queue(self.assignment_queue, expired_keys)
        for key in removed_keys:
            self.assignment_records.pop(key, None)
        return len(removed_keys)

    def reportable_identities(self) -> tuple[PlatformTaskAttemptIdentity, ...]:
        active_identities = tuple(
            _assignment_identity(record.assignment, session_id=record.session_id)
            for record in self.assignment_records.values()
        )
        if self.deferred_delivery_failure is None:
            return active_identities
        return active_identities + (
            _result_identity(self.deferred_delivery_failure),
        )

    def local_inflight_count(self) -> int:
        return len(self.assignment_records) + len(self.starting_records)

    def clear_assignments(self) -> None:
        self.assignment_records.clear()
        self.starting_records.clear()
        _drain_assignment_queue(self.assignment_queue)

    def close_for_delivery_failure(self, result: PlatformOwnedTaskResult) -> None:
        self.deferred_delivery_failure = result
        self.state = _ArtifactGroupState.CLOSING
        self.close_requested.set()
        self.assignment_records.pop(_result_key(result), None)
        self._clear_not_started_assignments()

    def flush_deferred_delivery_failure_if_idle(self) -> PlatformOwnedTaskResult | None:
        if self.local_inflight_count():
            return None
        return self.pop_deferred_delivery_failure()

    def pop_deferred_delivery_failure(self) -> PlatformOwnedTaskResult | None:
        result = self.deferred_delivery_failure
        self.deferred_delivery_failure = None
        return result

    def _claim_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> _ClaimedAssignment | None:
        key = _assignment_key(assignment)
        record = self.assignment_records.get(key)
        if record is None or record.state is not _AssignmentState.DISPATCHABLE_QUEUED:
            return None
        self.assignment_records.pop(key)
        self.starting_records[key] = record
        return _ClaimedAssignment(group=self, key=key, record=record)

    def _finish_starting_claim(
        self,
        claimed: _ClaimedAssignment,
        validator_session_id: UUID,
        *,
        started_at: datetime,
    ) -> None:
        if claimed.closed:
            raise RuntimeError("claimed assignment already closed")
        record = self.starting_records.get(claimed.key)
        if record is None:
            raise RuntimeError("claimed assignment is no longer starting")
        record.state = _AssignmentState.STARTED
        record.session_id = validator_session_id
        record.started_at = started_at
        self.assignment_records[claimed.key] = record
        self.starting_records.pop(claimed.key, None)
        claimed.closed = True

    def _fail_starting_claim(self, claimed: _ClaimedAssignment, result: PlatformOwnedTaskResult) -> None:
        if claimed.closed:
            return
        self.result_queue.put_nowait(result)
        self.starting_records.pop(claimed.key, None)
        claimed.closed = True

    def _clear_not_started_assignments(self) -> None:
        self.starting_records.clear()
        self.assignment_records = {
            key: record
            for key, record in self.assignment_records.items()
            if record.state is _AssignmentState.STARTED
        }
        _drain_assignment_queue(self.assignment_queue)


class PlatformWorkWorker:
    worker_name = "validator-platform-work-worker"

    def __init__(
        self,
        *,
        platform: PlatformPort,
        execute_artifact_assignments: ArtifactAssignmentExecutor,
        target_concurrency: int,
        max_active_artifacts: int,
        poll_interval_seconds: float = 1.0,
        dispatch_start_lease_seconds: float = _PLATFORM_WORK_DISPATCH_START_LEASE_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if target_concurrency < 1:
            raise ValueError("target_concurrency must be positive")
        if max_active_artifacts < 1:
            raise ValueError("max_active_artifacts must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if dispatch_start_lease_seconds <= 0:
            raise ValueError("dispatch_start_lease_seconds must be positive")
        self._platform = platform
        self._execute_artifact_assignments = execute_artifact_assignments
        self._target_concurrency = target_concurrency
        self._max_active_artifacts = max_active_artifacts
        self._poll_interval_seconds = poll_interval_seconds
        self._dispatch_start_lease_seconds = dispatch_start_lease_seconds
        self._monotonic_clock = monotonic_clock
        self._active_artifacts: dict[_ArtifactGroupKey, _AssignedArtifactGroup] = {}
        self._pending_results: list[PlatformOwnedTaskResult] = []
        self._work_request_task: asyncio.Task[tuple[MinerTaskWorkAssignment, ...]] | None = None
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=self.worker_name)

    async def stop(self, *, timeout: float = 5.0) -> None:
        task = self._task
        if task is None:
            return
        self._stop.set()
        self._request_all_artifact_groups_close()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        finally:
            await self._cancel_work_request_task()
            await self._cancel_artifact_group_tasks()
            self._task = None

    @property
    def running(self) -> bool:
        task = self._task
        return bool(task is not None and not task.done())

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                logger.exception("platform work worker iteration failed")
                capture_exception(exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval_seconds)
            except TimeoutError:
                continue

    async def run_once(self) -> None:
        self._collect_artifact_group_results()
        self._release_expired_dispatchable_assignments()
        self._collect_closed_artifact_groups()
        if self._pending_results:
            if not await self._submit_pending_results():
                return
        refilled_artifact_ids = self._consume_completed_work_request()
        self._request_idle_artifact_groups_close(refilled_artifact_ids)
        free_slots = self._target_concurrency - self._local_inflight_count()
        if free_slots <= 0:
            return
        if not self._can_request_platform_work():
            return
        if self._work_request_task is None:
            self._start_work_request()

    def _start_work_request(self) -> None:
        self._work_request_task = asyncio.create_task(
            self._request_platform_work(),
            name="validator-platform-work-request",
        )

    async def _request_platform_work(self) -> tuple[MinerTaskWorkAssignment, ...]:
        active_attempts = self._active_attempts()
        return await self._platform.request_miner_task_work(
            target_concurrency=self._target_concurrency,
            max_active_artifacts=self._max_active_artifacts,
            active_attempts=active_attempts,
        )

    def _consume_completed_work_request(self) -> set[_ArtifactGroupKey]:
        task = self._work_request_task
        if task is None or not task.done():
            return set()
        self._work_request_task = None
        try:
            assignments = task.result()
        except Exception as exc:
            logger.warning("platform work request failed; will retry", exc_info=exc)
            return set()
        free_slots = self._target_concurrency - self._local_inflight_count()
        if free_slots <= 0:
            return set()
        return self._enqueue_assignments(assignments[:free_slots])

    async def _cancel_work_request_task(self) -> None:
        task = self._work_request_task
        if task is None or task.done():
            self._work_request_task = None
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._work_request_task = None

    def _enqueue_assignments(
        self,
        assignments: Sequence[MinerTaskWorkAssignment],
    ) -> set[_ArtifactGroupKey]:
        refilled_group_keys: set[_ArtifactGroupKey] = set()
        for group_key, artifact_assignments in _group_assignments_by_batch_artifact(assignments):
            existing = self._active_artifacts.get(group_key)
            if existing is not None:
                if existing.state is _ArtifactGroupState.OPEN:
                    if self._enqueue_into_group(existing, artifact_assignments):
                        refilled_group_keys.add(group_key)
                continue

            if self._active_artifact_capacity_count() >= self._max_active_artifacts:
                continue

            _batch_id, artifact_id = group_key
            group = self._start_artifact_group(group_key, artifact_id)
            if self._enqueue_into_group(group, artifact_assignments):
                refilled_group_keys.add(group_key)
        return refilled_group_keys

    def _start_artifact_group(
        self,
        group_key: _ArtifactGroupKey,
        artifact_id: UUID,
    ) -> _AssignedArtifactGroup:
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment] = asyncio.Queue()
        close_requested = asyncio.Event()
        result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()

        group = _AssignedArtifactGroup(
            artifact_id=artifact_id,
            state=_ArtifactGroupState.OPEN,
            assignment_queue=assignment_queue,
            close_requested=close_requested,
            result_queue=result_queue,
            assignment_records={},
            starting_records={},
            monotonic_clock=self._monotonic_clock,
        )
        group.task = asyncio.create_task(
            self._execute_artifact_assignments(
                artifact_id,
                group,
                close_requested,
                result_queue,
            ),
            name=f"platform-work-artifact-{artifact_id}",
        )
        self._active_artifacts[group_key] = group
        return group

    def _enqueue_into_group(
        self,
        group: _AssignedArtifactGroup,
        assignments: Sequence[MinerTaskWorkAssignment],
    ) -> bool:
        enqueued = False
        for assignment in assignments:
            enqueued = group.put_nowait(assignment) or enqueued
        return enqueued

    def _can_request_platform_work(self) -> bool:
        if any(group.starting_records for group in self._active_artifacts.values()):
            return False
        if any(group.state is _ArtifactGroupState.OPEN for group in self._active_artifacts.values()):
            return True
        return self._active_artifact_capacity_count() < self._max_active_artifacts

    def _request_idle_artifact_groups_close(self, refilled_group_keys: set[_ArtifactGroupKey]) -> None:
        for group_key, group in tuple(self._active_artifacts.items()):
            if group.state is not _ArtifactGroupState.OPEN:
                continue
            if group.local_inflight_count():
                continue
            if group_key in refilled_group_keys:
                continue
            group.state = _ArtifactGroupState.CLOSING
            group.close_requested.set()

    def _release_expired_dispatchable_assignments(self) -> None:
        now = self._monotonic_clock()
        for group_key, group in tuple(self._active_artifacts.items()):
            if group.state is not _ArtifactGroupState.OPEN:
                continue
            if group.task is not None and group.task.done():
                continue
            removed_count = group.release_expired_queued(
                now=now,
                dispatch_start_lease_seconds=self._dispatch_start_lease_seconds,
            )
            if removed_count:
                logger.warning(
                    "released platform artifact assignments that stayed queued after dispatch readiness",
                    extra={
                        "batch_id": str(group_key[0]),
                        "artifact_id": str(group_key[1]),
                        "assignment_count": removed_count,
                        "dispatch_start_lease_seconds": self._dispatch_start_lease_seconds,
                    },
                )

    def _request_all_artifact_groups_close(self) -> None:
        for group in self._active_artifacts.values():
            group.state = _ArtifactGroupState.CLOSING
            group.close_requested.set()

    async def _cancel_artifact_group_tasks(self) -> None:
        tasks = tuple(
            group.task
            for group in self._active_artifacts.values()
            if group.task is not None and not group.task.done()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_artifacts.clear()

    def _collect_artifact_group_results(self) -> None:
        for group in tuple(self._active_artifacts.values()):
            self._collect_results_for_group(group)

    def _collect_results_for_group(self, group: _AssignedArtifactGroup) -> None:
        while True:
            try:
                result = group.result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if _is_delivery_failure_result(result):
                group.close_for_delivery_failure(result)
                continue
            key = _result_key(result)
            group.assignment_records.pop(key, None)
            self._pending_results.append(result)
        deferred_delivery_failure = group.flush_deferred_delivery_failure_if_idle()
        if deferred_delivery_failure is not None:
            self._pending_results.append(deferred_delivery_failure)

    def _collect_closed_artifact_groups(self) -> None:
        for group_key, group in tuple(self._active_artifacts.items()):
            if group.task is None or not group.task.done():
                continue
            self._collect_results_for_group(group)
            try:
                group.task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception(
                    "platform artifact assignment executor failed",
                    extra={
                        "batch_id": str(group_key[0]),
                        "artifact_id": str(group_key[1]),
                    },
                )
                capture_exception(exc)
            local_inflight_count = group.local_inflight_count()
            if local_inflight_count:
                logger.warning(
                    "clearing platform artifact assignments after executor closed",
                    extra={
                        "batch_id": str(group_key[0]),
                        "artifact_id": str(group_key[1]),
                        "assignment_count": local_inflight_count,
                    },
                )
                group.clear_assignments()
                deferred_delivery_failure = group.pop_deferred_delivery_failure()
                if deferred_delivery_failure is not None:
                    self._pending_results.append(deferred_delivery_failure)
            del self._active_artifacts[group_key]

    async def _submit_pending_results(self) -> bool:
        try:
            acknowledgements = await asyncio.to_thread(
                self._platform.submit_miner_task_work_results,
                tuple(self._pending_results),
            )
        except Exception as exc:
            logger.warning("platform result submission failed; will retry", exc_info=exc)
            return False

        for ack in acknowledgements:
            if ack.outcome == "rejected":
                logger.warning(
                    "platform rejected miner task result",
                    extra={
                        "batch_id": str(ack.batch_id),
                        "artifact_id": str(ack.artifact_id),
                        "task_id": str(ack.task_id),
                        "attempt_number": ack.attempt_number,
                        "reason_code": ack.reason_code,
                        "reason": ack.reason,
                    },
                )

        acknowledged = {
            (ack.batch_id, ack.artifact_id, ack.task_id, ack.attempt_number)
            for ack in acknowledgements
        }
        self._pending_results = [
            result
            for result in self._pending_results
            if (result.batch_id, result.artifact_id, result.task_id, result.attempt_number) not in acknowledged
        ]
        return not self._pending_results

    def _active_attempts(self) -> tuple[PlatformTaskAttemptIdentity, ...]:
        pending = tuple(
            PlatformTaskAttemptIdentity(
                batch_id=result.batch_id,
                artifact_id=result.artifact_id,
                task_id=result.task_id,
                attempt_number=result.attempt_number,
                validator_session_id=result.terminal_attempt.validator_session_id,
            )
            for result in self._pending_results
        )
        active = tuple(
            identity
            for group in self._active_artifacts.values()
            for identity in group.reportable_identities()
        )
        return active + pending

    def _local_inflight_count(self) -> int:
        return sum(group.local_inflight_count() for group in self._active_artifacts.values()) + len(
            self._pending_results
        )

    def _active_artifact_capacity_count(self) -> int:
        return sum(
            1
            for group in self._active_artifacts.values()
            if group.state is _ArtifactGroupState.OPEN or group.local_inflight_count()
        )


def _group_assignments_by_batch_artifact(
    assignments: Sequence[MinerTaskWorkAssignment],
) -> tuple[tuple[_ArtifactGroupKey, tuple[MinerTaskWorkAssignment, ...]], ...]:
    grouped: dict[_ArtifactGroupKey, list[MinerTaskWorkAssignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.batch_id, assignment.artifact.artifact_id)].append(assignment)
    return tuple((group_key, tuple(items)) for group_key, items in grouped.items())


def _assignment_key(assignment: MinerTaskWorkAssignment) -> _AssignmentKey:
    return (
        assignment.batch_id,
        assignment.artifact.artifact_id,
        assignment.task.task_id,
        assignment.attempt_number,
    )


def _assignment_identity(
    assignment: MinerTaskWorkAssignment,
    *,
    session_id: UUID | None,
) -> PlatformTaskAttemptIdentity:
    return PlatformTaskAttemptIdentity(
        batch_id=assignment.batch_id,
        artifact_id=assignment.artifact.artifact_id,
        task_id=assignment.task.task_id,
        attempt_number=assignment.attempt_number,
        validator_session_id=session_id,
    )


def _result_identity(result: PlatformOwnedTaskResult) -> PlatformTaskAttemptIdentity:
    return PlatformTaskAttemptIdentity(
        batch_id=result.batch_id,
        artifact_id=result.artifact_id,
        task_id=result.task_id,
        attempt_number=result.attempt_number,
        validator_session_id=result.terminal_attempt.validator_session_id,
    )


def _drain_assignment_queue(
    assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
    removed_keys: frozenset[_AssignmentKey] | None = None,
) -> tuple[MinerTaskWorkAssignment, ...]:
    assignments: list[MinerTaskWorkAssignment] = []
    while True:
        try:
            assignment = assignment_queue.get_nowait()
        except asyncio.QueueEmpty:
            return tuple(assignments)
        if removed_keys is None or _assignment_key(assignment) in removed_keys:
            assignments.append(assignment)


def _remove_assignments_from_queue(
    assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
    removed_keys: frozenset[_AssignmentKey],
) -> frozenset[_AssignmentKey]:
    removed: set[_AssignmentKey] = set()
    retained: list[MinerTaskWorkAssignment] = []
    while True:
        try:
            assignment = assignment_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        key = _assignment_key(assignment)
        if key in removed_keys:
            removed.add(key)
        else:
            retained.append(assignment)
    for assignment in retained:
        assignment_queue.put_nowait(assignment)
    return frozenset(removed)


def _result_key(result: PlatformOwnedTaskResult) -> _AssignmentKey:
    return (
        result.batch_id,
        result.artifact_id,
        result.task_id,
        result.attempt_number,
    )


def _is_delivery_failure_result(result: PlatformOwnedTaskResult) -> bool:
    return result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE


__all__ = ["ArtifactAssignmentExecutor", "PlatformWorkWorker"]
