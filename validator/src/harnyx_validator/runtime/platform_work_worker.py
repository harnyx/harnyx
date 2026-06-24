"""Background worker for platform-owned miner-task assignments."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptTerminalEffect,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
)
from harnyx_validator.application.ports.platform import PlatformPort, PlatformTaskAttemptIdentity
from harnyx_validator.infrastructure.observability.sentry import capture_exception

logger = logging.getLogger("harnyx_validator.platform_work_worker")

ArtifactAssignmentExecutor = Callable[
    [
        UUID,
        asyncio.Queue[MinerTaskWorkAssignment],
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


@dataclass(slots=True)
class _ActiveArtifactGroup:
    artifact_id: UUID
    state: _ArtifactGroupState
    assignment_queue: asyncio.Queue[MinerTaskWorkAssignment]
    close_requested: asyncio.Event
    result_queue: asyncio.Queue[PlatformOwnedTaskResult]
    active_assignments: dict[_AssignmentKey, MinerTaskWorkAssignment]
    task: asyncio.Task[None]

    def identities(self) -> tuple[PlatformTaskAttemptIdentity, ...]:
        return tuple(_assignment_identity(assignment) for assignment in self.active_assignments.values())


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
    ) -> None:
        if target_concurrency < 1:
            raise ValueError("target_concurrency must be positive")
        if max_active_artifacts < 1:
            raise ValueError("max_active_artifacts must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._platform = platform
        self._execute_artifact_assignments = execute_artifact_assignments
        self._target_concurrency = target_concurrency
        self._max_active_artifacts = max_active_artifacts
        self._poll_interval_seconds = poll_interval_seconds
        self._active_artifacts: dict[_ArtifactGroupKey, _ActiveArtifactGroup] = {}
        self._pending_results: list[PlatformOwnedTaskResult] = []
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
        self._collect_closed_artifact_groups()
        if self._pending_results:
            if not await self._submit_pending_results():
                return
        free_slots = self._target_concurrency - len(self._active_attempts())
        if free_slots <= 0:
            return
        assignments = await asyncio.to_thread(
            self._platform.request_miner_task_work,
            target_concurrency=self._target_concurrency,
            max_active_artifacts=self._max_active_artifacts,
            active_attempts=self._active_attempts(),
        )
        refilled_artifact_ids = self._enqueue_assignments(assignments[:free_slots])
        self._request_idle_artifact_groups_close(refilled_artifact_ids)

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

            if len(self._active_artifacts) >= self._max_active_artifacts:
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
    ) -> _ActiveArtifactGroup:
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment] = asyncio.Queue()
        close_requested = asyncio.Event()
        result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
        group = _ActiveArtifactGroup(
            artifact_id=artifact_id,
            state=_ArtifactGroupState.OPEN,
            assignment_queue=assignment_queue,
            close_requested=close_requested,
            result_queue=result_queue,
            active_assignments={},
            task=asyncio.create_task(
                self._execute_artifact_assignments(
                    artifact_id,
                    assignment_queue,
                    close_requested,
                    result_queue,
                ),
                name=f"platform-work-artifact-{artifact_id}",
            ),
        )
        self._active_artifacts[group_key] = group
        return group

    def _enqueue_into_group(
        self,
        group: _ActiveArtifactGroup,
        assignments: Sequence[MinerTaskWorkAssignment],
    ) -> bool:
        enqueued = False
        for assignment in assignments:
            key = _assignment_key(assignment)
            if key in group.active_assignments:
                continue
            group.active_assignments[key] = assignment
            group.assignment_queue.put_nowait(assignment)
            enqueued = True
        return enqueued

    def _request_idle_artifact_groups_close(self, refilled_group_keys: set[_ArtifactGroupKey]) -> None:
        for group_key, group in tuple(self._active_artifacts.items()):
            if group.state is not _ArtifactGroupState.OPEN:
                continue
            if group.active_assignments:
                continue
            if group_key in refilled_group_keys:
                continue
            group.state = _ArtifactGroupState.CLOSING
            group.close_requested.set()

    def _request_all_artifact_groups_close(self) -> None:
        for group in self._active_artifacts.values():
            group.state = _ArtifactGroupState.CLOSING
            group.close_requested.set()

    async def _cancel_artifact_group_tasks(self) -> None:
        tasks = tuple(group.task for group in self._active_artifacts.values() if not group.task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_artifacts.clear()

    def _collect_artifact_group_results(self) -> None:
        for group in tuple(self._active_artifacts.values()):
            self._collect_results_for_group(group)

    def _collect_results_for_group(self, group: _ActiveArtifactGroup) -> None:
        while True:
            try:
                result = group.result_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending_results.append(result)
            if _is_delivery_failure_result(result):
                group.active_assignments.clear()
                group.state = _ArtifactGroupState.CLOSING
                group.close_requested.set()
                continue
            group.active_assignments.pop(_result_key(result), None)

    def _collect_closed_artifact_groups(self) -> None:
        for group_key, group in tuple(self._active_artifacts.items()):
            if not group.task.done():
                continue
            self._collect_results_for_group(group)
            try:
                group.task.result()
            except Exception as exc:
                logger.exception(
                    "platform artifact assignment executor failed",
                    extra={
                        "batch_id": str(group_key[0]),
                        "artifact_id": str(group_key[1]),
                    },
                )
                capture_exception(exc)
            if group.active_assignments:
                logger.warning(
                    "clearing platform artifact assignments after executor closed",
                    extra={
                        "batch_id": str(group_key[0]),
                        "artifact_id": str(group_key[1]),
                        "assignment_count": len(group.active_assignments),
                    },
                )
                group.active_assignments.clear()
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

        terminal = {
            (ack.batch_id, ack.artifact_id, ack.task_id, ack.attempt_number)
            for ack in acknowledgements
            if ack.outcome != "retry_later"
        }
        self._pending_results = [
            result
            for result in self._pending_results
            if (result.batch_id, result.artifact_id, result.task_id, result.attempt_number) not in terminal
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
            for identity in group.identities()
        )
        return active + pending


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


def _assignment_identity(assignment: MinerTaskWorkAssignment) -> PlatformTaskAttemptIdentity:
    return PlatformTaskAttemptIdentity(
        batch_id=assignment.batch_id,
        artifact_id=assignment.artifact.artifact_id,
        task_id=assignment.task.task_id,
        attempt_number=assignment.attempt_number,
        validator_session_id=None,
    )


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
