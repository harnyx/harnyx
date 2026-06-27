from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    ScriptArtifactSpec,
)
from harnyx_validator.application.ports.platform import (
    PlatformTaskAttemptIdentity,
    PlatformTaskResultAcknowledgement,
)
from harnyx_validator.runtime import platform_work_worker as platform_work_worker_module
from harnyx_validator.runtime.platform_work_worker import PlatformWorkWorker, _AssignedArtifactGroup, _AssignmentState

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN_PREFIX = "assignment-token"  # noqa: S105 - fixed test-only assignment token prefix


class _MonotonicClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


async def test_platform_work_worker_offloads_pending_result_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the FastAPI event loop from blocking platform result submission."""

    result = _platform_result()
    observed: dict[str, object] = {}
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    class _Platform:
        def submit_miner_task_work_results(self, results: tuple[PlatformOwnedTaskResult, ...]):
            observed["submitted_results"] = results
            return (_ack(result),)

        def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
            observed["request_work_kwargs"] = kwargs
            return ()

    async def fake_to_thread(func, /, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(platform_work_worker_module.asyncio, "to_thread", fake_to_thread)
    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._pending_results.append(result)

    await worker.run_once()

    submit_func, submit_args, submit_kwargs = to_thread_calls[0]
    assert getattr(submit_func, "__self__", None) is worker._platform
    assert submit_args == ((result,),)
    assert submit_kwargs == {}
    poll_func, poll_args, poll_kwargs = to_thread_calls[1]
    assert getattr(poll_func, "__self__", None) is worker._platform
    assert poll_args == ()
    assert poll_kwargs == {
        "target_concurrency": 1,
        "max_active_artifacts": 1,
        "active_attempts": (),
    }
    assert observed["submitted_results"] == (result,)
    assert observed["request_work_kwargs"] == {
        "target_concurrency": 1,
        "max_active_artifacts": 1,
        "active_attempts": (),
    }
    assert worker._pending_results == []


async def test_platform_work_worker_keeps_pending_result_on_retry_later() -> None:
    result = _platform_result()

    class _Platform:
        def submit_miner_task_work_results(
            self,
            _results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            return (_ack(result, outcome="retry_later"),)

        def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            raise AssertionError("worker must not request more work until pending result is accepted or rejected")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._pending_results.append(result)

    await worker.run_once()

    assert worker._pending_results == [result]


async def test_worker_groups_assignments_by_artifact_and_reports_all_active_attempts() -> None:
    batch_id = uuid4()
    artifacts = tuple(_artifact(uid=index) for index in range(1, 5))
    assignments = tuple(
        _assignment(batch_id=batch_id, artifact=artifact, task=_task(f"{artifact.uid}-{task_index}"))
        for artifact in artifacts
        for task_index in range(5)
    )

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return assignments

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def idle_executor(
        _artifact_id: UUID,
        _assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        await close_requested.wait()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=idle_executor,
        target_concurrency=20,
        max_active_artifacts=4,
    )
    await worker.run_once()

    assert set(worker._active_artifacts) == {
        (batch_id, artifact.artifact_id) for artifact in artifacts
    }
    assert len(worker._active_attempts()) == 20
    for (_batch_id, artifact_id), group in worker._active_artifacts.items():
        assert group.local_inflight_count() == 5
        assert all(
            record.assignment.artifact.artifact_id == artifact_id
            for record in group.assignment_records.values()
        )

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_does_not_release_reservations_while_artifact_startup_is_in_flight() -> None:
    clock = _MonotonicClock()
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("never-started"))
    request_active_attempts: list[tuple[PlatformTaskAttemptIdentity, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(
            self,
            *,
            active_attempts: tuple[PlatformTaskAttemptIdentity, ...],
            **_kwargs: object,
        ) -> tuple[MinerTaskWorkAssignment, ...]:
            request_active_attempts.append(active_attempts)
            self.request_count += 1
            return (assignment,) if self.request_count == 1 else ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    startup_assignment_seen = asyncio.Event()

    async def slow_startup_executor(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        await assigned_work.take_for_startup()
        startup_assignment_seen.set()
        await close_requested.wait()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=slow_startup_executor,
        target_concurrency=1,
        max_active_artifacts=1,
        dispatch_start_lease_seconds=10.0,
        monotonic_clock=clock,
    )

    await worker.run_once()
    await asyncio.wait_for(startup_assignment_seen.wait(), timeout=1.0)
    group = next(iter(worker._active_artifacts.values()))
    record = next(iter(group.assignment_records.values()))
    assert record.state is _AssignmentState.STARTUP_RESERVED
    assert record.dispatchable_at is None
    assert worker._local_inflight_count() == 1

    clock.advance(11.0)
    await worker.run_once()

    assert group.task is not None
    assert not group.task.done()
    assert worker._local_inflight_count() == 1
    assert request_active_attempts[-1] == ()
    assert worker._active_attempts() == (
        PlatformTaskAttemptIdentity(
            batch_id=assignment.batch_id,
            artifact_id=assignment.artifact.artifact_id,
            task_id=assignment.task.task_id,
            attempt_number=assignment.attempt_number,
            validator_session_id=None,
        ),
    )

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_claimed_assignment_counts_capacity_but_is_not_reportable_until_started() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("claimed"))
    group = _assigned_group(artifact_id=artifact.artifact_id)
    session_id = uuid4()

    assert group.put_nowait(assignment)
    group.mark_dispatch_ready()
    claimed = group.claim_initial_for_dispatch(assignment)

    assert claimed is not None
    assert group.local_inflight_count() == 1
    assert group.reportable_identities() == ()

    claimed.mark_started(session_id)

    assert group.local_inflight_count() == 1
    assert group.reportable_identities() == (
        PlatformTaskAttemptIdentity(
            batch_id=assignment.batch_id,
            artifact_id=assignment.artifact.artifact_id,
            task_id=assignment.task.task_id,
            attempt_number=assignment.attempt_number,
            validator_session_id=session_id,
        ),
    )


async def test_claimed_assignment_failure_before_start_enqueues_result_and_clears_capacity() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("start-failed"))
    group = _assigned_group(artifact_id=artifact.artifact_id)
    result = _platform_result(
        assignment=assignment,
        terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
    )

    assert group.put_nowait(assignment)
    group.mark_dispatch_ready()
    claimed = group.claim_initial_for_dispatch(assignment)
    assert claimed is not None

    claimed.fail_before_start(result)
    claimed.fail_before_start(result)

    assert group.local_inflight_count() == 0
    assert group.reportable_identities() == ()
    assert group.result_queue.get_nowait() is result
    assert group.result_queue.empty()


async def test_mark_started_does_not_resurrect_released_dispatchable_assignment() -> None:
    clock = _MonotonicClock()
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("released"))
    dispatch_ready = asyncio.Event()

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            return (assignment,) if self.request_count == 1 else ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def queued_executor(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        dispatch_ready.set()
        await close_requested.wait()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=queued_executor,
        target_concurrency=1,
        max_active_artifacts=1,
        dispatch_start_lease_seconds=10.0,
        monotonic_clock=clock,
    )

    await worker.run_once()
    await asyncio.wait_for(dispatch_ready.wait(), timeout=1.0)
    group = next(iter(worker._active_artifacts.values()))

    clock.advance(11.0)
    await worker.run_once()

    assert group.claim_initial_for_dispatch(assignment) is None
    assert group.assignment_records == {}

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_counts_startup_reservations_against_capacity_without_time_expiry() -> None:
    clock = _MonotonicClock()
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("capacity"))

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            return (assignment,) if self.request_count == 1 else ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def never_starting_executor(
        _artifact_id: UUID,
        _assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        await close_requested.wait()

    platform = _Platform()
    worker = PlatformWorkWorker(
        platform=platform,  # type: ignore[arg-type]
        execute_artifact_assignments=never_starting_executor,
        target_concurrency=1,
        max_active_artifacts=1,
        dispatch_start_lease_seconds=10.0,
        monotonic_clock=clock,
    )

    await worker.run_once()
    await worker.run_once()
    assert platform.request_count == 1

    clock.advance(11.0)
    await worker.run_once()

    assert platform.request_count == 1

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_does_not_release_started_assignment_when_pre_start_lease_expires() -> None:
    clock = _MonotonicClock()
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("started"))
    session_id = uuid4()
    request_active_attempts: list[tuple[PlatformTaskAttemptIdentity, ...]] = []
    started = asyncio.Event()

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(
            self,
            *,
            active_attempts: tuple[PlatformTaskAttemptIdentity, ...],
            **_kwargs: object,
        ) -> tuple[MinerTaskWorkAssignment, ...]:
            request_active_attempts.append(active_attempts)
            self.request_count += 1
            return (assignment,) if self.request_count == 1 else ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def starting_executor(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        queued = await assigned_work.claim_for_dispatch()
        queued.mark_started(session_id)
        started.set()
        await close_requested.wait()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=starting_executor,
        target_concurrency=2,
        max_active_artifacts=1,
        dispatch_start_lease_seconds=10.0,
        monotonic_clock=clock,
    )

    await worker.run_once()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    clock.advance(11.0)
    await worker.run_once()

    group = next(iter(worker._active_artifacts.values()))
    assert next(iter(group.assignment_records.values())).state is _AssignmentState.STARTED
    assert group.task is not None
    assert not group.task.done()
    assert request_active_attempts[-1] == (
        PlatformTaskAttemptIdentity(
            batch_id=assignment.batch_id,
            artifact_id=assignment.artifact.artifact_id,
            task_id=assignment.task.task_id,
            attempt_number=assignment.attempt_number,
            validator_session_id=session_id,
        ),
    )

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_expires_queued_reservation_inside_started_group_without_dropping_reassignment() -> None:
    clock = _MonotonicClock()
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    started_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("started"))
    queued_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("queued"))
    session_id = uuid4()
    request_active_attempts: list[tuple[PlatformTaskAttemptIdentity, ...]] = []
    started = asyncio.Event()

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(
            self,
            *,
            active_attempts: tuple[PlatformTaskAttemptIdentity, ...],
            **_kwargs: object,
        ) -> tuple[MinerTaskWorkAssignment, ...]:
            request_active_attempts.append(active_attempts)
            self.request_count += 1
            if self.request_count == 1:
                return (started_assignment, queued_assignment)
            if self.request_count == 2:
                return (queued_assignment,)
            return ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def starts_only_first_assignment(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        claimed = await assigned_work.claim_for_dispatch()
        assert claimed.assignment.task.task_id == started_assignment.task.task_id
        claimed.mark_started(session_id)
        started.set()
        await close_requested.wait()

    platform = _Platform()
    worker = PlatformWorkWorker(
        platform=platform,  # type: ignore[arg-type]
        execute_artifact_assignments=starts_only_first_assignment,
        target_concurrency=2,
        max_active_artifacts=1,
        dispatch_start_lease_seconds=10.0,
        monotonic_clock=clock,
    )

    await worker.run_once()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    group = next(iter(worker._active_artifacts.values()))
    assert group.assignment_queue.qsize() == 1
    assert group.local_inflight_count() == 2

    clock.advance(11.0)
    await worker.run_once()

    assert platform.request_count == 2
    assert request_active_attempts[-1] == (
        PlatformTaskAttemptIdentity(
            batch_id=started_assignment.batch_id,
            artifact_id=started_assignment.artifact.artifact_id,
            task_id=started_assignment.task.task_id,
            attempt_number=started_assignment.attempt_number,
            validator_session_id=session_id,
        ),
    )
    assert group.local_inflight_count() == 2
    assert {
        record.assignment.task.task_id
        for record in group.assignment_records.values()
        if record.state is _AssignmentState.DISPATCHABLE_QUEUED
    } == {queued_assignment.task.task_id}
    assert group.assignment_queue.qsize() == 1

    await worker.run_once()

    assert group.local_inflight_count() == 2
    assert {
        record.assignment.task.task_id
        for record in group.assignment_records.values()
        if record.state is _AssignmentState.DISPATCHABLE_QUEUED
    } == {queued_assignment.task.task_id}
    assert group.assignment_queue.qsize() == 1

    clock.advance(11.0)
    await worker.run_once()

    assert group.local_inflight_count() == 1
    assert {
        record.assignment.task.task_id
        for record in group.assignment_records.values()
    } == {started_assignment.task.task_id}
    assert group.assignment_queue.empty()

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_submits_finished_task_results_without_waiting_for_artifact_group_to_close() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    first = _assignment(batch_id=batch_id, artifact=artifact, task=_task("first"))
    second = _assignment(batch_id=batch_id, artifact=artifact, task=_task("second"))
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []
    result_ready = asyncio.Event()

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            return (first, second) if self.request_count == 1 else ()

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def partial_executor(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        claimed = await assigned_work.claim_for_dispatch()
        claimed.mark_started(uuid4())
        await result_queue.put(_platform_result(assignment=claimed.assignment))
        result_ready.set()
        await close_requested.wait()

    platform = _Platform()
    worker = PlatformWorkWorker(
        platform=platform,  # type: ignore[arg-type]
        execute_artifact_assignments=partial_executor,
        target_concurrency=2,
        max_active_artifacts=1,
    )
    await worker.run_once()
    await asyncio.wait_for(result_ready.wait(), timeout=1)
    await worker.run_once()

    assert len(submitted) == 1
    assert submitted[0][0].task_id == first.task.task_id
    assert worker._active_attempts() == (
        PlatformTaskAttemptIdentity(
            batch_id=second.batch_id,
            artifact_id=second.artifact.artifact_id,
            task_id=second.task.task_id,
            attempt_number=second.attempt_number,
            validator_session_id=None,
        ),
    )

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_keeps_same_artifact_assignments_separate_across_batches() -> None:
    first_batch_id = uuid4()
    second_batch_id = uuid4()
    artifact = _artifact(uid=1)
    first = _assignment(batch_id=first_batch_id, artifact=artifact, task=_task("first"))
    second = _assignment(batch_id=second_batch_id, artifact=artifact, task=_task("second"))
    started_batches: list[UUID] = []

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            return (first,) if self.request_count == 1 else (second,)

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def idle_executor(
        _artifact_id: UUID,
        assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        claimed = await assigned_work.claim_for_dispatch()
        claimed.mark_started(uuid4())
        started_batches.append(claimed.assignment.batch_id)
        await close_requested.wait()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=idle_executor,
        target_concurrency=2,
        max_active_artifacts=2,
    )
    await worker.run_once()
    await asyncio.sleep(0)
    await worker.run_once()
    await asyncio.sleep(0)

    assert started_batches == [first_batch_id, second_batch_id]
    assert set(worker._active_artifacts) == {
        (first_batch_id, artifact.artifact_id),
        (second_batch_id, artifact.artifact_id),
    }
    assert {
        (attempt.batch_id, attempt.artifact_id, attempt.task_id)
        for attempt in worker._active_attempts()
    } == {
        (first_batch_id, artifact.artifact_id, first.task.task_id),
        (second_batch_id, artifact.artifact_id, second.task.task_id),
    }

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_collects_result_from_group_that_finishes_before_cleanup() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("only"))
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return (assignment,)

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def finishing_executor(
        _artifact_id: UUID,
        assigned_work,
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        queued = await assigned_work.claim_for_dispatch()
        queued.mark_started(uuid4())
        await result_queue.put(_platform_result(assignment=queued.assignment))

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=finishing_executor,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    await worker.run_once()
    task = next(iter(worker._active_artifacts.values())).task
    assert task is not None
    await asyncio.wait_for(task, timeout=1)
    await worker.run_once()

    assert submitted
    assert submitted[0][0].task_id == assignment.task.task_id
    assert worker._active_artifacts == {}
    assert worker._pending_results == []


async def test_worker_delivery_failure_clears_group_identities_and_frees_capacity() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignments = tuple(
        _assignment(batch_id=batch_id, artifact=artifact, task=_task(f"task-{index}"))
        for index in range(3)
    )
    emitted = asyncio.Event()
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return assignments

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def delivery_failure_executor(
        _artifact_id: UUID,
        assigned_work,
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        delivery_failure_claim = await assigned_work.claim_for_dispatch()
        delivery_failure_claim.mark_started(uuid4())
        delivery_failure_assignment = delivery_failure_claim.assignment
        completed_claim = await assigned_work.claim_for_dispatch()
        completed_claim.mark_started(uuid4())
        completed_assignment = completed_claim.assignment
        ignored_claim = await assigned_work.claim_for_dispatch()
        await result_queue.put(
            _platform_result(
                assignment=completed_assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
            )
        )
        await result_queue.put(
            _platform_result(
                assignment=delivery_failure_assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            )
        )
        assert ignored_claim.assignment is assignments[2]
        emitted.set()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=delivery_failure_executor,
        target_concurrency=3,
        max_active_artifacts=1,
    )
    await worker.run_once()
    assert len(worker._active_attempts()) == 3

    await asyncio.wait_for(emitted.wait(), timeout=1)
    await worker.run_once()

    assert len(submitted) == 1
    assert {result.task_id for result in submitted[0]} == {
        assignments[0].task.task_id,
        assignments[1].task.task_id,
    }
    assert worker._active_attempts() == ()
    assert worker._active_artifacts == {}


async def test_worker_collects_all_results_before_clearing_closed_artifact_group() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignments = tuple(
        _assignment(batch_id=batch_id, artifact=artifact, task=_task(f"task-{index}"))
        for index in range(2)
    )
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return assignments

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def finishing_executor(
        _artifact_id: UUID,
        assigned_work,
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        while True:
            try:
                claimed = assigned_work.claim_nowait_for_dispatch()
            except asyncio.QueueEmpty:
                break
            # Results represent completed task execution, so the claim must have crossed
            # the validator-session start boundary first.
            claimed.mark_started(uuid4())
            assignment = claimed.assignment
            await result_queue.put(_platform_result(assignment=assignment))

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=finishing_executor,
        target_concurrency=2,
        max_active_artifacts=1,
    )

    await worker.run_once()
    task = next(iter(worker._active_artifacts.values())).task
    assert task is not None
    await asyncio.wait_for(task, timeout=1)
    await worker.run_once()

    assert len(submitted) == 1
    assert {result.task_id for result in submitted[0]} == {assignment.task.task_id for assignment in assignments}
    assert all(
        result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
        for result in submitted[0]
    )
    assert worker._active_artifacts == {}
    assert worker._pending_results == []


async def _unexpected_execute_artifact_assignments(*args: object, **kwargs: object) -> None:
    raise AssertionError("worker must not execute assignments in this test")


def _task(text: str) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=0.05,
    )


def _artifact(*, uid: int) -> ScriptArtifactSpec:
    return ScriptArtifactSpec(
        uid=uid,
        artifact_id=uuid4(),
        content_hash=f"hash-{uid}",
        size_bytes=1,
        miner_hotkey_ss58=f"miner-hotkey-{uid}",
    )


def _assignment(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    attempt_number: int = 1,
    max_attempts: int = 2,
) -> MinerTaskWorkAssignment:
    return MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        assignment_token=f"{_ASSIGNMENT_TOKEN_PREFIX}-{attempt_number}",
    )


def _assigned_group(*, artifact_id: UUID) -> _AssignedArtifactGroup:
    return _AssignedArtifactGroup(
        artifact_id=artifact_id,
        state=platform_work_worker_module._ArtifactGroupState.OPEN,
        assignment_queue=asyncio.Queue(),
        close_requested=asyncio.Event(),
        result_queue=asyncio.Queue(),
        assignment_records={},
        starting_records={},
        monotonic_clock=_MonotonicClock(),
    )


def _platform_result(
    assignment: MinerTaskWorkAssignment | None = None,
    *,
    terminal_effect: MinerTaskAttemptTerminalEffect = MinerTaskAttemptTerminalEffect.TASK_RESULT,
) -> PlatformOwnedTaskResult:
    batch_id = assignment.batch_id if assignment is not None else uuid4()
    artifact_id = assignment.artifact.artifact_id if assignment is not None else uuid4()
    task_id = assignment.task.task_id if assignment is not None else uuid4()
    attempt_number = assignment.attempt_number if assignment is not None else 1
    uid = assignment.artifact.uid if assignment is not None else 7
    session_id = uuid4()
    now = datetime.now(UTC)
    return PlatformOwnedTaskResult(
        batch_id=batch_id,
        artifact_id=artifact_id,
        task_id=task_id,
        attempt_number=attempt_number,
        result=None,
        terminal_attempt=MinerTaskAttemptAuditRecord(
            validator_session_id=session_id,
            batch_id=batch_id,
            artifact_id=artifact_id,
            task_id=task_id,
            attempt_number=attempt_number,
            uid=uid,
            miner_hotkey_ss58="miner-hotkey",
            started_at=now,
            finished_at=now,
            status=MinerTaskAttemptStatus.FAILED
            if terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
            else MinerTaskAttemptStatus.SUCCEEDED,
            error_code="artifact_setup_failed"
            if terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
            else None,
            error_summary_code="artifact_setup_failed"
            if terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
            else None,
            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
            terminal_effect=terminal_effect,
            max_attempts=2,
            execution_log=(),
        ),
    )


def _ack(
    result: PlatformOwnedTaskResult,
    *,
    outcome: str = "accepted",
) -> PlatformTaskResultAcknowledgement:
    return PlatformTaskResultAcknowledgement(
        batch_id=result.batch_id,
        artifact_id=result.artifact_id,
        task_id=result.task_id,
        attempt_number=result.attempt_number,
        outcome=outcome,
        canonical=True,
    )
