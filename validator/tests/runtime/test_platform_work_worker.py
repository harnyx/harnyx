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
from harnyx_validator.runtime.platform_work_worker import PlatformWorkWorker

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN_PREFIX = "assignment-token"  # noqa: S105 - fixed test-only assignment token prefix


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
        _assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
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
        assert len(group.active_assignments) == 5
        assert all(assignment.artifact.artifact_id == artifact_id for assignment in group.active_assignments.values())

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
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assignment = await assignment_queue.get()
        await result_queue.put(_platform_result(assignment=assignment))
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
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assignment = await assignment_queue.get()
        started_batches.append(assignment.batch_id)
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
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        queued = await assignment_queue.get()
        await result_queue.put(_platform_result(assignment=queued))

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=finishing_executor,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    await worker.run_once()
    await asyncio.wait_for(next(iter(worker._active_artifacts.values())).task, timeout=1)
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
            return tuple(_ack(result) for result in results)

    async def delivery_failure_executor(
        _artifact_id: UUID,
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assignment = await assignment_queue.get()
        await result_queue.put(
            _platform_result(
                assignment=assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            )
        )
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

    assert worker._active_attempts() == ()
    assert worker._active_artifacts == {}


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
