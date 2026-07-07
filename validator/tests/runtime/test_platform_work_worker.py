from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from threading import Event as ThreadEvent
from uuid import UUID, uuid4

import pytest

from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunSubmission,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskExecution,
    PlatformOwnedTaskResult,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.ports.platform import (
    PlatformTaskAttemptIdentity,
    PlatformTaskResultAcknowledgement,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.runtime import platform_work_worker as platform_work_worker_module
from harnyx_validator.runtime.platform_work_worker import (
    PlatformWorkWorker,
    _AssignedArtifactGroup,
    _AssignmentState,
)

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN_PREFIX = "assignment-token"  # noqa: S105 - fixed test-only assignment token prefix


class _MonotonicClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


async def _run_once_and_consume_platform_work(worker: PlatformWorkWorker) -> None:
    await worker.run_once()
    await _wait_for_work_request_done(worker)
    await worker.run_once()


async def _wait_for_work_request_done(worker: PlatformWorkWorker) -> None:
    for _ in range(100):
        task = worker._work_request_task
        if task is None or task.done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("platform work request did not finish")


async def _wait_for_scoreable_request_done(worker: PlatformWorkWorker) -> None:
    for _ in range(100):
        task = worker._scoreable_execution_request_task
        if task is None or task.done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("platform scoreable execution request did not finish")


async def test_platform_work_worker_offloads_pending_result_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect the FastAPI event loop from blocking platform result submission."""

    result = _platform_result()
    observed: dict[str, object] = {}
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    class _Platform:
        def submit_miner_task_work_results(self, results: tuple[PlatformOwnedTaskResult, ...]):
            observed["submitted_results"] = results
            return (_ack(result),)

        async def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
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
    await _wait_for_work_request_done(worker)

    submit_func, submit_args, submit_kwargs = to_thread_calls[0]
    assert getattr(submit_func, "__self__", None) is worker._platform
    assert submit_args == ((result,),)
    assert submit_kwargs == {}
    assert len(to_thread_calls) == 1
    assert observed["submitted_results"] == (result,)
    assert observed["request_work_kwargs"] == {
        "target_concurrency": 1,
        "max_active_artifacts": 1,
        "active_attempts": (),
    }
    assert worker._pending_results == []


async def test_platform_work_worker_retains_pending_result_when_report_transport_fails() -> None:
    """Prevent a lost HTTP response from dropping a terminal task result."""

    result = _platform_result()
    work_requests = 0
    scoreable_requests = 0

    class _Platform:
        def submit_miner_task_work_results(self, _results: tuple[PlatformOwnedTaskResult, ...]):
            raise RuntimeError("platform result report failed before acknowledgement")

        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            _ = limit, active_scoring
            nonlocal scoreable_requests
            scoreable_requests += 1
            return ()

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            nonlocal work_requests
            work_requests += 1
            return ()

    async def unexpected_score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        raise AssertionError("worker must not start scoring after result submission fails")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=unexpected_score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._pending_results.append(result)

    await worker.run_once()

    assert worker._pending_results == [result]
    assert worker._work_request_task is None
    assert work_requests == 0
    assert scoreable_requests == 0


async def test_platform_work_worker_scoreable_poll_does_not_block_work_poll() -> None:
    """Slow persisted-execution polling must not block miner task query refill."""

    scoreable_started = ThreadEvent()
    release_scoreable = ThreadEvent()
    work_requests: list[dict[str, object]] = []

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            assert limit == 20
            assert active_scoring == ()
            scoreable_started.set()
            release_scoreable.wait()
            return ()

        async def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
            work_requests.append(kwargs)
            return ()

    async def unexpected_score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        raise AssertionError("no scoreable execution should be returned")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=unexpected_score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    assert await asyncio.to_thread(scoreable_started.wait, 1.0)
    await _wait_for_work_request_done(worker)

    assert work_requests
    assert all(
        request
        == {
            "target_concurrency": 1,
            "max_active_artifacts": 1,
            "active_attempts": (),
        }
        for request in work_requests
    )

    await worker._cancel_scoreable_execution_request_task()
    release_scoreable.set()


async def test_platform_work_worker_scores_persisted_executions_without_consuming_execution_slots() -> None:
    """Scoring persisted execution evidence must not reduce miner task query capacity."""

    batch_id = uuid4()
    artifact = _artifact(uid=7)
    task = _task("scoreable")
    execution = _platform_execution(batch_id=batch_id, artifact=artifact, task=task)
    final_result = _platform_result(
        _assignment(batch_id=batch_id, artifact=artifact, task=task),
        terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
    )
    score_started = asyncio.Event()
    release_score = asyncio.Event()
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []
    work_requests: list[dict[str, object]] = []
    scoreable_requests: list[tuple[int, tuple[PlatformTaskAttemptIdentity, ...]]] = []

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            scoreable_requests.append((limit, active_scoring))
            return (execution,) if not active_scoring else ()

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(item) for item in results)

        async def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
            work_requests.append(kwargs)
            return ()

    async def score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        assert _execution is execution
        score_started.set()
        await release_score.wait()
        return final_result

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    await _wait_for_scoreable_request_done(worker)
    await worker.run_once()
    await asyncio.wait_for(score_started.wait(), timeout=1.0)
    await _wait_for_work_request_done(worker)

    scoring_identity = PlatformTaskAttemptIdentity(
        batch_id=execution.batch_id,
        artifact_id=execution.artifact.artifact_id,
        task_id=execution.task_id,
        attempt_number=execution.attempt_number,
        validator_session_id=execution.validator_session_id,
    )
    assert worker._local_inflight_count() == 0
    assert scoreable_requests[0] == (20, ())
    assert all(request == (19, (scoring_identity,)) for request in scoreable_requests[1:])
    assert work_requests
    assert all(
        request
        == {
            "target_concurrency": 1,
            "max_active_artifacts": 1,
            "active_attempts": (),
        }
        for request in work_requests
    )

    release_score.set()
    await asyncio.sleep(0)
    await worker.run_once()

    assert submitted == [(final_result,)]
    assert worker._pending_results == []


async def test_platform_work_worker_suppresses_stale_scoreable_response_with_pending_result() -> None:
    """Do not rescore an execution whose final result is already awaiting delivery."""

    batch_id = uuid4()
    artifact = _artifact(uid=9)
    task = _task("stale-scoreable")
    execution = _platform_execution(batch_id=batch_id, artifact=artifact, task=task)
    result = _platform_result(
        _assignment(batch_id=batch_id, artifact=artifact, task=task),
        terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
        successful=True,
        validator_session_id=execution.validator_session_id,
    )
    score_calls = 0

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            assert limit == 20
            assert active_scoring == ()
            return (execution,)

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        nonlocal score_calls
        score_calls += 1
        return result

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    await _wait_for_scoreable_request_done(worker)
    worker._pending_results.append(result)
    worker._consume_completed_scoreable_execution_request()

    assert score_calls == 0
    assert worker._active_scoring == {}


async def test_platform_work_worker_requests_only_remaining_scoring_slots() -> None:
    observed_limits: list[int] = []
    observed_active_scoring: list[tuple[PlatformTaskAttemptIdentity, ...]] = []

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            observed_limits.append(limit)
            observed_active_scoring.append(active_scoring)
            return ()

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def never_complete() -> PlatformOwnedTaskResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=lambda _execution: never_complete(),
        target_concurrency=1,
        max_active_artifacts=1,
        scoring_limit=20,
    )
    for _ in range(18):
        task = asyncio.create_task(never_complete())
        worker._active_scoring[task] = PlatformTaskAttemptIdentity(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            attempt_number=1,
            validator_session_id=uuid4(),
        )
    worker._pending_results.append(_platform_result(successful=True))

    worker._start_scoreable_execution_request()
    await _wait_for_scoreable_request_done(worker)

    assert observed_limits == [1]
    assert len(observed_active_scoring[0]) == 19
    await worker._cancel_scoring_tasks()


async def test_platform_work_worker_does_not_poll_scoreable_executions_when_scoring_full() -> None:
    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            _ = limit, active_scoring
            raise AssertionError("worker must not request scoreable executions when scoring is full")

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def never_complete() -> PlatformOwnedTaskResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=lambda _execution: never_complete(),
        target_concurrency=1,
        max_active_artifacts=1,
        scoring_limit=2,
    )
    for _ in range(2):
        task = asyncio.create_task(never_complete())
        worker._active_scoring[task] = PlatformTaskAttemptIdentity(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            attempt_number=1,
            validator_session_id=uuid4(),
        )

    await worker.run_once()

    assert worker._scoreable_execution_request_task is None
    await worker._cancel_scoring_tasks()


async def test_platform_work_worker_does_not_start_more_scoring_than_remaining_slots_from_large_response() -> None:
    batch_id = uuid4()
    executions = tuple(
        _platform_execution(batch_id=batch_id, artifact=_artifact(uid=index), task=_task(f"over-return-{index}"))
        for index in range(1, 6)
    )
    score_started: list[UUID] = []
    release_score = asyncio.Event()

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            assert limit == 2
            assert active_scoring == ()
            return executions

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def score_execution(execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        score_started.append(execution.task_id)
        await release_score.wait()
        return _platform_result(
            _assignment(batch_id=execution.batch_id, artifact=execution.artifact, task=execution.task),
            successful=True,
            validator_session_id=execution.validator_session_id,
        )

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
        scoring_limit=2,
    )

    await worker.run_once()
    await _wait_for_scoreable_request_done(worker)
    await worker.run_once()
    await asyncio.sleep(0)

    assert len(score_started) == 2
    assert len(worker._active_scoring) == 2
    release_score.set()
    await worker._cancel_scoring_tasks()


async def test_platform_work_worker_submits_results_while_work_poll_is_pending() -> None:
    """Prevent slow assignment polling from blocking terminal result delivery."""

    result = _platform_result()
    poll_started = asyncio.Event()
    release_poll = asyncio.Event()
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            poll_started.set()
            await release_poll.wait()
            return ()

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(item) for item in results)

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    await asyncio.wait_for(poll_started.wait(), timeout=1.0)

    worker._pending_results.append(result)
    await worker.run_once()

    assert submitted == [(result,)]
    assert worker._pending_results == []

    release_poll.set()
    await worker._cancel_work_request_task()


async def test_platform_work_worker_cancels_pending_work_poll() -> None:
    """Prevent worker shutdown from waiting on a long Platform work-poll read."""

    poll_started = asyncio.Event()
    poll_cancelled = asyncio.Event()

    class _Platform:
        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            poll_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                poll_cancelled.set()
                raise

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    await asyncio.wait_for(poll_started.wait(), timeout=1.0)

    await worker._cancel_work_request_task()

    assert poll_cancelled.is_set()
    assert worker._work_request_task is None


async def test_platform_work_worker_cancels_pending_scoreable_poll() -> None:
    """Prevent worker shutdown from retaining a long scoreable-execution poll task."""

    poll_started = ThreadEvent()
    release_poll = ThreadEvent()

    class _Platform:
        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            _ = limit, active_scoring
            poll_started.set()
            release_poll.wait()
            return ()

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def unexpected_score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        raise AssertionError("no scoreable execution should be returned")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=unexpected_score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    await worker.run_once()
    assert await asyncio.to_thread(poll_started.wait, 1.0)

    await worker._cancel_scoreable_execution_request_task()
    release_poll.set()

    assert worker._scoreable_execution_request_task is None


async def test_platform_work_worker_cancels_active_scoring_on_stop() -> None:
    """Stop must not leave persisted-execution scoring tasks running in the background."""

    batch_id = uuid4()
    artifact = _artifact(uid=8)
    task = _task("long-running-score")
    execution = _platform_execution(batch_id=batch_id, artifact=artifact, task=task)
    scoring_started = asyncio.Event()
    scoring_cancelled = asyncio.Event()

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        def request_scoreable_miner_task_work_executions(
            self,
            *,
            limit: int,
            active_scoring: tuple[PlatformTaskAttemptIdentity, ...],
        ) -> tuple[PlatformOwnedTaskExecution, ...]:
            if active_scoring:
                assert limit == 19
                return ()
            assert limit == 20
            if self.requested:
                return ()
            self.requested = True
            return (execution,)

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    async def score_execution(_execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        scoring_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            scoring_cancelled.set()
        raise AssertionError("cancelled scoring task should not return a result")

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        score_execution=score_execution,
        target_concurrency=1,
        max_active_artifacts=1,
        poll_interval_seconds=0.01,
    )

    worker.start()
    await asyncio.wait_for(scoring_started.wait(), timeout=1.0)
    await worker.stop()

    assert scoring_cancelled.is_set()
    assert worker._active_scoring == {}


async def test_platform_work_worker_captures_active_attempts_inside_work_request_coroutine() -> None:
    """Prevent scheduling a work-poll task with a stale active-attempt snapshot."""

    result = _platform_result()
    captured_active_attempts: list[tuple[PlatformTaskAttemptIdentity, ...]] = []

    class _Platform:
        async def request_miner_task_work(
            self,
            *,
            active_attempts: tuple[PlatformTaskAttemptIdentity, ...],
            **_kwargs: object,
        ) -> tuple[object, ...]:
            captured_active_attempts.append(active_attempts)
            return ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )

    worker._start_work_request()
    worker._pending_results.append(result)
    await _wait_for_work_request_done(worker)

    assert captured_active_attempts == [
        (
            PlatformTaskAttemptIdentity(
                batch_id=result.batch_id,
                artifact_id=result.artifact_id,
                task_id=result.task_id,
                attempt_number=result.attempt_number,
                validator_session_id=result.terminal_attempt.validator_session_id,
            ),
        )
    ]


async def test_platform_work_worker_removes_acknowledged_rejected_result(caplog: pytest.LogCaptureFixture) -> None:
    """Prevent an explicitly rejected result from blocking future work forever."""

    result = _platform_result()
    observed: dict[str, object] = {}

    class _Platform:
        def submit_miner_task_work_results(
            self,
            _results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            return (
                _ack(
                    result,
                    outcome="rejected",
                    reason_code="conflicting_replay",
                    reason="terminal result conflicts with accepted attempt",
                ),
            )

        async def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
            observed["request_work_kwargs"] = kwargs
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._pending_results.append(result)

    caplog.set_level(logging.WARNING, logger="harnyx_validator.platform_work_worker")
    await worker.run_once()
    await _wait_for_work_request_done(worker)

    assert worker._pending_results == []
    assert "platform rejected miner task result" in caplog.text
    assert any(record.__dict__.get("reason_code") == "conflicting_replay" for record in caplog.records)
    assert observed["request_work_kwargs"] == {
        "target_concurrency": 1,
        "max_active_artifacts": 1,
        "active_attempts": (),
    }


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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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
    await _run_once_and_consume_platform_work(worker)

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

        async def request_miner_task_work(
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

    await _run_once_and_consume_platform_work(worker)
    await asyncio.wait_for(startup_assignment_seen.wait(), timeout=1.0)
    group = next(iter(worker._active_artifacts.values()))
    record = next(iter(group.assignment_records.values()))
    assert record.state is _AssignmentState.STARTUP_RESERVED
    assert record.dispatchable_at is None
    assert worker._local_inflight_count() == 1

    clock.advance(11.0)
    await _run_once_and_consume_platform_work(worker)

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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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

    await _run_once_and_consume_platform_work(worker)
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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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

    await _run_once_and_consume_platform_work(worker)
    await worker.run_once()
    assert platform.request_count == 1

    clock.advance(11.0)
    await _run_once_and_consume_platform_work(worker)

    assert platform.request_count == 1

    worker._request_all_artifact_groups_close()
    await worker._cancel_artifact_group_tasks()


async def test_worker_does_not_poll_platform_when_closing_group_still_consumes_only_artifact_slot() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("closing"))
    group = _assigned_group(artifact_id=artifact.artifact_id)
    group.state = platform_work_worker_module._ArtifactGroupState.CLOSING
    group.put_nowait(assignment)

    class _Platform:
        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            raise AssertionError("worker must not request work while an artifact group is closing")

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._active_artifacts[(batch_id, artifact.artifact_id)] = group

    await worker.run_once()

    assert group.state is platform_work_worker_module._ArtifactGroupState.CLOSING


async def test_worker_polls_platform_when_idle_closing_group_does_not_own_capacity() -> None:
    """Prevent an idle cleanup task from blocking unrelated platform-owned assignments."""

    old_batch_id = uuid4()
    old_artifact = _artifact(uid=1)
    old_group = _assigned_group(artifact_id=old_artifact.artifact_id)
    old_group.state = platform_work_worker_module._ArtifactGroupState.CLOSING

    new_batch_id = uuid4()
    new_artifact = _artifact(uid=2)
    new_assignment = _assignment(batch_id=new_batch_id, artifact=new_artifact, task=_task("new"))

    class _Platform:
        request_count = 0

        async def request_miner_task_work(self, **kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            assert kwargs["active_attempts"] == ()
            return (new_assignment,)

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    platform = _Platform()
    worker = PlatformWorkWorker(
        platform=platform,  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._active_artifacts[(old_batch_id, old_artifact.artifact_id)] = old_group

    await _run_once_and_consume_platform_work(worker)

    assert platform.request_count == 1
    assert (new_batch_id, new_artifact.artifact_id) in worker._active_artifacts
    assert worker._active_artifacts[(new_batch_id, new_artifact.artifact_id)].local_inflight_count() == 1


async def test_worker_can_poll_after_result_ack_releases_idle_group_capacity() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("finished"))
    result = _platform_result(assignment=assignment)
    group = _assigned_group(artifact_id=artifact.artifact_id)
    assert group.put_nowait(assignment)
    group.result_queue.put_nowait(result)
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []
    requests: list[dict[str, object]] = []

    class _Platform:
        async def request_miner_task_work(self, **kwargs: object) -> tuple[object, ...]:
            requests.append(kwargs)
            return ()

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(item) for item in results)

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    worker._active_artifacts[(batch_id, artifact.artifact_id)] = group

    await worker.run_once()
    await _wait_for_work_request_done(worker)

    assert submitted == [(result,)]
    assert requests == [
        {
            "target_concurrency": 1,
            "max_active_artifacts": 1,
            "active_attempts": (),
        }
    ]
    assert group.state is platform_work_worker_module._ArtifactGroupState.CLOSING
    assert group.close_requested.is_set()
    assert group.local_inflight_count() == 0


async def test_worker_does_not_poll_platform_while_claimed_assignment_is_unreportable() -> None:
    batch_id = uuid4()
    artifact = _artifact(uid=1)
    assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("claimed"))
    group = _assigned_group(artifact_id=artifact.artifact_id)
    assert group.put_nowait(assignment)
    group.mark_dispatch_ready()
    claimed = group.claim_initial_for_dispatch(assignment)
    assert claimed is not None
    assert group.local_inflight_count() == 1
    assert group.reportable_identities() == ()

    class _Platform:
        async def request_miner_task_work(self, **_kwargs: object) -> tuple[object, ...]:
            raise AssertionError("worker must not request work while claimed assignments are unreportable")

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=2,
        max_active_artifacts=1,
    )
    worker._active_artifacts[(batch_id, artifact.artifact_id)] = group

    await worker.run_once()

    assert group.starting_records
    assert worker._local_inflight_count() == 1


async def test_worker_can_poll_when_full_artifact_capacity_is_open_and_reportable() -> None:
    batch_id = uuid4()
    artifacts = (_artifact(uid=1), _artifact(uid=2))
    first_assignment = _assignment(batch_id=batch_id, artifact=artifacts[0], task=_task("first"))
    second_assignment = _assignment(batch_id=batch_id, artifact=artifacts[1], task=_task("second"))
    request_active_attempts: list[tuple[PlatformTaskAttemptIdentity, ...]] = []

    class _Platform:
        async def request_miner_task_work(
            self,
            *,
            active_attempts: tuple[PlatformTaskAttemptIdentity, ...],
            **_kwargs: object,
        ) -> tuple[object, ...]:
            request_active_attempts.append(active_attempts)
            return ()

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=_unexpected_execute_artifact_assignments,
        target_concurrency=3,
        max_active_artifacts=2,
    )
    for index, assignment in enumerate((first_assignment, second_assignment), start=1):
        group = _assigned_group(artifact_id=assignment.artifact.artifact_id)
        assert group.put_nowait(assignment)
        group.mark_dispatch_ready()
        claimed = group.claim_initial_for_dispatch(assignment)
        assert claimed is not None
        claimed.mark_started(UUID(int=index))
        worker._active_artifacts[(batch_id, assignment.artifact.artifact_id)] = group

    await worker.run_once()
    await _wait_for_work_request_done(worker)

    assert len(request_active_attempts) == 1
    assert {
        (attempt.artifact_id, attempt.task_id, attempt.validator_session_id)
        for attempt in request_active_attempts[0]
    } == {
        (first_assignment.artifact.artifact_id, first_assignment.task.task_id, UUID(int=1)),
        (second_assignment.artifact.artifact_id, second_assignment.task.task_id, UUID(int=2)),
    }


async def test_worker_resumes_polling_after_closed_artifact_group_is_collected() -> None:
    old_batch_id = uuid4()
    old_artifact = _artifact(uid=1)
    new_batch_id = uuid4()
    new_artifact = _artifact(uid=2)
    new_assignment = _assignment(batch_id=new_batch_id, artifact=new_artifact, task=_task("new"))

    class _Platform:
        def __init__(self) -> None:
            self.request_count = 0

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            self.request_count += 1
            return (new_assignment,)

        def submit_miner_task_work_results(self, _results: object) -> tuple[object, ...]:
            return ()

    async def completed_executor() -> None:
        return

    async def idle_executor(
        _artifact_id: UUID,
        _assigned_work,
        close_requested: asyncio.Event,
        _result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        await close_requested.wait()

    platform = _Platform()
    worker = PlatformWorkWorker(
        platform=platform,  # type: ignore[arg-type]
        execute_artifact_assignments=idle_executor,
        target_concurrency=1,
        max_active_artifacts=1,
    )
    old_group = _assigned_group(artifact_id=old_artifact.artifact_id)
    old_group.state = platform_work_worker_module._ArtifactGroupState.CLOSING
    old_group.task = asyncio.create_task(completed_executor())
    await old_group.task
    worker._active_artifacts[(old_batch_id, old_artifact.artifact_id)] = old_group

    await _run_once_and_consume_platform_work(worker)

    assert platform.request_count == 1
    assert set(worker._active_artifacts) == {(new_batch_id, new_artifact.artifact_id)}
    new_group = worker._active_artifacts[(new_batch_id, new_artifact.artifact_id)]
    assert new_group.local_inflight_count() == 1

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

        async def request_miner_task_work(
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

    await _run_once_and_consume_platform_work(worker)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    clock.advance(11.0)
    await _run_once_and_consume_platform_work(worker)
    await _wait_for_work_request_done(worker)

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

        async def request_miner_task_work(
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

    await _run_once_and_consume_platform_work(worker)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    group = next(iter(worker._active_artifacts.values()))
    assert group.assignment_queue.qsize() == 1
    assert group.local_inflight_count() == 2

    clock.advance(11.0)
    await _run_once_and_consume_platform_work(worker)

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

    await _run_once_and_consume_platform_work(worker)

    assert group.local_inflight_count() == 2
    assert {
        record.assignment.task.task_id
        for record in group.assignment_records.values()
        if record.state is _AssignmentState.DISPATCHABLE_QUEUED
    } == {queued_assignment.task.task_id}
    assert group.assignment_queue.qsize() == 1

    clock.advance(11.0)
    await _run_once_and_consume_platform_work(worker)

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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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
    await _run_once_and_consume_platform_work(worker)
    await asyncio.wait_for(result_ready.wait(), timeout=1)
    await _run_once_and_consume_platform_work(worker)

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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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
    await _run_once_and_consume_platform_work(worker)
    await asyncio.sleep(0)
    await _run_once_and_consume_platform_work(worker)
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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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
    await _run_once_and_consume_platform_work(worker)
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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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
    await _run_once_and_consume_platform_work(worker)
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


async def test_worker_defers_delivery_failure_until_started_sibling_result_is_collected() -> None:
    """Prevent platform-wide dispatch failure from racing ahead of a started sibling result."""

    batch_id = uuid4()
    artifact = _artifact(uid=1)
    delivery_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("delivery"))
    sibling_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("sibling"))
    delivery_emitted = asyncio.Event()
    release_sibling = asyncio.Event()
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return (delivery_assignment, sibling_assignment)

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def delivery_then_sibling_executor(
        _artifact_id: UUID,
        assigned_work,
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        delivery_claim = await assigned_work.claim_for_dispatch()
        delivery_claim.mark_started(uuid4())
        sibling_claim = await assigned_work.claim_for_dispatch()
        sibling_claim.mark_started(uuid4())

        await result_queue.put(
            _platform_result(
                assignment=delivery_claim.assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            )
        )
        delivery_emitted.set()
        await release_sibling.wait()
        await result_queue.put(
            _platform_result(
                assignment=sibling_claim.assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
            )
        )

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=delivery_then_sibling_executor,
        target_concurrency=2,
        max_active_artifacts=2,
    )

    await _run_once_and_consume_platform_work(worker)
    await asyncio.wait_for(delivery_emitted.wait(), timeout=1)
    await worker.run_once()

    assert submitted == []
    assert len(worker._active_attempts()) == 2
    assert {
        (identity.task_id, identity.attempt_number)
        for identity in worker._active_attempts()
    } == {
        (delivery_assignment.task.task_id, delivery_assignment.attempt_number),
        (sibling_assignment.task.task_id, sibling_assignment.attempt_number),
    }
    assert worker._pending_results == []

    release_sibling.set()
    task = next(iter(worker._active_artifacts.values())).task
    assert task is not None
    await asyncio.wait_for(task, timeout=1)
    await worker.run_once()

    assert len(submitted) == 1
    assert [result.task_id for result in submitted[0]] == [
        sibling_assignment.task.task_id,
        delivery_assignment.task.task_id,
    ]
    assert submitted[0][0].terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert submitted[0][1].terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert worker._pending_results == []
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

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
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

    await _run_once_and_consume_platform_work(worker)
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


async def test_worker_submits_deferred_delivery_failure_when_executor_closes_with_started_sibling() -> None:
    """Prevent a deferred delivery-wide failure from disappearing during executor-close cleanup."""

    batch_id = uuid4()
    artifact = _artifact(uid=1)
    delivery_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("delivery"))
    sibling_assignment = _assignment(batch_id=batch_id, artifact=artifact, task=_task("sibling"))
    submitted: list[tuple[PlatformOwnedTaskResult, ...]] = []

    class _Platform:
        def __init__(self) -> None:
            self.requested = False

        async def request_miner_task_work(self, **_kwargs: object) -> tuple[MinerTaskWorkAssignment, ...]:
            if self.requested:
                return ()
            self.requested = True
            return (delivery_assignment, sibling_assignment)

        def submit_miner_task_work_results(
            self,
            results: tuple[PlatformOwnedTaskResult, ...],
        ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
            submitted.append(results)
            return tuple(_ack(result) for result in results)

    async def closing_executor(
        _artifact_id: UUID,
        assigned_work,
        _close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assigned_work.mark_dispatch_ready()
        delivery_claim = await assigned_work.claim_for_dispatch()
        delivery_claim.mark_started(uuid4())
        sibling_claim = await assigned_work.claim_for_dispatch()
        sibling_claim.mark_started(uuid4())
        await result_queue.put(
            _platform_result(
                assignment=delivery_claim.assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            )
        )

    worker = PlatformWorkWorker(
        platform=_Platform(),  # type: ignore[arg-type]
        execute_artifact_assignments=closing_executor,
        target_concurrency=2,
        max_active_artifacts=1,
    )

    await _run_once_and_consume_platform_work(worker)
    task = next(iter(worker._active_artifacts.values())).task
    assert task is not None
    await asyncio.wait_for(task, timeout=1)
    await worker.run_once()

    assert len(submitted) == 1
    assert [result.task_id for result in submitted[0]] == [delivery_assignment.task.task_id]
    assert submitted[0][0].terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert worker._pending_results == []
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
    successful: bool = False,
    validator_session_id: UUID | None = None,
) -> PlatformOwnedTaskResult:
    batch_id = assignment.batch_id if assignment is not None else uuid4()
    artifact_id = assignment.artifact.artifact_id if assignment is not None else uuid4()
    task_id = assignment.task.task_id if assignment is not None else uuid4()
    attempt_number = assignment.attempt_number if assignment is not None else 1
    uid = assignment.artifact.uid if assignment is not None else 7
    session_id = validator_session_id or uuid4()
    now = datetime.now(UTC)
    session = Session(
        session_id=session_id,
        uid=uid,
        task_id=task_id,
        issued_at=now,
        expires_at=now + timedelta(minutes=5),
        budget_usd=assignment.task.budget_usd if assignment is not None else 0.05,
        status=SessionStatus.COMPLETED,
    )
    result = (
        MinerTaskRunSubmission(
            batch_id=batch_id,
            run=MinerTaskRun(
                session_id=session_id,
                uid=uid,
                artifact_id=artifact_id,
                task_id=task_id,
                response=Response(text="answer"),
                details=EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        total_score=1.0,
                        scoring_version="test",
                    )
                ),
                completed_at=now,
            ),
            score=1.0,
            session=session,
        )
        if successful
        else None
    )
    return PlatformOwnedTaskResult(
        batch_id=batch_id,
        artifact_id=artifact_id,
        task_id=task_id,
        attempt_number=attempt_number,
        result=result,
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


def _platform_execution(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
) -> PlatformOwnedTaskExecution:
    issued_at = datetime.now(UTC)
    session_id = uuid4()
    return PlatformOwnedTaskExecution(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        attempt_number=1,
        max_attempts=2,
        validator_session_id=session_id,
        uid=artifact.uid,
        miner_hotkey_ss58=artifact.miner_hotkey_ss58 or "miner-hotkey",
        started_at=issued_at,
        execution_completed_at=issued_at,
        response=Response(text="answer"),
        session=Session(
            session_id=session_id,
            uid=artifact.uid,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(minutes=5),
            budget_usd=task.budget_usd,
            status=SessionStatus.COMPLETED,
        ),
        usage=TokenUsageSummary.empty(),
        total_tool_usage=ToolUsageSummary.zero(),
    )


def _ack(
    result: PlatformOwnedTaskResult,
    *,
    outcome: str = "accepted",
    reason_code: str | None = None,
    reason: str | None = None,
) -> PlatformTaskResultAcknowledgement:
    return PlatformTaskResultAcknowledgement(
        batch_id=result.batch_id,
        artifact_id=result.artifact_id,
        task_id=result.task_id,
        attempt_number=result.attempt_number,
        outcome=outcome,
        canonical=True,
        reason_code=reason_code,
        reason=reason,
    )
