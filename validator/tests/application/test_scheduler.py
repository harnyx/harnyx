from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

import harnyx_validator.application.scheduler as scheduler_module
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolExecutionFacts
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunSubmission,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.invoke_entrypoint import SandboxInvocationError
from harnyx_validator.application.ports.progress import RunProgressPage, RunProgressSummary
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    ArtifactExecutionFailedError,
    UnexpectedArtifactExecutionError,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError
from validator.tests.fixtures.fakes import FakeReceiptLog
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token
_ASSIGNMENT_TOKEN_1 = "assignment-token-1"  # noqa: S105 - fixed test-only assignment token
_ASSIGNMENT_TOKEN_2 = "assignment-token-2"  # noqa: S105 - fixed test-only assignment token


def _runner_for(func: Callable[..., Awaitable[ArtifactEvaluationOutcome]]) -> object:
    class _Runner:
        async def evaluate_artifact_with_state(self, **kwargs: Any) -> ArtifactEvaluationOutcome:
            return await func(**kwargs)

    return _Runner()


@pytest.fixture
def blocking_executor() -> ThreadPoolExecutor:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-validator-batch-blocking")
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class DummySandboxManager(SandboxManager):
    def __init__(self) -> None:
        self.starts: list[object | None] = []
        self.stops: list[SandboxDeployment] = []

    def start(self, options: object | None = None) -> SandboxDeployment:
        self.starts.append(options)
        return SandboxDeployment(client=object())

    def stop(self, deployment: SandboxDeployment) -> None:
        self.stops.append(deployment)


class DummyEvaluationRecordStore:
    def __init__(self) -> None:
        self.records_by_batch: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records_by_batch.append(result)


class DummyReceiptLog(ReceiptLogPort):
    def __init__(self) -> None:
        self._records: dict[str, object] = {}

    def record(self, receipt: object) -> None:
        self._records[str(len(self._records))] = receipt

    def start_pending_receipt(self, *, started_call) -> None:
        return None

    def complete_pending_receipt(self, receipt, settle_usage):
        return settle_usage()

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        return None

    def lookup(self, receipt_id: str) -> object | None:
        return self._records.get(receipt_id)

    def values(self):
        return tuple(self._records.values())

    def for_session(self, session_id):
        return ()

    def clear_session(self, session_id) -> None:
        return None


class DummyProgressRecorder:
    def __init__(self, recorded: frozenset[tuple[UUID, UUID]] = frozenset()) -> None:
        self._recorded = set(recorded)
        self._submissions_by_pair: dict[tuple[UUID, UUID], MinerTaskRunSubmission] = {}
        self._sequence_by_pair: dict[tuple[UUID, UUID], int] = {}
        self._pair_by_sequence: dict[int, tuple[UUID, UUID]] = {}
        self._latest_attempt_number_by_pair: dict[tuple[UUID, UUID], int] = {}
        self._next_sequence = 1

    def register(self, _batch) -> None:
        return None

    def record(self, result: MinerTaskRunSubmission) -> None:
        pair = (result.run.artifact_id, result.run.task_id)
        self._recorded.add(pair)
        if pair not in self._sequence_by_pair:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._sequence_by_pair[pair] = sequence
            self._pair_by_sequence[sequence] = pair
        self._submissions_by_pair[pair] = result
        self._latest_attempt_number_by_pair[pair] = max(
            self._latest_attempt_number_by_pair.get(pair, 0),
            1,
        )

    def record_terminated_attempt(self, attempt: MinerTaskAttemptAuditRecord) -> None:
        pair = (attempt.artifact_id, attempt.task_id)
        self._latest_attempt_number_by_pair[pair] = max(
            self._latest_attempt_number_by_pair.get(pair, 0),
            attempt.attempt_number,
        )

    def next_attempt_number(self, _batch_id: UUID, artifact_id: UUID, task_id: UUID) -> int:
        return self._latest_attempt_number_by_pair.get((artifact_id, task_id), 0) + 1

    def recorded_pairs(self, _batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return frozenset(self._recorded)

    def summary(self, batch_id: UUID) -> RunProgressSummary:
        completed = len(self._recorded)
        return {
            "batch_id": batch_id,
            "total": completed,
            "completed": completed,
            "remaining": 0,
            "latest_sequence": self._next_sequence - 1,
            "provider_evidence": (),
        }

    def completed_run_page(
        self,
        _batch_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> RunProgressPage:
        latest_sequence = self._next_sequence - 1
        sequences = tuple(range(after_sequence + 1, min(latest_sequence, after_sequence + limit) + 1))
        items = []
        for sequence in sequences:
            pair = self._pair_by_sequence.get(sequence)
            if pair is None:
                continue
            items.append(
                {
                    "sequence": sequence,
                    "kind": "completed_run",
                    "submission": self._submissions_by_pair[pair],
                    "attempt": None,
                }
            )
        next_after_sequence = items[-1]["sequence"] if items else after_sequence
        return {
            "batch_id": _batch_id,
            "after_sequence": after_sequence,
            "limit": limit,
            "latest_sequence": latest_sequence,
            "next_after_sequence": next_after_sequence,
            "has_more": next_after_sequence < latest_sequence,
            "items": tuple(items),
        }

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        return None

    def record_provider_call(self, *, session_id: UUID, provider: str, model: str) -> None:
        return None

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        return None

    def consume_provider_failures(self, session_id: UUID) -> tuple[dict[str, object], ...]:
        return ()

    def clear_task_session(self, session_id: UUID) -> None:
        return None


def _task(text: str, *, budget_usd: float = 0.05) -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text=text),
        reference_answer=ReferenceAnswer(text=f"reference {text}"),
        budget_usd=budget_usd,
    )


def _sandbox_invocation_error(
    message: str,
    *,
    status_code: int = 0,
    detail_exception: str = "RuntimeError",
    detail_error: str | None = None,
) -> SandboxInvocationError:
    return SandboxInvocationError(
        message,
        status_code=status_code,
        detail_code=None,
        detail_exception=detail_exception,
        detail_error=detail_error or message,
    )


def _llm_receipt(
    *,
    session_id: UUID,
    uid: int,
    total_tokens: int,
    elapsed_ms: float,
    cost_usd: float = 0.0042,
    active_attempt: int = 1,
) -> ToolCall:
    return ToolCall(
        receipt_id=uuid4().hex,
        session_id=session_id,
        uid=uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 27, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="req",
            request_payload={
                "args": [],
                "kwargs": {
                    "provider": "chutes",
                    "model": "google/gemma-4-31B-turbo-TEE",
                },
            },
            response_hash="res",
            response_payload={"usage": {"total_tokens": total_tokens}},
            cost_usd=cost_usd,
            actual_cost_usd=cost_usd,
            actual_cost_provider="openrouter",
            execution=ToolExecutionFacts(elapsed_ms=elapsed_ms),
            extra={"session_active_attempt": str(active_attempt)},
        ),
    )


def _submission_for_task(
    *,
    batch_id: UUID,
    validator_uid: int,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    error: EvaluationError | None = None,
) -> MinerTaskRunSubmission:
    issued_at = datetime(2025, 10, 27, tzinfo=UTC)
    session_id = uuid4()
    session = Session(
        session_id=session_id,
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        usage=SessionUsage(),
        status=SessionStatus.ERROR if error is not None else SessionStatus.COMPLETED,
    )
    if error is None:
        run = MinerTaskRun(
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text=f"answer {task.query.text}"),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    total_score=1.0,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=issued_at,
        )
        return MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=validator_uid,
            run=run,
            score=1.0,
            usage=TokenUsageSummary.empty(),
            session=session,
        )

    run = MinerTaskRun(
        session_id=session_id,
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        details=EvaluationDetails(
            error=error,
            total_tool_usage=ToolUsageSummary.zero(),
        ),
        completed_at=issued_at,
    )
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=validator_uid,
        run=run,
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


async def test_scheduler_runs_all_tasks_for_each_artifact(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    recorded_requests: list[tuple[int, MinerTask]] = []
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=2,
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (
        ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),
        ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0),
    )
    batch_id = uuid4()
    result = await scheduler.run(batch_id=batch_id, requested_artifacts=artifacts)
    assert len(sandbox_manager.starts) == 2
    assert len(sandbox_manager.stops) == 2
    assert len(recorded_requests) == len(tasks) * 2
    assert result.completed_run_count == len(recorded_requests)
    assert result.tasks == tasks
    assert len(evaluation_records.records_by_batch) == result.completed_run_count

    batch_logs = [extra for message, extra in captured_logs if message == "miner-task batch execution started"]
    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]

    assert batch_logs == [
        {
            "batch_id": str(batch_id),
            "artifact_count": 2,
            "task_count": 2,
            "artifact_parallelism": 2,
            "artifact_task_parallelism": 20,
            "recorded_pair_count": 0,
        }
    ]
    assert len(artifact_logs) == 2
    assert {extra["artifact_id"] for extra in artifact_logs} == {str(artifact.artifact_id) for artifact in artifacts}
    for artifact_index, extra in enumerate(artifact_logs, start=1):
        assert extra["batch_id"] == str(batch_id)
        assert extra["artifact_index"] == artifact_index
        assert extra["artifact_count"] == 2
        assert extra["planned_task_count"] == 2
        assert extra["success_count"] == 2
        assert extra["failure_count"] == 0
        assert extra["unresolved_count"] == 0
        assert extra["setup_ms"] >= 0.0
        assert extra["evaluation_ms"] >= 0.0
        assert extra["teardown_ms"] >= 0.0
        assert extra["total_ms"] >= 0.0
        assert extra["outcome"] == "completed"
        assert extra["error_code"] is None


async def test_scheduler_caps_task_sessions_across_whole_batch(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = tuple(_task(f"task {index}") for index in range(4))
    artifacts = (
        ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),
        ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0),
    )
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    active_task_sessions = 0
    max_active_task_sessions = 0
    entered_task_sessions = 0
    first_wave_entered = asyncio.Event()
    active_lock = asyncio.Lock()

    async def run_tracked_task_session(task_session_limiter) -> None:
        nonlocal active_task_sessions, max_active_task_sessions, entered_task_sessions
        async with task_session_limiter:
            async with active_lock:
                active_task_sessions += 1
                entered_task_sessions += 1
                max_active_task_sessions = max(max_active_task_sessions, active_task_sessions)
                if entered_task_sessions == 2:
                    first_wave_entered.set()
            await first_wave_entered.wait()
            await asyncio.sleep(0)
            async with active_lock:
                active_task_sessions -= 1

    class _TrackedRunner:
        async def evaluate_artifact_with_state(
            self,
            *,
            batch_id: UUID,
            artifact: ScriptArtifactSpec,
            tasks,
            task_session_limiter,
            **_kwargs,
        ) -> ArtifactEvaluationOutcome:
            await asyncio.gather(*(run_tracked_task_session(task_session_limiter) for _task_item in tasks))
            return ArtifactEvaluationOutcome(
                submissions=tuple(
                    _submission_for_task(
                        batch_id=batch_id,
                        validator_uid=41,
                        artifact=artifact,
                        task=task,
                    )
                    for task in tasks
                ),            )

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=2,
            artifact_task_parallelism=2,
        ),
        progress=DummyProgressRecorder(),
    )
    scheduler._runner = _TrackedRunner()

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert result.completed_run_count == len(tasks) * len(artifacts)
    assert max_active_task_sessions == 2


async def test_scheduler_flattens_runs_in_requested_artifact_order_when_completion_is_out_of_order(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("ordered")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=2,
        ),
        progress=DummyProgressRecorder(),
    )
    batch_id = uuid4()
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    first_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=task,
    )
    second_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=second_artifact,
        task=task,
    )
    second_finished = asyncio.Event()

    async def run_single_artifact(**kwargs):
        artifact = kwargs["artifact"]
        if artifact.artifact_id == first_artifact.artifact_id:
            await second_finished.wait()
            return scheduler_module._CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=(first_submission,),
            )
        second_finished.set()
        return scheduler_module._CompletedArtifactResult(
            artifact_id=artifact.artifact_id,
            submissions=(second_submission,),
        )

    scheduler._run_single_artifact = run_single_artifact  # type: ignore[method-assign]

    result = await scheduler.run(
        batch_id=batch_id,
        requested_artifacts=(first_artifact, second_artifact),
    )
    assert result.completed_run_count == 2


async def test_scheduler_refills_artifact_slots_without_waiting_for_all_started_artifacts(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("slot refill")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=2,
        ),
        progress=DummyProgressRecorder(),
    )
    batch_id = uuid4()
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    third_artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="c", size_bytes=0)
    first_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=task,
    )
    second_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=second_artifact,
        task=task,
    )
    third_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=third_artifact,
        task=task,
    )
    first_artifact_release = asyncio.Event()
    second_artifact_finished = asyncio.Event()
    third_artifact_started = asyncio.Event()

    async def run_single_artifact(**kwargs):
        artifact = kwargs["artifact"]
        if artifact.artifact_id == first_artifact.artifact_id:
            await first_artifact_release.wait()
            return scheduler_module._CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=(first_submission,),
            )
        if artifact.artifact_id == second_artifact.artifact_id:
            second_artifact_finished.set()
            return scheduler_module._CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=(second_submission,),
            )
        third_artifact_started.set()
        return scheduler_module._CompletedArtifactResult(
            artifact_id=artifact.artifact_id,
            submissions=(third_submission,),
        )

    scheduler._run_single_artifact = run_single_artifact  # type: ignore[method-assign]

    run_task = asyncio.create_task(
        scheduler.run(
            batch_id=batch_id,
            requested_artifacts=(first_artifact, second_artifact, third_artifact),
        )
    )

    await asyncio.wait_for(second_artifact_finished.wait(), timeout=1.0)
    await asyncio.wait_for(third_artifact_started.wait(), timeout=1.0)
    first_artifact_release.set()

    result = await run_task

    assert result.completed_run_count == 3


async def test_scheduler_stops_dequeuing_queued_artifacts_when_validator_failure_is_discovered() -> None:
    task = _task("stop queued work")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    third_artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="c", size_bytes=0)
    failure_teardown_release = Event()
    second_artifact_stopped = Event()

    class ControlledSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            assert isinstance(options, dict)
            return SandboxDeployment(client=options["artifact_id"])

        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            if deployment.client == first_artifact.artifact_id:
                assert failure_teardown_release.wait(timeout=1.0)
                return
            if deployment.client == second_artifact.artifact_id:
                second_artifact_stopped.set()

    sandbox_manager = ControlledSandboxManager()
    progress = DummyProgressRecorder()
    blocking_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-validator-race-blocking")
    try:
        scheduler = EvaluationScheduler(
            tasks=(task,),
            subtensor_client=subtensor,
            sandbox_manager=sandbox_manager,
            session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
            evaluation_records=DummyEvaluationRecordStore(),
            receipt_log=DummyReceiptLog(),
            blocking_executor=blocking_executor,
            orchestrator_factory=lambda _client: object(),
            sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
            clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
            config=SchedulerConfig(
                token_secret_bytes=8,
                session_ttl=timedelta(minutes=5),
                artifact_parallelism=2,
            ),
            progress=progress,
        )
        batch_id = uuid4()
        failure_discovered = asyncio.Event()
        successful_submission = _submission_for_task(
            batch_id=batch_id,
            validator_uid=41,
            artifact=second_artifact,
            task=task,
        )

        class _FailureRaceRunner:
            async def evaluate_artifact_with_state(
                self,
                *,
                artifact: ScriptArtifactSpec,
                **_kwargs,
            ) -> ArtifactEvaluationOutcome:
                if artifact.artifact_id == first_artifact.artifact_id:
                    failure_discovered.set()
                    raise ValidatorBatchFailedError(
                        error_code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                        message="terminal timeout",
                        failure_detail=ValidatorBatchFailureDetail(
                            error_code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                            error_message="terminal timeout",
                            occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                            artifact_id=artifact.artifact_id,
                            task_id=task.task_id,
                            uid=artifact.uid,
                            exception_type="TimeoutException",
                        ),
                    )
                await failure_discovered.wait()
                progress.record(successful_submission)
                return ArtifactEvaluationOutcome(
                    submissions=(successful_submission,),
                )

        scheduler._runner = _FailureRaceRunner()  # type: ignore[assignment]

        run_task = asyncio.create_task(
            scheduler.run(
                batch_id=batch_id,
                requested_artifacts=(first_artifact, second_artifact, third_artifact),
            )
        )
        assert await asyncio.to_thread(second_artifact_stopped.wait, 1.0)
        await asyncio.sleep(0.05)
        failure_teardown_release.set()

        with pytest.raises(ValidatorBatchFailedError, match="terminal timeout") as exc_info:
            await run_task
        assert exc_info.value.completed_submissions == (successful_submission,)
    finally:
        failure_teardown_release.set()
        blocking_executor.shutdown(wait=True, cancel_futures=True)
    assert [start["artifact_id"] for start in sandbox_manager.starts] == [
        first_artifact.artifact_id,
        second_artifact.artifact_id,
    ]


async def test_scheduler_logs_setup_failure_timing_summary(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    class FailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox boot failed")

    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = FailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    def orchestrator_factory(_client: object):
        raise AssertionError("orchestrator should not be created when sandbox start fails")

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    with pytest.raises(ValidatorBatchFailedError, match="sandbox boot failed"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))
    assert len(sandbox_manager.starts) == 2

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["uid"] == artifact.uid
    assert payload["artifact_index"] == 1
    assert payload["artifact_count"] == 1
    assert payload["planned_task_count"] == 1
    assert payload["success_count"] == 0
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["setup_ms"] == 123.0
    assert payload["evaluation_ms"] == 0.0
    assert payload["teardown_ms"] == 0.0
    assert payload["total_ms"] == 123.0
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "sandbox_start_failed"


async def test_scheduler_logs_teardown_failure_timing_summary(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    class FailingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            raise RuntimeError("sandbox stop failed")

    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = FailingStopSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    successful_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=tasks[0],
    )

    async def evaluate_successfully(**kwargs):
        _ = kwargs
        return ArtifactEvaluationOutcome(
            submissions=(successful_submission,),        )

    scheduler._runner = _runner_for(evaluate_successfully)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="sandbox stop failed"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["planned_task_count"] == 1
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 0
    assert payload["setup_ms"] == 123.0
    assert payload["evaluation_ms"] == 123.0
    assert payload["teardown_ms"] == 123.0
    assert payload["total_ms"] == 123.0
    assert payload["outcome"] == "teardown_failed"
    assert payload["error_code"] == str(MinerTaskErrorCode.SANDBOX_FAILED)


async def test_scheduler_preserves_validator_batch_failure_when_teardown_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    class FailingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            raise RuntimeError("sandbox stop failed")

    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = FailingStopSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )

    async def fail_evaluation(**kwargs):
        _ = kwargs
        raise ValidatorBatchFailedError(
            error_code="validator_internal_timeout",
            message="validator timeout",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="validator_internal_timeout",
                error_message="validator timeout",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
            ),
        )

    scheduler._runner = _runner_for(fail_evaluation)  # type: ignore[assignment]

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["planned_task_count"] == 1
    assert payload["success_count"] == 0
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["setup_ms"] == 123.0
    assert payload["evaluation_ms"] == 123.0
    assert payload["teardown_ms"] == 123.0
    assert payload["total_ms"] == 123.0
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "validator_internal_timeout"


async def test_scheduler_logs_evaluation_timing_summary_for_validator_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    def orchestrator_factory(_client: object):
        return object()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    async def fail_evaluation(**kwargs):
        _ = kwargs
        raise ValidatorBatchFailedError(
            error_code="validator_internal_timeout",
            message="validator timeout",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="validator_internal_timeout",
                error_message="validator timeout",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
            ),
        )

    scheduler._runner = _runner_for(fail_evaluation)  # type: ignore[assignment]

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["planned_task_count"] == 1
    assert payload["success_count"] == 0
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["setup_ms"] == 123.0
    assert payload["evaluation_ms"] == 123.0
    assert payload["teardown_ms"] == 123.0
    assert payload["total_ms"] == 123.0
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "validator_internal_timeout"


async def test_scheduler_logs_partial_progress_for_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    completed_task = _task("completed")
    pending_task = _task("pending")
    tasks = (completed_task, pending_task)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )

    async def fail_unexpectedly(**kwargs):
        _ = kwargs
        raise UnexpectedArtifactExecutionError(
            cause=RuntimeError("progress store failed"),
            completed_submissions=(completed_submission,),
            remaining_tasks=(pending_task,),
        )

    scheduler._runner = _runner_for(fail_unexpectedly)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="progress store failed"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["planned_task_count"] == 2
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["setup_ms"] == 123.0
    assert payload["evaluation_ms"] == 123.0
    assert payload["teardown_ms"] == 123.0
    assert payload["total_ms"] == 123.0
    assert payload["outcome"] == "unexpected_failure"
    assert payload["error_code"] is None


async def test_scheduler_cancels_remaining_artifact_workers_when_one_raises_unexpectedly(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("cancel sibling worker")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=2,
        ),
        progress=DummyProgressRecorder(),
    )
    second_worker_started = asyncio.Event()
    second_worker_cancelled = asyncio.Event()
    first_worker_released = asyncio.Event()

    async def run_artifact_worker(**_kwargs) -> None:
        if second_worker_started.is_set():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                second_worker_cancelled.set()
                raise
        second_worker_started.set()
        await first_worker_released.wait()
        raise RuntimeError("worker boom")

    scheduler._run_artifact_worker = run_artifact_worker  # type: ignore[method-assign]

    run_task = asyncio.create_task(
        scheduler.run(
            batch_id=uuid4(),
            requested_artifacts=(
                ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),
                ScriptArtifactSpec(uid=8, artifact_id=uuid4(), content_hash="b", size_bytes=0),
            ),
        )
    )

    await asyncio.wait_for(second_worker_started.wait(), timeout=1.0)
    first_worker_released.set()

    with pytest.raises(RuntimeError, match="worker boom"):
        await run_task

    assert second_worker_cancelled.is_set() is True


async def test_scheduler_logs_accounted_summary_for_validator_batch_failure_after_partial_progress(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    completed_task = _task("completed")
    unresolved_task = _task("unresolved")
    tasks = (completed_task, unresolved_task)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )
    failed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=unresolved_task,
        error=EvaluationError(code="sandbox_invocation_failed", message="sandbox failed"),
    )

    async def fail_artifact(**kwargs):
        _ = kwargs
        raise ValidatorBatchFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
            message="sandbox failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_invocation_failed",
                error_message="sandbox failed",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
                exception_type="SandboxInvocationError",
            ),
            completed_submissions=(completed_submission, failed_submission),
            remaining_tasks=(),
        )

    scheduler._runner = _runner_for(fail_artifact)  # type: ignore[assignment]

    with pytest.raises(ValidatorBatchFailedError, match="sandbox failed"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["planned_task_count"] == 2
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 1
    assert payload["unresolved_count"] == 0
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "sandbox_invocation_failed"


async def test_scheduler_preserves_validator_batch_failure_after_partial_progress_when_teardown_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    class FailingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            raise RuntimeError("sandbox stop failed")

    completed_task = _task("completed")
    unresolved_task = _task("unresolved")
    tasks = (completed_task, unresolved_task)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = FailingStopSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []
    captured_warnings: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    def capture_warning(message: str, *args, **kwargs) -> None:
        captured_warnings.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module.logger, "warning", capture_warning)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )
    failed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=unresolved_task,
        error=EvaluationError(code="sandbox_invocation_failed", message="sandbox failed"),
    )

    async def fail_artifact(**kwargs):
        _ = kwargs
        raise ValidatorBatchFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
            message="sandbox failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_invocation_failed",
                error_message="sandbox failed",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
                exception_type="SandboxInvocationError",
            ),
            completed_submissions=(completed_submission, failed_submission),
            remaining_tasks=(),
        )

    scheduler._runner = _runner_for(fail_artifact)  # type: ignore[assignment]

    with pytest.raises(ValidatorBatchFailedError, match="sandbox failed"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["planned_task_count"] == 2
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 1
    assert payload["unresolved_count"] == 0
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "sandbox_invocation_failed"

    assert captured_warnings == [
        (
            "artifact teardown failed after primary failure",
            {
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "primary_outcome": "validator_batch_failure",
                "primary_error_code": "sandbox_invocation_failed",
            },
        )
    ]


async def test_scheduler_logs_partial_progress_when_setup_failure_backfill_raises(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            message="artifact setup failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_start_failed",
                error_message="artifact setup failed",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=tasks,
        )

    async def fail_if_backfill_called(**kwargs):
        _ = kwargs
        raise AssertionError("conclusive setup failure should not backfill untouched tasks")

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)
    monkeypatch.setattr(scheduler, "_record_artifact_failure", fail_if_backfill_called)

    with pytest.raises(ValidatorBatchFailedError, match="artifact setup failed") as exc_info:
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))
    assert exc_info.value.error_code == MinerTaskErrorCode.SANDBOX_START_FAILED
    assert exc_info.value.completed_submissions == ()
    assert exc_info.value.remaining_tasks == tasks
    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["planned_task_count"] == 2
    assert payload["success_count"] == 0
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 2
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "sandbox_start_failed"


async def test_scheduler_returns_platform_result_for_assigned_task_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("assigned")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    progress = DummyProgressRecorder()
    now = datetime(2025, 10, 27, tzinfo=UTC)

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )
    batch_id = uuid4()

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            message="artifact setup failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="sandbox_start_failed",
                error_message="artifact setup failed",
                occurred_at=now,
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=(task,),
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    result = await scheduler.run_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert result.batch_id == batch_id
    assert result.artifact_id == artifact.artifact_id
    assert result.task_id == task.task_id
    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "sandbox_start_failed"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert result.terminal_attempt.validator_session_id != UUID(int=0)
    assert progress.next_attempt_number(batch_id, artifact.artifact_id, task.task_id) == 2


async def test_scheduler_runs_multiple_platform_assignments_in_one_artifact_sandbox(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    now = datetime(2025, 10, 27, tzinfo=UTC)

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                return TaskRunOutcome(
                    run=MinerTaskRun(
                        session_id=request.session_id,
                        uid=request.uid,
                        artifact_id=request.artifact_id,
                        task_id=request.task.task_id,
                        response=Response(text=f"answer {request.task.query.text}"),
                        details=EvaluationDetails(
                            score_breakdown=ScoreBreakdown(
                                comparison_score=1.0,
                                total_score=1.0,
                                scoring_version="v1",
                            ),
                            total_tool_usage=ToolUsageSummary.zero(),
                        ),
                        completed_at=now,
                    ),
                    usage=TokenUsageSummary.empty(),
                )

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    assignments = tuple(
        MinerTaskWorkAssignment(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            attempt_number=index,
            max_attempts=3,
            assignment_token=f"{_ASSIGNMENT_TOKEN}-{index}",
        )
        for index, task in enumerate(tasks, start=1)
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    close_requested = asyncio.Event()
    close_requested.set()

    await scheduler.run_assigned_artifact_queue(
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=assignments,
        assignment_queue=asyncio.Queue(),
        close_requested=close_requested,
        result_queue=result_queue,
    )

    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1
    results = (result_queue.get_nowait(), result_queue.get_nowait())
    assert {result.task_id for result in results} == {task.task_id for task in tasks}
    assert {result.attempt_number for result in results} == {1, 2}
    assert all(
        result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
        for result in results
    )


async def test_scheduler_returns_pair_results_for_assigned_artifact_script_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    progress = DummyProgressRecorder()
    now = datetime(2025, 10, 27, tzinfo=UTC)
    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry()),
        evaluation_records=DummyEvaluationRecordStore(),
        receipt_log=DummyReceiptLog(),
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: now,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        progress=progress,
    )
    artifact = ScriptArtifactSpec(
        uid=3,
        artifact_id=uuid4(),
        content_hash="a",
        size_bytes=0,
        miner_hotkey_ss58="miner-hotkey",
    )
    batch_id = uuid4()
    first_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[0],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_1,
    )
    second_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=tasks[1],
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN_2,
    )
    assignment_queue: asyncio.Queue[MinerTaskWorkAssignment] = asyncio.Queue()
    assignment_queue.put_nowait(second_assignment)
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()

    async def fail_setup(**kwargs):
        _ = kwargs
        raise ArtifactExecutionFailedError(
            error_code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
            message="script validation failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="script_validation_failed",
                error_message="script validation failed",
                occurred_at=now,
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(),
            remaining_tasks=tasks,
        )

    monkeypatch.setattr(scheduler, "_start_artifact_with_retry", fail_setup)

    await scheduler.run_assigned_artifact_queue(
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(first_assignment,),
        assignment_queue=assignment_queue,
        close_requested=asyncio.Event(),
        result_queue=result_queue,
    )

    results = (result_queue.get_nowait(), result_queue.get_nowait())
    assert {result.task_id for result in results} == {task.task_id for task in tasks}
    assert all(result.result is not None for result in results)
    assert all(
        result.result is not None
        and result.result.run.details.error is not None
        and result.result.run.details.error.code == MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED
        for result in results
    )
    assert all(
        result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
        for result in results
    )
    assert all(
        result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
        for result in results
    )


async def test_scheduler_logs_partial_progress_for_validator_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    completed_task = _task("completed")
    pending_task = _task("pending")
    tasks = (completed_task, pending_task)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(scheduler_module.measurement_logger, "info", capture_info)
    monkeypatch.setattr(scheduler_module, "_monotonic_elapsed_ms", lambda **_: 123.0)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    batch_id = uuid4()
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )

    async def fail_evaluation(**kwargs):
        _ = kwargs
        raise ValidatorBatchFailedError(
            error_code="validator_internal_timeout",
            message="validator timeout",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="validator_internal_timeout",
                error_message="validator timeout",
                occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
            completed_submissions=(completed_submission,),
            remaining_tasks=(pending_task,),
        )

    scheduler._runner = _runner_for(fail_evaluation)  # type: ignore[assignment]

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout"):
        await scheduler.run(batch_id=batch_id, requested_artifacts=(artifact,))

    artifact_logs = [extra for message, extra in captured_logs if message == "miner-task artifact execution finished"]
    assert len(artifact_logs) == 1
    payload = artifact_logs[0]
    assert payload["planned_task_count"] == 2
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["outcome"] == "validator_batch_failure"
    assert payload["error_code"] == "validator_internal_timeout"


async def test_scheduler_avoids_asyncio_to_thread_for_blocking_work(
    monkeypatch: pytest.MonkeyPatch,
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer one"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    async def _unexpected_to_thread(*args, **kwargs):
        raise AssertionError("scheduler should not use asyncio.to_thread")

    monkeypatch.setattr(scheduler_module.asyncio, "to_thread", _unexpected_to_thread)

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert result.completed_run_count == 1
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1


async def test_scheduler_cancellation_does_not_wait_for_blocking_lane_shutdown(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("shutdown")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    stop_started = Event()
    release_stop = Event()
    stop_finished = Event()

    class BlockingStopSandboxManager(DummySandboxManager):
        def stop(self, deployment: SandboxDeployment) -> None:
            self.stops.append(deployment)
            stop_started.set()
            try:
                release_stop.wait(timeout=5.0)
            finally:
                stop_finished.set()

    sandbox_manager = BlockingStopSandboxManager()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer shutdown"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    run_task = asyncio.create_task(
        scheduler.run(
            batch_id=uuid4(),
            requested_artifacts=(ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),),
        )
    )

    try:
        assert await asyncio.to_thread(stop_started.wait, 1.0) is True
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=0.2)
        assert release_stop.is_set() is False
    finally:
        release_stop.set()
    assert await asyncio.to_thread(stop_finished.wait, 1.0) is True


async def test_scheduler_records_zero_score_when_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("unstable")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class FailingOrchestrator:
            async def evaluate(self, request):
                raise _sandbox_invocation_error("upstream tool failure")

        return FailingOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, 0, 0, 0, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0, task_retry_count=1),)
    with pytest.raises(ValidatorBatchFailedError, match="upstream tool failure") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc_info.value.completed_submissions is not None
    assert len(exc_info.value.completed_submissions) == 1
    assert exc_info.value.completed_submissions[0].score == 0.0
    assert exc_info.value.completed_submissions[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="upstream tool failure",
    )
    assert evaluation_records.records_by_batch == list(exc_info.value.completed_submissions)


async def test_scheduler_retries_only_transient_sandbox_invocation_errors(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class FailingThenSuccessfulOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise _sandbox_invocation_error("upstream tool failure")
            details = EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=0.75,
                    total_score=0.75,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            )
            run = MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text=f"answer {request.task.query.text}"),
                details=details,
                completed_at=datetime(2025, 10, 27, tzinfo=UTC),
            )
            return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

    orchestrator = FailingThenSuccessfulOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0, task_retry_count=1),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert result.completed_run_count == 2
    assert len(evaluation_records.records_by_batch) == 2
    assert orchestrator.calls == 3
    assert evaluation_records.records_by_batch[0].score == pytest.approx(0.75)
    assert evaluation_records.records_by_batch[0].run.response == Response(text="answer first")
    assert evaluation_records.records_by_batch[1].score == pytest.approx(0.75)
    assert evaluation_records.records_by_batch[1].run.response == Response(text="answer second")


async def test_scheduler_fails_batch_for_generic_post_invoke_error(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"), _task("second"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    class GenericFailureThenSuccessOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("embedding client unavailable")
            details = EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=0.75,
                    total_score=0.75,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            )
            run = MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text=f"answer {request.task.query.text}"),
                details=details,
                completed_at=datetime(2025, 10, 27, tzinfo=UTC),
            )
            return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

    orchestrator = GenericFailureThenSuccessOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), task_retry_count=1, content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="embedding client unavailable") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert orchestrator.calls == 1
    assert evaluation_records.records_by_batch == []


async def test_scheduler_records_retry_exhausted_internal_timeout_as_task_failure(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("first"),)
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = FakeReceiptLog()

    class AlwaysTimeoutOrchestrator:
        def __init__(self) -> None:
            self.calls = 0
            self.session_ids: list[UUID] = []

        async def evaluate(self, request):
            self.calls += 1
            self.session_ids.append(request.session_id)
            raise httpx.ReadTimeout(
                "embedding timed out",
                request=httpx.Request("POST", "https://validator.invalid/scoring"),
            )

    orchestrator = AlwaysTimeoutOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), task_retry_count=1, content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="embedding timed out") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == "validator_internal_timeout"
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert evaluation_records.records_by_batch == []


async def test_scheduler_uses_successful_baseline_across_execution_for_timeout_miner_owned(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("baseline"), _task("timeout"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = FakeReceiptLog()

    class BaselineThenTimeoutOrchestrator:
        def __init__(self, receipt_log: FakeReceiptLog) -> None:
            self._receipt_log = receipt_log
            self.timeout_calls = 0

        async def evaluate(self, request):
            if request.task.query.text == "baseline":
                receipt = _llm_receipt(
                    session_id=request.session_id,
                    uid=request.uid,
                    total_tokens=100,
                    elapsed_ms=1000.0,
                )
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="baseline answer"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(
                    run=run,
                    tool_receipts=(receipt,),
                    usage=TokenUsageSummary.empty(),
                )

            self.timeout_calls += 1
            self._receipt_log.record(
                _llm_receipt(
                    session_id=request.session_id,
                    uid=request.uid,
                    total_tokens=100,
                    elapsed_ms=2500.0,
                )
            )
            raise _sandbox_invocation_error(
                "sandbox entrypoint request timed out",
                status_code=504,
                detail_exception="TimeoutException",
                detail_error="sandbox entrypoint request timed out",
            )

    orchestrator = BaselineThenTimeoutOrchestrator(receipt_log)
    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: orchestrator,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )
    artifacts = (
        ScriptArtifactSpec(
            uid=7,
            artifact_id=uuid4(),
            content_hash="a",
            size_bytes=0,
            task_retry_count=2,
        ),
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert result.completed_run_count == 2
    assert orchestrator.timeout_calls == 3
    assert evaluation_records.records_by_batch[0].score == pytest.approx(0.75)
    assert evaluation_records.records_by_batch[-1].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message="terminal timeout",
    )


async def test_scheduler_stops_after_conclusive_failure_outcome_without_running_later_artifacts(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    successful_task = _task("baseline carry success")
    failed_task = _task("baseline carry failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(successful_task, failed_task),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )
    batch_id = uuid4()
    first_artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=8, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    successful_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=successful_task,
    )
    failed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=first_artifact,
        task=failed_task,
        error=EvaluationError(code="sandbox_invocation_failed", message="shared sandbox failure"),
    )
    later_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=second_artifact,
        task=successful_task,
    )

    class _FailureOutcomeRunner:
        def __init__(self) -> None:
            self.seen_artifacts: list[UUID] = []

        async def evaluate_artifact_with_state(
            self,
            *,
            artifact: ScriptArtifactSpec,
            **_kwargs,
        ) -> ArtifactEvaluationOutcome:
            self.seen_artifacts.append(artifact.artifact_id)
            if artifact.artifact_id == first_artifact.artifact_id:
                raise ValidatorBatchFailedError(
                    error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                    message="shared sandbox failure",
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code="sandbox_invocation_failed",
                        error_message="shared sandbox failure",
                        occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                        artifact_id=artifact.artifact_id,
                        uid=artifact.uid,
                        exception_type="SandboxInvocationError",
                    ),
                    completed_submissions=(successful_submission, failed_submission),
                    remaining_tasks=(),
                )
            return ArtifactEvaluationOutcome(
                submissions=(later_submission,),
            )

    runner = _FailureOutcomeRunner()
    scheduler._runner = runner  # type: ignore[assignment]

    with pytest.raises(ValidatorBatchFailedError, match="shared sandbox failure") as exc_info:
        await scheduler.run(
            batch_id=batch_id,
            requested_artifacts=(first_artifact, second_artifact),
    )

    exc = exc_info.value
    assert runner.seen_artifacts == [first_artifact.artifact_id]
    assert exc.completed_submissions == (successful_submission, failed_submission)
    assert exc.remaining_tasks == ()


async def test_scheduler_retries_sandbox_start_once_before_running_tasks(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class FlakySandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            if len(self.starts) == 1:
                raise RuntimeError("sandbox cold start failed")
            return SandboxDeployment(client=object())

    sandbox_manager = FlakySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text="answer startup"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    option_calls: list[dict[str, object]] = []

    def sandbox_options_factory(artifact: ScriptArtifactSpec) -> dict[str, object]:
        option = {
            "call": len(option_calls),
            "uid": artifact.uid,
            "artifact_id": artifact.artifact_id,
        }
        option_calls.append(option)
        return option

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=sandbox_options_factory,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert len(sandbox_manager.starts) == 2
    assert sandbox_manager.starts[0] == {
        "call": 0,
        "uid": artifacts[0].uid,
        "artifact_id": artifacts[0].artifact_id,
    }
    assert sandbox_manager.starts[1] == {
        "call": 1,
        "uid": artifacts[0].uid,
        "artifact_id": artifacts[0].artifact_id,
    }
    assert result.completed_run_count == 1
    assert evaluation_records.records_by_batch[0].score == pytest.approx(0.75)


async def test_scheduler_does_not_synthesize_zero_score_rows_for_conclusive_setup_failures(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = (ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="a", size_bytes=0),)
    with pytest.raises(ValidatorBatchFailedError, match="sandbox cold start failed") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.SANDBOX_START_FAILED
    assert exc_info.value.completed_submissions == ()
    assert exc_info.value.remaining_tasks == (task,)
    assert len(sandbox_manager.starts) == 2
    assert evaluation_records.records_by_batch == []


async def test_scheduler_fails_batch_immediately_on_conclusive_sandbox_start_failure(
    blocking_executor: ThreadPoolExecutor,
    tmp_path: Path,
) -> None:
    task = _task("startup failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)

    class AlwaysFailingSandboxManager(DummySandboxManager):
        def start(self, options: object | None = None) -> SandboxDeployment:
            self.starts.append(options)
            raise RuntimeError("sandbox cold start failed")

    sandbox_manager = AlwaysFailingSandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    diagnostics_dir = tmp_path / "sandbox-diagnostics"
    diagnostics_dir.mkdir()
    (diagnostics_dir / "sandbox-options.json").write_text(
        """
        {
          "image": "harnyx/sandbox:test",
          "container_name": "sandbox-uid-7",
          "pull_policy": "always",
          "env": {"MINER_UID": "7", "SECRET_TOKEN": "<redacted>"}
        }
        """,
        encoding="utf-8",
    )
    (diagnostics_dir / "error.txt").write_text(
        "RuntimeError: sandbox failed without secret value\n",
        encoding="utf-8",
    )
    (diagnostics_dir / "docker-inspect.json").write_text(
        """
        [
          {
            "Id": "container-123",
            "Config": {"Env": ["SECRET_TOKEN=should-not-leak"]},
            "State": {
              "Status": "exited",
              "ExitCode": 255,
              "OOMKilled": false,
              "Error": "process exited"
            }
          }
        ]
        """,
        encoding="utf-8",
    )
    (diagnostics_dir / "docker-logs.txt").write_text(
        "safe log line\nSECRET_TOKEN=should-not-leak was already redacted upstream\n",
        encoding="utf-8",
    )
    (diagnostics_dir / "docker-pull-result.json").write_text(
        '{"returncode": 1, "stdout": "pull out", "stderr": "pull err"}',
        encoding="utf-8",
    )
    (diagnostics_dir / "docker-run-result.json").write_text(
        '{"returncode": 125, "stdout": "run out", "stderr": "run err"}',
        encoding="utf-8",
    )

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda artifact: SandboxOptions(
            image="harnyx/sandbox:test",
            container_name=f"sandbox-{artifact.uid}",
            pull_policy="always",
            env={"MINER_UID": str(artifact.uid), "SECRET_TOKEN": "should-not-leak"},
            failure_diagnostics_dir=str(diagnostics_dir),
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0) for uid in (3, 5, 7)
    )
    with pytest.raises(ValidatorBatchFailedError, match="sandbox cold start failed") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.SANDBOX_START_FAILED
    assert exc_info.value.failure_detail.error_code == "sandbox_start_failed"
    diagnostics = exc_info.value.failure_detail.sandbox_diagnostics
    assert diagnostics is not None
    assert diagnostics.image == "harnyx/sandbox:test"
    assert diagnostics.container_name == "sandbox-3"
    assert diagnostics.container_id == "container-123"
    assert diagnostics.status == "exited"
    assert diagnostics.exit_code == 255
    assert diagnostics.oom_killed is False
    assert diagnostics.state_error == "process exited"
    assert diagnostics.error_text == "RuntimeError: sandbox failed without secret value\n"
    assert diagnostics.docker_logs_tail is not None
    assert "safe log line" in diagnostics.docker_logs_tail
    assert diagnostics.pull_returncode == 1
    assert diagnostics.pull_stderr_tail == "pull err"
    assert diagnostics.run_returncode == 125
    assert diagnostics.run_stderr_tail == "run err"
    serialized = str(diagnostics)
    assert "Config" not in serialized
    assert "should-not-leak" not in serialized
    assert exc_info.value.remaining_tasks == (task,)
    assert len(sandbox_manager.starts) == 2
    assert evaluation_records.records_by_batch == []


async def test_scheduler_fails_batch_immediately_on_conclusive_artifact_fetch_failure(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("fetch failure")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_fetch_failed",
                message="platform artifact fetch exhausted retries",
                exception_type="PlatformClientError",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0) for uid in (3, 5, 7)
    )

    with pytest.raises(
        ValidatorBatchFailedError,
        match="platform artifact fetch exhausted retries",
    ) as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.ARTIFACT_FETCH_FAILED
    assert exc_info.value.failure_detail.error_code == "artifact_fetch_failed"
    assert exc_info.value.remaining_tasks == (task,)
    assert len(sandbox_manager.starts) == 0
    assert evaluation_records.records_by_batch == []


async def test_scheduler_fails_batch_immediately_on_conclusive_artifact_hash_mismatch(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("hash mismatch")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="artifact_hash_mismatch",
                message="platform agent sha256 mismatch",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0) for uid in (3, 5, 7)
    )

    with pytest.raises(ValidatorBatchFailedError, match="platform agent sha256 mismatch") as exc_info:
        await scheduler.run(batch_id=uuid4(), requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.ARTIFACT_HASH_MISMATCH
    assert exc_info.value.failure_detail.error_code == "artifact_hash_mismatch"
    assert exc_info.value.remaining_tasks == (task,)
    assert len(sandbox_manager.starts) == 0
    assert evaluation_records.records_by_batch == []


async def test_scheduler_records_script_validation_failures_for_all_tasks_without_failing_delivery(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("size mismatch"), _task("later mismatch"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: _client,
        sandbox_options_factory=lambda _artifact: (_ for _ in ()).throw(
            ArtifactPreparationError(
                error_code="script_validation_failed",
                message="platform artifact script invalid",
            )
        ),
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="hash-3", size_bytes=0)
    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(artifact,))
    assert len(sandbox_manager.starts) == 0
    assert result.completed_run_count == len(tasks)
    assert len(evaluation_records.records_by_batch) == len(tasks)
    for submission in evaluation_records.records_by_batch:
        assert submission.score == 0.0
        assert submission.run.details.error == EvaluationError(
            code="script_validation_failed",
            message="platform artifact script invalid",
        )


async def test_scheduler_fails_batch_on_first_conclusive_evaluation_failure(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    task = _task("artifact breaker")
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    batch_id = uuid4()
    artifacts = tuple(
        ScriptArtifactSpec(uid=uid, artifact_id=uuid4(), content_hash=f"hash-{uid}", size_bytes=0) for uid in (3, 5, 7)
    )

    scheduler = EvaluationScheduler(
        tasks=(task,),
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda _client: object(),
        sandbox_options_factory=lambda artifact: {"uid": artifact.uid, "artifact_id": artifact.artifact_id},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_parallelism=1,
        ),
        progress=DummyProgressRecorder(),
    )

    class ArtifactFailingRunner:
        def __init__(self) -> None:
            self.failed_artifact_ids: list[UUID] = []

        async def evaluate_artifact_with_state(self, *, artifact: ScriptArtifactSpec, tasks, **_kwargs):
            self.failed_artifact_ids.append(artifact.artifact_id)
            raise ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                message="shared sandbox failure",
                failure_detail=ValidatorBatchFailureDetail(
                    error_code="sandbox_invocation_failed",
                    error_message="shared sandbox failure",
                    occurred_at=datetime(2025, 10, 27, tzinfo=UTC),
                    artifact_id=artifact.artifact_id,
                    uid=artifact.uid,
                    exception_type="SandboxInvocationError",
                ),
                completed_submissions=(),
                remaining_tasks=tuple(tasks),
            )

    failing_runner = ArtifactFailingRunner()
    scheduler._runner = failing_runner

    with pytest.raises(ValidatorBatchFailedError, match="shared sandbox failure") as exc_info:
        await scheduler.run(batch_id=batch_id, requested_artifacts=artifacts)
    assert exc_info.value.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc_info.value.failure_detail.error_code == "sandbox_invocation_failed"
    assert len(sandbox_manager.starts) == 1
    assert len(sandbox_manager.stops) == 1
    assert failing_runner.failed_artifact_ids == [artifacts[0].artifact_id]


async def test_evaluation_runner_issues_session_with_task_budget(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    evaluation_records = DummyEvaluationRecordStore()
    receipt_log = DummyReceiptLog()
    scheduler = EvaluationScheduler(
        tasks=(),
        subtensor_client=subtensor,
        sandbox_manager=DummySandboxManager(),
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=lambda client: client,
        sandbox_options_factory=lambda artifact: artifact,
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=DummyProgressRecorder(),
    )

    task = _task("budgeted", budget_usd=0.123)
    issued = scheduler._runner._issue_session(
        batch_id=uuid4(),
        uid=3,
        task=task,
    )
    assert issued.session.task_id == task.task_id
    assert issued.session.budget_usd == pytest.approx(0.123)


async def test_scheduler_runs_only_remaining_pairs(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    progress = DummyProgressRecorder(recorded=frozenset({(artifact.artifact_id, tasks[0].task_id)}))
    recorded_requests: list[tuple[int, MinerTask]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.task))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(artifact,))
    assert len(sandbox_manager.starts) == 1
    assert [(uid, task.task_id) for uid, task in recorded_requests] == [(artifact.uid, tasks[1].task_id)]
    assert result.completed_run_count == 1
    assert evaluation_records.records_by_batch[0].run.task_id == tasks[1].task_id


async def test_scheduler_skips_artifact_when_all_pairs_are_already_recorded(
    blocking_executor: ThreadPoolExecutor,
) -> None:
    tasks = (_task("one"), _task("two"))
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    sandbox_manager = DummySandboxManager()
    evaluation_records = DummyEvaluationRecordStore()
    session_manager = SessionManager(InMemorySessionRegistry(), InMemoryTokenRegistry())
    receipt_log = DummyReceiptLog()
    first_artifact = ScriptArtifactSpec(uid=3, artifact_id=uuid4(), content_hash="a", size_bytes=0)
    second_artifact = ScriptArtifactSpec(uid=5, artifact_id=uuid4(), content_hash="b", size_bytes=0)
    progress = DummyProgressRecorder(recorded=frozenset((first_artifact.artifact_id, task.task_id) for task in tasks))
    recorded_requests: list[tuple[int, UUID]] = []

    def orchestrator_factory(_client: object):
        class StubOrchestrator:
            async def evaluate(self, request):
                recorded_requests.append((request.uid, request.artifact_id))
                details = EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=0.75,
                        total_score=0.75,
                        scoring_version="v1",
                    ),
                    total_tool_usage=ToolUsageSummary.zero(),
                )
                run = MinerTaskRun(
                    session_id=request.session_id,
                    uid=request.uid,
                    artifact_id=request.artifact_id,
                    task_id=request.task.task_id,
                    response=Response(text=f"answer {request.task.query.text}"),
                    details=details,
                    completed_at=datetime(2025, 10, 27, tzinfo=UTC),
                )
                return TaskRunOutcome(run=run, usage=TokenUsageSummary.empty())

        return StubOrchestrator()

    scheduler = EvaluationScheduler(
        tasks=tasks,
        subtensor_client=subtensor,
        sandbox_manager=sandbox_manager,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        blocking_executor=blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=lambda current_artifact: {"uid": current_artifact.uid},
        clock=lambda: datetime(2025, 10, 27, tzinfo=UTC),
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
        ),
        progress=progress,
    )

    result = await scheduler.run(batch_id=uuid4(), requested_artifacts=(first_artifact, second_artifact))
    assert len(sandbox_manager.starts) == 1
    assert all(uid == second_artifact.uid for uid, _artifact_id in recorded_requests)
    assert result.completed_run_count == len(tasks)
