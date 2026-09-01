from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest

import harnyx_validator.application.services.evaluation_runner as evaluation_runner_module
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    EvaluationTrace,
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.domain.tool_usage import SearchToolUsageSummary, ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskExecution,
    PlatformOwnedTaskResult,
    SandboxFailureDiagnostics,
    ScriptArtifactSpec,
    TaskExecutionOutcome,
    TaskRunOutcome,
    TokenUsageSummary,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, UsageSummarizer
from harnyx_validator.application.invoke_entrypoint import (
    MinerResponseValidationError,
    SandboxInvocationError,
)
from harnyx_validator.application.platform_tool_proxy import PlatformToolProxyScopeRegistry
from harnyx_validator.application.ports.platform import PlatformToolProxyControlError
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    TERMINAL_TIMEOUT_ERROR_MESSAGE,
    EvaluationRunner,
    UnexpectedArtifactExecutionError,
    ValidatorBatchFailedError,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.scoring.vertex_embedding import VertexEmbeddingRetryExhaustedError
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token
_FAILED_ASSIGNMENT_TOKEN = "failed-assignment-token"  # noqa: S105 - fixed test-only assignment token
_COMPLETED_ASSIGNMENT_TOKEN = "completed-assignment-token"  # noqa: S105 - fixed test-only assignment token


class _AssignedWork:
    def __init__(
        self,
        queued: tuple[MinerTaskWorkAssignment, ...] = (),
        result_queue: asyncio.Queue[PlatformOwnedTaskResult] | None = None,
    ) -> None:
        self.queue: asyncio.Queue[MinerTaskWorkAssignment] = asyncio.Queue()
        for assignment in queued:
            self.queue.put_nowait(assignment)
        self.result_queue = result_queue
        self.initial_claims: list[MinerTaskWorkAssignment] = []
        self.dispatch_claims: list[MinerTaskWorkAssignment] = []
        self.started: list[tuple[MinerTaskWorkAssignment, UUID]] = []
        self.failed_before_start: list[tuple[MinerTaskWorkAssignment, PlatformOwnedTaskResult]] = []

    async def take_for_startup(self) -> MinerTaskWorkAssignment:
        return await self.queue.get()

    def take_nowait_for_startup(self) -> MinerTaskWorkAssignment:
        return self.queue.get_nowait()

    def drain_for_setup_failure(self) -> tuple[MinerTaskWorkAssignment, ...]:
        drained: list[MinerTaskWorkAssignment] = []
        while True:
            try:
                drained.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return tuple(drained)

    def mark_dispatch_ready(self) -> None:
        return None

    def claim_initial_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> _ClaimedAssignedTaskFake:
        self.initial_claims.append(assignment)
        return _ClaimedAssignedTaskFake(self, assignment)

    async def claim_for_dispatch(self) -> _ClaimedAssignedTaskFake:
        assignment = await self.queue.get()
        self.dispatch_claims.append(assignment)
        return _ClaimedAssignedTaskFake(self, assignment)

    def claim_nowait_for_dispatch(self) -> _ClaimedAssignedTaskFake:
        assignment = self.queue.get_nowait()
        self.dispatch_claims.append(assignment)
        return _ClaimedAssignedTaskFake(self, assignment)

    def _mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> None:
        self.started.append((assignment, validator_session_id))

    def _fail_before_start(self, assignment: MinerTaskWorkAssignment, result: PlatformOwnedTaskResult) -> None:
        self.failed_before_start.append((assignment, result))
        if self.result_queue is not None:
            self.result_queue.put_nowait(result)


class _ClaimedAssignedTaskFake:
    def __init__(self, owner: _AssignedWork, assignment: MinerTaskWorkAssignment) -> None:
        self._owner = owner
        self._assignment = assignment

    @property
    def assignment(self) -> MinerTaskWorkAssignment:
        return self._assignment

    def mark_started(self, validator_session_id: UUID, **_: object) -> None:
        self._owner._mark_started(self._assignment, validator_session_id)

    def fail_before_start(self, result: PlatformOwnedTaskResult) -> None:
        self._owner._fail_before_start(self._assignment, result)


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


class _ClockSequence:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)
        self._last_value = values[-1] if values else None

    def __call__(self) -> datetime:
        if not self._values:
            if self._last_value is None:
                raise AssertionError("clock sequence exhausted")
            return self._last_value
        self._last_value = self._values.pop(0)
        return self._last_value


class _RecordingEvaluationStore:
    def __init__(self) -> None:
        self.records: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records.append(result)


class _FailOnNthRecordEvaluationStore(_RecordingEvaluationStore):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def record(self, result: MinerTaskRunSubmission) -> None:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError("evaluation record write failed")
        super().record(result)


class _FailingSubmissionProgress(FileBackedRunProgress):
    def record(self, result: MinerTaskRunSubmission) -> None:
        raise RuntimeError("submission progress write failed")


class _FailingTerminatedAttemptProgress(FileBackedRunProgress):
    def record_terminated_attempt(self, attempt: MinerTaskAttemptAuditRecord) -> None:
        raise RuntimeError("terminated attempt progress write failed")


class _ExplodingValidatorInfoSubtensorClient(FakeSubtensorClient):
    def validator_info(self) -> ValidatorNodeInfo:
        raise AssertionError("assigned task result construction must not read validator UID")


def _assigned_task_test_context(
    tmp_path: Path,
    *,
    subtensor: FakeSubtensorClient | None = None,
    progress: FileBackedRunProgress | None = None,
    evaluation_records: _RecordingEvaluationStore | None = None,
    clock: _ClockSequence | None = None,
) -> tuple[
    EvaluationRunner,
    UUID,
    ScriptArtifactSpec,
    MinerTask,
    FakeSessionRegistry,
    FakeReceiptLog,
    FileBackedRunProgress,
    _RecordingEvaluationStore,
]:
    subtensor = FakeSubtensorClient() if subtensor is None else subtensor
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path) if progress is None else progress
    evaluation_records = _RecordingEvaluationStore() if evaluation_records is None else evaluation_records
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_records,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=clock or _ClockSequence(datetime(2025, 10, 17, 12, 0, tzinfo=UTC)),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="assigned local recording"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    return runner, uuid4(), artifact, task, session_registry, receipt_log, progress, evaluation_records


def _assert_local_recording_warning(
    caplog: pytest.LogCaptureFixture,
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    result: PlatformOwnedTaskResult,
    exception_type: str,
    local_write_target: str,
) -> None:
    matches = [
        record
        for record in caplog.records
        if getattr(record, "data", {}).get("local_write_target") == local_write_target
    ]
    assert len(matches) == 1
    data = matches[0].data
    assert data["batch_id"] == str(batch_id)
    assert data["artifact_id"] == str(artifact.artifact_id)
    assert data["task_id"] == str(task.task_id)
    assert data["validator_session_id"] == str(result.terminal_attempt.validator_session_id)
    assert data["attempt_number"] == result.attempt_number
    assert data["max_attempts"] == result.terminal_attempt.max_attempts
    assert data["error_code"] == result.terminal_attempt.error_code
    assert data["local_write_target"] == local_write_target
    assert data["exception_type"] == exception_type


def _record_receipt(
    receipt_log: FakeReceiptLog,
    *,
    session_id,
    uid: int,
    receipt_id: str,
    issued_at: datetime,
    cost_usd: float,
    tool: str = "search_web",
    outcome: ToolCallOutcome = ToolCallOutcome.OK,
    response_payload: dict[str, object] | None = None,
    execution: ToolExecutionFacts | None = None,
    request_payload: dict[str, object] | None = None,
    active_attempt: int | None = 1,
    extra: dict[str, str] | None = None,
) -> None:
    receipt_extra = {} if extra is None else dict(extra)
    if active_attempt is not None:
        receipt_extra["session_active_attempt"] = str(active_attempt)
    receipt_log.record(
        ToolCall(
            receipt_id=receipt_id,
            session_id=session_id,
            uid=uid,
            tool=tool,
            issued_at=issued_at,
            outcome=outcome,
            details=ToolCallDetails(
                request_hash=f"{receipt_id}-req",
                request_payload=request_payload,
                response_hash=f"{receipt_id}-res",
                cost_usd=cost_usd,
                response_payload=response_payload,
                execution=execution,
                extra=receipt_extra or None,
            ),
        )
    )


def _search_usage(receipt_log: FakeReceiptLog, session_id) -> ToolUsageSummary:
    receipts = tuple(receipt_log.for_session(session_id))
    total_cost = sum(float(receipt.details.cost_usd or 0.0) for receipt in receipts)
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(
            call_count=len(receipts),
            cost=round(total_cost, 6),
        ),
        search_tool_cost=total_cost,
        llm=ToolUsageSummary.zero().llm,
        llm_cost=0.0,
    )


def _judge_usage(*, actual_cost_usd: float = 0.031) -> JudgeUsageSummary:
    model_usage = JudgeModelUsage(
        provider="chutes",
        model="google/gemma-4-31B-turbo-TEE",
        call_count=1,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        reasoning_tokens=7,
        actual_cost_usd=actual_cost_usd,
        actual_cost_source="provider_actual",
        actual_cost_provider="chutes",
        actual_cost_evidence="test-cost-id",
    )
    return JudgeUsageSummary(
        call_count=1,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        reasoning_tokens=7,
        actual_cost_usd=actual_cost_usd,
        models=(model_usage,),
    )


def _successful_outcome(
    request: MinerTaskRunRequest,
    *,
    score: float = 0.75,
) -> TaskRunOutcome:
    return TaskRunOutcome(
        run=MinerTaskRun(
            session_id=request.session_id,
            uid=request.uid,
            artifact_id=request.artifact_id,
            task_id=request.task.task_id,
            response=Response(text=f"answer {request.task.query.text}"),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=score,
                    total_score=score,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ),
        usage=TokenUsageSummary.empty(),
    )


def _successful_execution(
    request: MinerTaskRunRequest,
    *,
    completed_at: datetime | None = None,
) -> TaskExecutionOutcome:
    return TaskExecutionOutcome(
        batch_id=request.batch_id,
        artifact_id=request.artifact_id,
        task=request.task,
        session=Session(
            session_id=request.session_id,
            uid=request.uid,
            task_id=request.task.task_id,
            issued_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            expires_at=datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            budget_usd=request.task.budget_usd,
            usage=SessionUsage(),
            status=SessionStatus.ACTIVE,
        ),
        uid=request.uid,
        response=Response(text=f"execution {request.task.query.text}"),
        execution_completed_at=completed_at or datetime(2025, 10, 17, 12, 3, tzinfo=UTC),
        usage=TokenUsageSummary.empty(),
        total_tool_usage=ToolUsageSummary.zero(),
        trace=EvaluationTrace(entrypoint_invocation_ms=12.0, scoring_ms=0.0),
    )


def _submission_for_task(
    *,
    batch_id,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    error: EvaluationError | None = None,
) -> MinerTaskRunSubmission:
    issued_at = datetime(2025, 10, 17, 12, 0, tzinfo=UTC)
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
        run=run,
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def _platform_owned_task_result(
    assignment: MinerTaskWorkAssignment,
    *,
    terminal_effect: MinerTaskAttemptTerminalEffect,
) -> PlatformOwnedTaskResult:
    now = datetime(2025, 10, 17, 12, 0, tzinfo=UTC)
    failed = terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    return PlatformOwnedTaskResult(
        batch_id=assignment.batch_id,
        artifact_id=assignment.artifact.artifact_id,
        task_id=assignment.task.task_id,
        attempt_number=assignment.attempt_number,
        result=None,
        terminal_attempt=MinerTaskAttemptAuditRecord(
            validator_session_id=uuid4(),
            batch_id=assignment.batch_id,
            artifact_id=assignment.artifact.artifact_id,
            task_id=assignment.task.task_id,
            attempt_number=assignment.attempt_number,
            uid=assignment.artifact.uid,
            miner_hotkey_ss58="miner-hotkey",
            started_at=now,
            finished_at=now,
            status=MinerTaskAttemptStatus.FAILED if failed else MinerTaskAttemptStatus.SUCCEEDED,
            error_code="artifact_setup_failed" if failed else None,
            error_summary_code="artifact_setup_failed" if failed else None,
            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
            terminal_effect=terminal_effect,
            max_attempts=assignment.max_attempts,
            execution_log=(),
        ),
    )


async def _run_assigned_task_queue_until_results(
    runner: EvaluationRunner,
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    initial_assignments: tuple[MinerTaskWorkAssignment, ...],
    assigned_work: _AssignedWork,
    result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    orchestrator: TaskRunOrchestrator,
    expected_result_count: int,
) -> tuple[PlatformOwnedTaskResult, ...]:
    close_requested = asyncio.Event()
    execution = asyncio.create_task(
        runner.evaluate_assigned_task_queue(
            batch_id=batch_id,
            artifact=artifact,
            initial_assignments=initial_assignments,
            assigned_work=assigned_work,
            close_requested=close_requested,
            result_queue=result_queue,
            orchestrator=orchestrator,
        )
    )
    results: list[PlatformOwnedTaskResult] = []
    try:
        for _ in range(expected_result_count):
            results.append(await asyncio.wait_for(result_queue.get(), timeout=1.0))
    finally:
        close_requested.set()
    await asyncio.wait_for(execution, timeout=1.0)
    return tuple(results)


async def test_evaluate_assigned_task_queue_success_queues_execution_before_scoring(tmp_path: Path) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    completed_at = datetime(2025, 10, 17, 12, 3, tzinfo=UTC)

    class _ExecutionOnlyOrchestrator:
        async def execute(self, request: MinerTaskRunRequest, *, phase_recorder=None) -> TaskExecutionOutcome:
            _ = phase_recorder
            assert request.execution_time_limit_seconds == 300.0
            return TaskExecutionOutcome(
                batch_id=request.batch_id,
                artifact_id=request.artifact_id,
                task=request.task,
                session=Session(
                    session_id=request.session_id,
                    uid=request.uid,
                    task_id=request.task.task_id,
                    issued_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
                    expires_at=datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
                    budget_usd=request.task.budget_usd,
                    usage=SessionUsage(),
                    status=SessionStatus.ACTIVE,
                ),
                uid=request.uid,
                response=Response(text="execution response"),
                execution_completed_at=completed_at,
                usage=TokenUsageSummary.empty(),
                total_tool_usage=ToolUsageSummary.zero(),
                trace=EvaluationTrace(entrypoint_invocation_ms=12.0, scoring_ms=0.0),
            )

    result_queue: asyncio.Queue[PlatformOwnedTaskExecution | PlatformOwnedTaskResult] = asyncio.Queue()
    close_requested = asyncio.Event()
    execution_task = asyncio.create_task(
        runner.evaluate_assigned_task_queue(
            batch_id=batch_id,
            artifact=artifact,
            initial_assignments=(assignment,),
            assigned_work=_AssignedWork(),
            close_requested=close_requested,
            result_queue=result_queue,
            orchestrator=cast(TaskRunOrchestrator, _ExecutionOnlyOrchestrator()),
        )
    )
    try:
        queued = await asyncio.wait_for(result_queue.get(), timeout=1.0)
    finally:
        close_requested.set()
    await asyncio.wait_for(execution_task, timeout=1.0)

    assert isinstance(queued, PlatformOwnedTaskExecution)
    assert queued.batch_id == batch_id
    assert queued.artifact == artifact
    assert queued.task == task
    assert queued.attempt_number == 1
    assert queued.max_attempts == 2
    assert queued.response == Response(text="execution response")
    assert queued.execution_completed_at == completed_at
    assert queued.session.status is SessionStatus.COMPLETED


async def test_evaluate_assigned_task_queue_final_invocation_exception_becomes_attempt_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=2,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    async def evaluate_issued_assigned_task(**_kwargs):
        raise RuntimeError("miner invocation raised before task result")

    monkeypatch.setattr(runner, "_evaluate_issued_assigned_task", evaluate_issued_assigned_task)

    assigned_work = _AssignedWork()
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, object()),
        expected_result_count=1,
    )

    (result,) = results
    assert result.result is None
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.ATTEMPT_FAILURE
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.failure_owner == "RuntimeError"
    assert session_registry.get(result.terminal_attempt.validator_session_id) is None
    assert result_queue.empty()


async def test_evaluation_runner_assigned_task_retryable_delivery_disqualifying_validator_failure_does_not_carry_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(tmp_path)
    failure_detail = ValidatorBatchFailureDetail(
        error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
        error_message="sandbox start failed",
        occurred_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        uid=artifact.uid,
        sandbox_diagnostics=SandboxFailureDiagnostics(exit_code=255),
    )

    async def evaluate_task_attempt(**_kwargs) -> evaluation_runner_module.TaskAttemptDecision:
        return evaluation_runner_module._validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
                message="sandbox start failed",
                failure_detail=failure_detail,
            )
        )

    monkeypatch.setattr(runner, "_evaluate_task_attempt", evaluate_task_attempt)

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, object()),
    )

    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == str(MinerTaskErrorCode.SANDBOX_START_FAILED)
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    assert result.terminal_attempt.delivery_failure_detail is None


async def test_evaluate_assigned_task_queue_final_delivery_failure_closes_group_after_started_sibling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, started_task, _, _, _, _ = _assigned_task_test_context(tmp_path)
    delivery_failure_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="delivery failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    started_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=started_task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    delivery_failure_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=delivery_failure_task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_FAILED_ASSIGNMENT_TOKEN,
    )
    failure_detail = ValidatorBatchFailureDetail(
        error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
        error_message="sandbox start failed",
        occurred_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        artifact_id=artifact.artifact_id,
        task_id=delivery_failure_task.task_id,
        uid=artifact.uid,
        sandbox_diagnostics=SandboxFailureDiagnostics(exit_code=255),
    )
    started_entered = asyncio.Event()
    release_started = asyncio.Event()
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    close_requested = asyncio.Event()

    async def evaluate_task_attempt(**kwargs) -> evaluation_runner_module.TaskAttemptDecision:
        if kwargs["task"].task_id == delivery_failure_task.task_id:
            return evaluation_runner_module._validator_batch_failure_decision(
                ValidatorBatchFailedError(
                    error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
                    message="sandbox start failed",
                    failure_detail=failure_detail,
                )
            )
        started_entered.set()
        await release_started.wait()
        return evaluation_runner_module._submission_decision(
            _submission_for_task(
                batch_id=batch_id,
                artifact=artifact,
                task=started_task,
            )
        )

    monkeypatch.setattr(runner, "_evaluate_task_attempt", evaluate_task_attempt)
    execution = asyncio.create_task(
        runner.evaluate_assigned_task_queue(
            batch_id=batch_id,
            artifact=artifact,
            initial_assignments=(started_assignment, delivery_failure_assignment),
            assigned_work=_AssignedWork(result_queue=result_queue),
            close_requested=close_requested,
            result_queue=result_queue,
            orchestrator=cast(TaskRunOrchestrator, object()),
        )
    )

    try:
        await asyncio.wait_for(started_entered.wait(), timeout=1.0)
        delivery_result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
        assert delivery_result.task_id == delivery_failure_task.task_id
        assert delivery_result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
        assert delivery_result.terminal_attempt.delivery_failure_detail == failure_detail
        assert not execution.done()

        release_started.set()
        started_result = await asyncio.wait_for(result_queue.get(), timeout=1.0)
        assert started_result.task_id == started_task.task_id
        assert started_result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    finally:
        close_requested.set()
        release_started.set()

    await asyncio.wait_for(execution, timeout=1.0)


async def test_evaluation_runner_assigned_task_final_validator_failure_becomes_attempt_failure(
    tmp_path: Path,
) -> None:
    """Prevent a task-owned validator failure from becoming delivery-wide failure."""

    runner, batch_id, artifact, task, session_registry, receipt_log, _, _ = _assigned_task_test_context(tmp_path)

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(
            TaskRunOrchestrator,
            _GenericFailureOrchestrator(sessions=session_registry, receipt_log=receipt_log),
        ),
    )

    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.ATTEMPT_FAILURE
    assert result.terminal_attempt.delivery_failure_detail is None


async def test_evaluation_runner_assigned_proxy_grant_failure_closes_delivery_with_exact_code(
    tmp_path: Path,
) -> None:
    """Prevent a known proxy-control failure from becoming an unexpected attempt failure."""

    runner, batch_id, artifact, task, session_registry, receipt_log, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=_AssignedWork(),
        result_queue=result_queue,
        orchestrator=cast(
            TaskRunOrchestrator,
            _PlatformToolProxyReceiptOrchestrator(
                sessions=session_registry,
                receipt_log=receipt_log,
                error_code="platform_tool_proxy_grant_failed",
                caught_by_miner=True,
            ),
        ),
        expected_result_count=1,
    )

    (result,) = results
    attempt = result.terminal_attempt
    assert result.result is None
    assert attempt.error_code == "platform_tool_proxy_grant_failed"
    assert attempt.error_summary_code == "platform_tool_proxy_grant_failed"
    assert attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
    assert attempt.delivery_failure_detail is not None
    assert attempt.delivery_failure_detail.error_code == "platform_tool_proxy_grant_failed"
    assert attempt.diagnostics is not None
    assert attempt.diagnostics.failure_owner == "platform_tool_proxy_grant_failed"


async def test_evaluate_assigned_task_queue_invocation_timeout_records_phase_owned_diagnostics(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    class _TimeoutOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest, *, phase_recorder=None) -> TaskRunOutcome:
            _ = request
            if phase_recorder is not None:
                phase_recorder.mark("entrypoint_invocation")
            raise _sandbox_invocation_error(
                "sandbox worker timed out",
                status_code=504,
                detail_exception="TimeoutError",
            )

    assigned_work = _AssignedWork()
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _TimeoutOrchestrator()),
        expected_result_count=1,
    )

    (result,) = results
    assert result.result is None
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    assert result.terminal_attempt.error_code == str(MinerTaskErrorCode.TIMEOUT_MINER_OWNED)
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.phase == "entrypoint_invocation"
    assert result.terminal_attempt.diagnostics.timeout_owner == "sandbox_entrypoint"
    assert result.terminal_attempt.diagnostics.failure_owner == "SandboxInvocationError"
    assert session_registry.get(result.terminal_attempt.validator_session_id) is None
    assert result_queue.empty()


async def test_evaluate_assigned_task_queue_platform_tool_proxy_timeout_records_proxy_owner(
    tmp_path: Path,
) -> None:
    """Keep platform-tool-proxy execution timeouts distinct from sandbox invocation timeouts."""

    runner, batch_id, artifact, task, session_registry, receipt_log, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    class _TimeoutOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest, *, phase_recorder=None) -> TaskRunOutcome:
            session = session_registry.get(request.session_id)
            assert session is not None
            if phase_recorder is not None:
                phase_recorder.mark("entrypoint_invocation")
                phase_recorder.mark("platform_tool_proxy_execute")
            _record_platform_tool_proxy_timeout_receipt(
                receipt_log,
                request=request,
                active_attempt=session.active_attempt,
                receipt_id="platform-tool-proxy-timeout-assigned-1",
            )
            raise _provider_tool_failure_error()

    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=_AssignedWork(),
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _TimeoutOrchestrator()),
        expected_result_count=1,
    )

    (result,) = results
    assert result.result is None
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    assert result.terminal_attempt.error_code == "tool_timeout"
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.phase == "platform_tool_proxy_execute"
    assert result.terminal_attempt.diagnostics.timeout_owner == "platform_tool_proxy_execute"
    assert result.terminal_attempt.diagnostics.platform_tool_activity_observed is True


async def test_evaluate_assigned_task_queue_final_platform_tool_proxy_timeout_records_proxy_owner(
    tmp_path: Path,
) -> None:
    """Keep final platform-tool-proxy timeout diagnostics tied to the proxy execution phase."""

    runner, batch_id, artifact, task, session_registry, receipt_log, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=2,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    class _TimeoutOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest, *, phase_recorder=None) -> TaskRunOutcome:
            session = session_registry.get(request.session_id)
            assert session is not None
            if phase_recorder is not None:
                phase_recorder.mark("entrypoint_invocation")
                phase_recorder.mark("platform_tool_proxy_execute")
            _record_platform_tool_proxy_timeout_receipt(
                receipt_log,
                request=request,
                active_attempt=session.active_attempt,
                receipt_id="platform-tool-proxy-timeout-assigned-final",
            )
            raise _provider_tool_failure_error()

    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=_AssignedWork(),
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _TimeoutOrchestrator()),
        expected_result_count=1,
    )

    (result,) = results
    assert result.result is not None
    assert result.result.run.details.error is not None
    assert result.result.run.details.error.code == "timeout_miner_owned"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.phase == "platform_tool_proxy_execute"
    assert result.terminal_attempt.diagnostics.timeout_owner == "platform_tool_proxy_execute"
    assert result.terminal_attempt.diagnostics.platform_tool_activity_observed is True


async def test_evaluate_assigned_task_queue_drains_active_result_before_queue_start_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, completed_task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    blocking_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="blocking active"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    queued_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="queued start failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    completed_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=completed_task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_COMPLETED_ASSIGNMENT_TOKEN,
    )
    blocking_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=blocking_task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    queued_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=queued_task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_FAILED_ASSIGNMENT_TOKEN,
    )
    real_wait = asyncio.wait
    release_blocking = asyncio.Event()

    async def wait_after_ready_tick(wait_for, *, return_when=asyncio.ALL_COMPLETED):
        await asyncio.sleep(0)
        done = {task for task in wait_for if task.done()}
        if done:
            return done, set(wait_for) - done
        return await real_wait(wait_for, return_when=return_when)

    async def evaluate_issued_assigned_task(**kwargs):
        if kwargs["task"].task_id == completed_task.task_id:
            return _platform_owned_task_result(
                completed_assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
            )
        if kwargs["task"].task_id == blocking_task.task_id:
            await release_blocking.wait()
            return _platform_owned_task_result(
                blocking_assignment,
                terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
            )
        raise AssertionError("unexpected assignment execution")

    class _QueueStartFailureAssignedWork(_AssignedWork):
        def claim_nowait_for_dispatch(self) -> _ClaimedAssignedTaskFake:
            raise asyncio.QueueEmpty

        def _mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> None:
            if assignment is queued_assignment:
                self.started.append((assignment, validator_session_id))
                raise RuntimeError("assigned work start mark failed")
            return super()._mark_started(assignment, validator_session_id)

        def _fail_before_start(self, assignment: MinerTaskWorkAssignment, result: PlatformOwnedTaskResult) -> None:
            super()._fail_before_start(assignment, result)
            if assignment is queued_assignment:
                release_blocking.set()

    monkeypatch.setattr(evaluation_runner_module.asyncio, "wait", wait_after_ready_tick)
    monkeypatch.setattr(runner, "_evaluate_issued_assigned_task", evaluate_issued_assigned_task)

    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    assigned_work = _QueueStartFailureAssignedWork(
        queued=(queued_assignment,),
        result_queue=result_queue,
    )

    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(completed_assignment, blocking_assignment),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, object()),
        expected_result_count=3,
    )

    results_by_task_id = {result.task_id: result for result in results}
    assert results_by_task_id[completed_task.task_id].terminal_attempt.terminal_effect is (
        MinerTaskAttemptTerminalEffect.TASK_RESULT
    )
    assert results_by_task_id[blocking_task.task_id].terminal_attempt.terminal_effect is (
        MinerTaskAttemptTerminalEffect.TASK_RESULT
    )
    queued_failure = results_by_task_id[queued_task.task_id]
    assert queued_failure.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert queued_failure.terminal_attempt.terminal_effect is None
    assert queued_failure.terminal_attempt.diagnostics is not None
    assert queued_failure.terminal_attempt.diagnostics.failure_owner == "RuntimeError"
    assert result_queue.empty()
    blocking_session_id = next(
        session_id for assignment, session_id in assigned_work.started if assignment is blocking_assignment
    )
    queued_session_id = next(
        session_id for assignment, session_id in assigned_work.started if assignment is queued_assignment
    )
    assert session_registry.get(blocking_session_id) is None
    assert session_registry.get(queued_session_id) is None


async def test_evaluate_assigned_task_calls_session_start_callback_after_session_issuance(tmp_path: Path) -> None:
    runner, batch_id, artifact, task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    started_session_ids: list[UUID] = []
    evaluate_entered = asyncio.Event()
    release_evaluation = asyncio.Event()

    class _BlockingOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            evaluate_entered.set()
            await release_evaluation.wait()
            return _successful_outcome(request, score=1.0)

    execution = asyncio.create_task(
        runner.evaluate_assigned_task(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            attempt_number=1,
            max_attempts=2,
            assignment_token=_ASSIGNMENT_TOKEN,
            orchestrator=cast(TaskRunOrchestrator, _BlockingOrchestrator()),
            on_session_started=started_session_ids.append,
        )
    )

    try:
        await asyncio.wait_for(evaluate_entered.wait(), timeout=1.0)
        assert len(started_session_ids) == 1
        assert session_registry.get(started_session_ids[0]) is not None
    finally:
        release_evaluation.set()

    result = await asyncio.wait_for(execution, timeout=1.0)
    assert result.terminal_attempt.validator_session_id == started_session_ids[0]


async def test_evaluate_assigned_task_queue_cancels_active_children_on_queue_cancellation(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    assigned_work = _AssignedWork()
    evaluate_entered = asyncio.Event()
    release_evaluation = asyncio.Event()
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()

    class _BlockingOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            evaluate_entered.set()
            await release_evaluation.wait()
            return _successful_outcome(request, score=1.0)

    execution = asyncio.create_task(
        runner.evaluate_assigned_task_queue(
            batch_id=batch_id,
            artifact=artifact,
            initial_assignments=(assignment,),
            assigned_work=assigned_work,
            close_requested=asyncio.Event(),
            result_queue=result_queue,
            orchestrator=cast(TaskRunOrchestrator, _BlockingOrchestrator()),
        )
    )

    try:
        await asyncio.wait_for(evaluate_entered.wait(), timeout=1.0)
        session_id = assigned_work.started[0][1]
        assert session_registry.get(session_id) is not None
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(execution, timeout=1.0)
        assert session_registry.get(session_id) is None
    finally:
        release_evaluation.set()


async def test_evaluate_assigned_task_queue_mark_started_failure_cleans_issued_session(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, session_registry, _, progress, _ = _assigned_task_test_context(tmp_path)
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()

    class _RejectingAssignedWork(_AssignedWork):
        def __init__(self, result_queue: asyncio.Queue[PlatformOwnedTaskResult]) -> None:
            super().__init__(result_queue=result_queue)
            self.rejected: list[tuple[MinerTaskWorkAssignment, UUID]] = []

        def _mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> None:
            self.rejected.append((assignment, validator_session_id))
            raise RuntimeError("assigned work start mark failed")

    class _UnexpectedOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            raise AssertionError("execution should not spawn when mark_started fails")

    assigned_work = _RejectingAssignedWork(result_queue=result_queue)

    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _UnexpectedOrchestrator()),
        expected_result_count=1,
    )

    rejected_assignment, issued_session_id = assigned_work.rejected[0]
    (result,) = results
    assert rejected_assignment is assignment
    assert result.result is None
    assert result.terminal_attempt.validator_session_id == issued_session_id
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.phase == "session_start"
    assert result.terminal_attempt.diagnostics.failure_owner == "RuntimeError"
    assert assigned_work.started == []
    assert session_registry.get(issued_session_id) is None
    assert issued_session_id not in progress.session_context_by_id


async def test_evaluate_assigned_task_queue_initial_start_failure_cleans_prior_child(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, first_task, session_registry, _, _, _ = _assigned_task_test_context(tmp_path)
    second_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="second initial assignment"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    first_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=first_task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    second_assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=second_task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_FAILED_ASSIGNMENT_TOKEN,
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    release_first = asyncio.Event()

    class _RejectSecondAssignedWork(_AssignedWork):
        def __init__(self, result_queue: asyncio.Queue[PlatformOwnedTaskResult]) -> None:
            super().__init__(result_queue=result_queue)
            self.rejected: list[tuple[MinerTaskWorkAssignment, UUID]] = []

        def _mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> None:
            if assignment is second_assignment:
                self.rejected.append((assignment, validator_session_id))
                raise RuntimeError("assigned work start mark failed")
            return super()._mark_started(assignment, validator_session_id)

        def _fail_before_start(self, assignment: MinerTaskWorkAssignment, result: PlatformOwnedTaskResult) -> None:
            super()._fail_before_start(assignment, result)
            if assignment is second_assignment:
                release_first.set()

    class _BlockingOrchestrator:
        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            await release_first.wait()
            return _successful_outcome(request, score=1.0)

    assigned_work = _RejectSecondAssignedWork(result_queue=result_queue)

    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(first_assignment, second_assignment),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _BlockingOrchestrator()),
        expected_result_count=2,
    )

    first_session_id = assigned_work.started[0][1]
    second_session_id = assigned_work.rejected[0][1]
    results_by_task_id = {result.task_id: result for result in results}
    assert results_by_task_id[first_task.task_id].terminal_attempt.terminal_effect is (
        MinerTaskAttemptTerminalEffect.TASK_RESULT
    )
    second_result = results_by_task_id[second_task.task_id]
    assert second_result.terminal_attempt.validator_session_id == second_session_id
    assert second_result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert second_result.terminal_attempt.terminal_effect is None
    assert second_result.terminal_attempt.diagnostics is not None
    assert second_result.terminal_attempt.diagnostics.failure_owner == "RuntimeError"
    assert session_registry.get(first_session_id) is None
    assert session_registry.get(second_session_id) is None


async def test_evaluate_assigned_task_queue_internal_issue_registration_failure_cleans_session(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    token_registry = InMemoryTokenRegistry()
    progress = _progress(tmp_path)

    class _FailingPlatformToolProxyScopeRegistry(PlatformToolProxyScopeRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.issued_session_ids: list[UUID] = []

        def register_session(self, **kwargs: object) -> None:
            self.issued_session_ids.append(cast(UUID, kwargs["session_id"]))
            raise RuntimeError("proxy scope registration failed")

    platform_tool_proxy_scopes = _FailingPlatformToolProxyScopeRegistry()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=SessionManager(session_registry, token_registry),
        evaluation_records=_RecordingEvaluationStore(),
        receipt_log=FakeReceiptLog(),
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(datetime(2025, 10, 17, 12, 0, tzinfo=UTC)),
        progress=progress,
        platform_tool_proxy_scopes=platform_tool_proxy_scopes,
    )
    batch_id = uuid4()
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="internal issue registration failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    assignment = MinerTaskWorkAssignment(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    result_queue: asyncio.Queue[PlatformOwnedTaskResult] = asyncio.Queue()
    assigned_work = _AssignedWork(result_queue=result_queue)

    results = await _run_assigned_task_queue_until_results(
        runner,
        batch_id=batch_id,
        artifact=artifact,
        initial_assignments=(assignment,),
        assigned_work=assigned_work,
        result_queue=result_queue,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
        expected_result_count=1,
    )

    issued_session_id = platform_tool_proxy_scopes.issued_session_ids[0]
    (result,) = results
    assert result.result is None
    assert result.terminal_attempt.validator_session_id == issued_session_id
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    assert result.terminal_attempt.diagnostics is not None
    assert result.terminal_attempt.diagnostics.phase == "session_start"
    assert result.terminal_attempt.diagnostics.failure_owner == "RuntimeError"
    assert assigned_work.started == []
    assert session_registry.get(issued_session_id) is None
    assert token_registry.get_hash(issued_session_id) is None
    assert issued_session_id not in progress.session_context_by_id
    with pytest.raises(PlatformToolProxyControlError):
        platform_tool_proxy_scopes.require_session(issued_session_id)


async def test_evaluation_runner_assigned_success_does_not_read_validator_uid(tmp_path: Path) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        subtensor=_ExplodingValidatorInfoSubtensorClient(),
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT


async def test_evaluation_runner_assigned_task_failure_does_not_read_validator_uid(tmp_path: Path) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        subtensor=_ExplodingValidatorInfoSubtensorClient(),
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.error_code == "scoring_llm_retry_exhausted"
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT


async def test_evaluation_runner_assigned_pre_session_failure_returns_delivery_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(tmp_path)

    def fail_issue_session(**_kwargs):
        raise RuntimeError("session issue failed")

    monkeypatch.setattr(runner, "_issue_session", fail_issue_session)

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE


async def test_evaluation_runner_assigned_result_survives_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(tmp_path)

    def fail_cleanup(_session_id: UUID) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runner, "_cleanup_attempt_session", fail_cleanup)

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT


def test_usage_summarizer_falls_back_to_referenceable_result_count_when_search_cost_is_missing() -> None:
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(),
    )
    receipt = ToolCall(
        receipt_id="receipt-missing-cost",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="req",
            response_hash="res",
            cost_usd=None,
            results=(
                SearchToolResult(index=0, result_id="result-1", url="https://a.example", note="A"),
                SearchToolResult(index=1, result_id="result-2", url="https://b.example", note="B"),
            ),
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))
    assert total_tool_usage.search_tool.call_count == 1
    assert total_tool_usage.search_tool.cost == pytest.approx(0.0002)
    assert total_tool_usage.search_tool_cost == pytest.approx(0.0002)


def test_usage_summarizer_ignores_failed_search_receipts() -> None:
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(),
    )
    successful = ToolCall(
        receipt_id="successful-search",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="successful-search-req",
            response_hash="successful-search-res",
            cost_usd=0.02,
            response_payload={"data": [{"link": "https://example.com"}]},
        ),
    )
    failed = ToolCall(
        receipt_id="failed-search",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        outcome=ToolCallOutcome.PROVIDER_ERROR,
        details=ToolCallDetails(
            request_hash="failed-search-req",
            request_payload={"args": [], "kwargs": {"query": "Task B"}},
            response_hash=None,
            response_payload=None,
            cost_usd=None,
            extra={"error_type": "ToolProviderError", "error_message": "tool provider failed"},
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (successful, failed))
    assert total_tool_usage.search_tool.call_count == 1
    assert total_tool_usage.search_tool.cost == pytest.approx(0.02)
    assert total_tool_usage.search_tool_cost == pytest.approx(0.02)


def test_usage_summarizer_preserves_reasoning_tokens_in_llm_summary() -> None:
    model = "deepseek-ai/DeepSeek-V3.2-TEE"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=22,
        reasoning_tokens=7,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "chutes": {
                    model: totals,
                },
            },
        ),
    )

    receipt = ToolCall(
        receipt_id="llm-chutes-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-chutes-summary-req",
            request_payload={"kwargs": {"provider": "chutes", "model": model}},
            response_hash="llm-chutes-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0123,
            actual_cost_usd=0.0123,
            actual_cost_provider="chutes",
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))

    model_usage = total_tool_usage.llm.providers["chutes"][model]
    assert total_tool_usage.llm.reasoning_tokens == 7
    assert model_usage.usage.reasoning_tokens == 7
    assert model_usage.cost == pytest.approx(0.0123)


def test_usage_summarizer_rejects_successful_llm_receipt_without_provider() -> None:
    model = "deepseek-ai/DeepSeek-V3.2-TEE"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "chutes": {
                    model: totals,
                },
            },
        ),
    )
    receipt = ToolCall(
        receipt_id="llm-missing-provider-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-missing-provider-summary-req",
            request_payload={"kwargs": {"model": model}},
            response_hash="llm-missing-provider-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0123,
            actual_cost_usd=0.0123,
            actual_cost_provider="chutes",
        ),
    )

    with pytest.raises(ValueError, match="llm_chat receipt requires request provider"):
        UsageSummarizer().summarize(session, (receipt,))


def test_usage_summarizer_prices_openrouter_native_llm_summary() -> None:
    model = "deepseek/deepseek-v3.2"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "openrouter": {
                    model: totals,
                },
            },
        ),
    )

    receipt = ToolCall(
        receipt_id="llm-openrouter-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-openrouter-summary-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="llm-openrouter-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0042,
            actual_cost_usd=0.0042,
            actual_cost_provider="openrouter",
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))

    model_usage = total_tool_usage.llm.providers["openrouter"][model]
    assert total_tool_usage.llm.call_count == 1
    assert total_tool_usage.llm_cost == pytest.approx(0.0042)
    assert model_usage.cost == pytest.approx(0.0042)


def test_receipt_usage_rejects_successful_llm_receipt_without_provider() -> None:
    model = "openai/gpt-oss-120b"
    receipt = ToolCall(
        receipt_id="llm-missing-provider",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-missing-provider-req",
            request_payload={"kwargs": {"model": model}},
            response_hash="llm-missing-provider-res",
            response_payload={
                "message": {"role": "assistant", "content": "ok"},
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            cost_usd=0.02,
            actual_cost_usd=0.02,
            actual_cost_provider="chutes",
        ),
    )

    with pytest.raises(ValueError, match="llm_chat receipt requires request provider"):
        evaluation_runner_module._usage_from_receipts((receipt,))


def test_receipt_usage_rejects_divergent_search_receipt_costs() -> None:
    receipt = ToolCall(
        receipt_id="search-parallel",
        session_id=uuid4(),
        uid=7,
        tool="search_web",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="search-parallel-req",
            request_payload={"kwargs": {"provider": "parallel", "search_queries": ["harnyx"]}},
            response_hash="search-parallel-res",
            response_payload={"data": []},
            cost_usd=0.0,
            actual_cost_usd=0.005,
            actual_cost_provider="parallel",
        ),
    )
    object.__setattr__(receipt.details, "cost_usd", 0.0)

    with pytest.raises(ValueError, match="must match"):
        evaluation_runner_module._usage_from_receipts((receipt,))


def test_receipt_usage_uses_cost_usd_when_actual_cost_is_missing() -> None:
    receipt = ToolCall(
        receipt_id="cost-only",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="cost-only-req",
            request_payload={"kwargs": {"provider": "chutes"}},
            response_hash="cost-only-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.02,
            reference_cost_usd=0.01,
            actual_cost_usd=None,
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    assert usage.total_cost_usd == pytest.approx(0.02)
    assert usage.reference_total_cost_usd == pytest.approx(0.02)
    assert usage.cost_by_provider == {"chutes": pytest.approx(0.02)}
    assert usage.reference_cost_by_provider == {"chutes": pytest.approx(0.02)}


def test_receipt_replay_keeps_explicitly_unavailable_actual_cost_unknown() -> None:
    receipt = ToolCall(
        receipt_id="unknown-openrouter-embedding-cost",
        session_id=uuid4(),
        uid=7,
        tool="embed_text",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="unknown-openrouter-embedding-cost-req",
            request_payload={
                "kwargs": {
                    "provider": "openrouter",
                    "model": "qwen/qwen3-embedding-8b",
                }
            },
            response_hash="unknown-openrouter-embedding-cost-res",
            response_payload={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
            cost_usd=None,
            actual_cost_usd=None,
            actual_cost_provider="openrouter",
            extra={"actual_cost_settlement_source": "unavailable"},
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    assert usage.total_cost_usd == 0.0
    assert usage.actual_total_cost_usd is None
    assert usage.actual_cost_by_provider == {}


def test_receipt_replay_records_zero_token_llm_usage_for_priced_receipt() -> None:
    model = "deepseek/deepseek-v3.2"
    receipt = ToolCall(
        receipt_id="priced-zero-token-llm",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="priced-zero-token-llm-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="priced-zero-token-llm-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.003,
            reference_cost_usd=0.003,
            actual_cost_usd=0.003,
            actual_cost_provider="openrouter",
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    totals = usage.llm_usage_totals["openrouter"][model]
    assert totals.call_count == 1
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0
    assert totals.total_tokens == 0
    assert totals.reasoning_tokens == 0
    assert usage.total_cost_usd == pytest.approx(0.003)


def test_receipt_replay_coalesces_unavailable_reasoning_tokens() -> None:
    model = "deepseek/deepseek-v3.2"
    receipt = ToolCall(
        receipt_id="priced-null-reasoning-llm",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="priced-null-reasoning-llm-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="priced-null-reasoning-llm-res",
            response_payload={
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 6,
                    "reasoning_tokens": None,
                    "total_tokens": 9,
                }
            },
            cost_usd=0.003,
            reference_cost_usd=0.003,
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    totals = usage.llm_usage_totals["openrouter"][model]
    assert totals.prompt_tokens == 3
    assert totals.completion_tokens == 6
    assert totals.total_tokens == 9
    assert totals.reasoning_tokens == 0


def _sandbox_invocation_error(
    message: str,
    *,
    status_code: int = 0,
    detail_code: str | None = None,
    detail_exception: str = "RuntimeError",
    detail_error: str | None = None,
) -> SandboxInvocationError:
    return SandboxInvocationError(
        message,
        status_code=status_code,
        detail_code=detail_code,
        detail_exception=detail_exception,
        detail_error=detail_error or message,
    )


def _provider_tool_failure_error() -> SandboxInvocationError:
    return _sandbox_invocation_error(
        "tool route failed",
        status_code=500,
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
    )


class _ExhaustingOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log

    async def evaluate(self, request: MinerTaskRunRequest) -> None:
        session = self._sessions.get(request.session_id)
        assert session is not None
        self._sessions.update(session.mark_exhausted())
        self._receipt_log.record(
            ToolCall(
                receipt_id="receipt-1",
                session_id=request.session_id,
                uid=request.uid,
                tool="search_web",
                issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
                outcome=ToolCallOutcome.OK,
                details=ToolCallDetails(
                    request_hash="req",
                    response_hash="res",
                    cost_usd=0.25,
                ),
            )
        )
        raise SessionBudgetExhaustedError("session exhausted during entrypoint invocation")


class _ExhaustingAfterPlatformToolProxyControlReceiptOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log

    async def evaluate(self, request: MinerTaskRunRequest) -> None:
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id="platform-tool-proxy-control",
            issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            cost_usd=0.0,
            outcome=ToolCallOutcome.INTERNAL_ERROR,
            active_attempt=session.active_attempt,
            extra={
                "platform_tool_proxy_error_code": "platform_tool_proxy_denied",
                "platform_tool_proxy_status_code": "403",
                "error_type": "PlatformToolProxyInvocationError",
                "error_message": "platform tool proxy denied",
            },
        )
        self._sessions.update(session.mark_exhausted())
        raise SessionBudgetExhaustedError("session exhausted after proxy control failure")


class _RetryThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        assert session.status is SessionStatus.ACTIVE
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"receipt-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
        )
        if self.calls == 1:
            raise _sandbox_invocation_error("transient sandbox failure")
        details = EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=0.75,
                total_score=0.75,
                scoring_version="v1",
            ),
            total_tool_usage=_search_usage(self._receipt_log, request.session_id),
        )
        tool_receipts = tuple(self._receipt_log.for_session(request.session_id))
        self._receipt_log.clear_session(request.session_id)
        return TaskRunOutcome(
            run=MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text="answer"),
                details=details,
                completed_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
            ),
            tool_receipts=tool_receipts,
            usage=TokenUsageSummary.empty(),
        )


class _GenericFailureOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"generic-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            cost_usd=0.25,
        )
        raise RuntimeError("scoring failed")


class _ScoringTimeoutThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
    ) -> None:
        self._sessions = sessions
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        if self.calls == 1:
            raise httpx.ReadTimeout(
                "embedding timed out",
                request=httpx.Request("POST", "https://validator.invalid/scoring"),
            )
        return _successful_outcome(request)


class _AlwaysMinerTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
        total_tokens: int | None = None,
        elapsed_ms: float | None = None,
        status_code: int = 504,
        detail_exception: str = "TimeoutError",
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self._total_tokens = total_tokens
        self._elapsed_ms = elapsed_ms
        self._status_code = status_code
        self._detail_exception = detail_exception
        self.calls = 0
        self.session_ids: list = []
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        if self._total_tokens is not None and self._elapsed_ms is not None:
            _record_receipt(
                self._receipt_log,
                session_id=request.session_id,
                uid=request.uid,
                receipt_id=f"timeout-{self.calls}",
                issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
                cost_usd=0.0,
                tool="llm_chat",
                request_payload={
                    "args": [],
                    "kwargs": {
                        "provider": "chutes",
                        "model": "google/gemma-4-31B-turbo-TEE",
                    },
                },
                response_payload={"usage": {"total_tokens": self._total_tokens}},
                execution=ToolExecutionFacts(elapsed_ms=self._elapsed_ms),
            )
        raise _sandbox_invocation_error(
            "sandbox entrypoint request timed out",
            status_code=self._status_code,
            detail_exception=self._detail_exception,
            detail_error="sandbox entrypoint request timed out",
        )


class _AlwaysScoringTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
    ) -> None:
        self._sessions = sessions
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        raise httpx.ReadTimeout(
            "embedding timed out",
            request=httpx.Request("POST", "https://validator.invalid/scoring"),
        )


class _MinerTimeoutWithNonQualifyingReceiptsOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"search-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
            tool="search_web",
        )
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"failed-llm-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.0,
            tool="llm_chat",
            outcome=ToolCallOutcome.PROVIDER_ERROR,
            request_payload={
                "args": [],
                "kwargs": {
                    "provider": "chutes",
                    "model": "google/gemma-4-31B-turbo-TEE",
                },
            },
            response_payload={"usage": {"total_tokens": 500}},
            execution=ToolExecutionFacts(elapsed_ms=500.0),
        )
        raise _sandbox_invocation_error(
            "sandbox entrypoint request timed out",
            status_code=504,
            detail_exception="TimeoutError",
            detail_error="sandbox entrypoint request timed out",
        )


class _RetryThenExhaustedOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        if self.calls == 1:
            _record_receipt(
                self._receipt_log,
                session_id=request.session_id,
                uid=request.uid,
                receipt_id="near-limit",
                issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
                cost_usd=0.04,
            )
            raise _sandbox_invocation_error("transient sandbox failure")
        self._sessions.update(session.mark_exhausted())
        raise SessionBudgetExhaustedError("session exhausted during retry")


def _record_provider_failure(
    progress: FileBackedRunProgress,
    *,
    request: MinerTaskRunRequest,
    provider: str = "desearch",
    model: str = "search_web",
    reason: str = "http_402: subscription usage cap exceeded",
) -> None:
    progress.record_provider_call(
        session_id=request.session_id,
        provider=provider,
        model=model,
    )
    progress.record_provider_failure(
        session_id=request.session_id,
        provider=provider,
        model=model,
        reason=reason,
    )


def _record_platform_tool_proxy_timeout_receipt(
    receipt_log: FakeReceiptLog,
    *,
    request: MinerTaskRunRequest,
    active_attempt: int,
    receipt_id: str,
) -> None:
    _record_receipt(
        receipt_log,
        session_id=request.session_id,
        uid=request.uid,
        receipt_id=receipt_id,
        issued_at=datetime(2025, 10, 17, 12, active_attempt, tzinfo=UTC),
        cost_usd=0.0,
        outcome=ToolCallOutcome.TIMEOUT,
        active_attempt=active_attempt,
        extra={
            "platform_tool_proxy_error_code": "tool_timeout",
            "platform_tool_proxy_status_code": "0",
            "error_type": "PlatformToolProxyToolTimeoutError",
            "error_message": "platform tool proxy execution timed out while awaiting tool result",
        },
    )


def _seed_provider_evidence(
    progress: FileBackedRunProgress,
    *,
    batch_id,
    provider: str,
    model: str,
    total_calls: int,
    failed_calls: int,
) -> None:
    for index in range(total_calls):
        session_id = uuid4()
        progress.register_task_session(
            batch_id=batch_id,
            session_id=session_id,
        )
        progress.record_provider_call(
            session_id=session_id,
            provider=provider,
            model=model,
        )
        if index < failed_calls:
            progress.record_provider_failure(
                session_id=session_id,
                provider=provider,
                model=model,
                reason="http_402: subscription usage cap exceeded",
            )
        progress.clear_task_session(session_id)


class _ProviderFailureThenSandboxFailureOrchestrator:
    def __init__(self, *, progress: FileBackedRunProgress) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        if self.calls == 1:
            _record_provider_failure(self._progress, request=request)
            raise _sandbox_invocation_error("tool route failed")
        raise _sandbox_invocation_error("plain sandbox failure")


class _ProviderFailureThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        progress: FileBackedRunProgress,
    ) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        _record_provider_failure(self._progress, request=request)
        return _successful_outcome(request)

    async def execute(self, request: MinerTaskRunRequest, *, phase_recorder=None) -> TaskExecutionOutcome:
        _ = phase_recorder
        self.calls += 1
        _record_provider_failure(self._progress, request=request)
        return _successful_execution(request)


class _ProviderBatchFailureOrchestrator:
    def __init__(self, *, progress: FileBackedRunProgress) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        _record_provider_failure(self._progress, request=request)
        raise _provider_tool_failure_error()


class _PlatformToolProxyReceiptOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
        error_code: str,
        caught_by_miner: bool,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self._error_code = error_code
        self._caught_by_miner = caught_by_miner
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"platform-tool-proxy-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.0,
            outcome=ToolCallOutcome.INTERNAL_ERROR,
            active_attempt=session.active_attempt,
            extra={
                "platform_tool_proxy_error_code": self._error_code,
                "platform_tool_proxy_status_code": "400",
                "error_type": "PlatformToolProxyInvocationError",
                "error_message": f"{self._error_code} message",
            },
        )
        if self._caught_by_miner:
            return _successful_outcome(request)
        raise _sandbox_invocation_error(
            f"{self._error_code} message",
            status_code=400,
            detail_code="UnhandledException",
            detail_exception="ToolInvocationError",
            detail_error=f"{self._error_code} message",
        )


class _PlatformToolProxyTimeoutThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        if self.calls == 1:
            _record_platform_tool_proxy_timeout_receipt(
                self._receipt_log,
                request=request,
                active_attempt=session.active_attempt,
                receipt_id="platform-tool-proxy-timeout-1",
            )
            raise _provider_tool_failure_error()
        receipts = tuple(self._receipt_log.for_session(request.session_id))
        self._receipt_log.clear_session(request.session_id)
        outcome = _successful_outcome(request)
        return TaskRunOutcome(
            run=outcome.run,
            tool_receipts=receipts,
            usage=outcome.usage,
        )


class _AlwaysPlatformToolProxyTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        _record_platform_tool_proxy_timeout_receipt(
            self._receipt_log,
            request=request,
            active_attempt=session.active_attempt,
            receipt_id=f"platform-tool-proxy-timeout-{self.calls}",
        )
        raise _provider_tool_failure_error()


class _SandboxTimeoutAfterPlatformToolProxyControlReceiptOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id="platform-tool-proxy-grant-failed",
            issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            cost_usd=0.0,
            outcome=ToolCallOutcome.INTERNAL_ERROR,
            active_attempt=session.active_attempt,
            extra={
                "platform_tool_proxy_error_code": "platform_tool_proxy_grant_failed",
                "platform_tool_proxy_status_code": "0",
                "error_type": "PlatformToolProxyInvocationError",
                "error_message": "platform tool proxy grant failed",
            },
        )
        _record_platform_tool_proxy_timeout_receipt(
            self._receipt_log,
            request=request,
            active_attempt=session.active_attempt,
            receipt_id="platform-tool-proxy-timeout",
        )
        raise _sandbox_invocation_error(
            "sandbox entrypoint request timed out",
            status_code=504,
            detail_exception="TimeoutException",
            detail_error="sandbox entrypoint request timed out",
        )


class _PlatformToolProxyTimeoutThenDifferentMinerExceptionOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_platform_tool_proxy_timeout_receipt(
            self._receipt_log,
            request=request,
            active_attempt=session.active_attempt,
            receipt_id="platform-tool-proxy-timeout-before-crash",
        )
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="KeyError",
            detail_error="missing key",
        )


class _UnhandledMinerCrashOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="KeyError",
            detail_error="missing key",
        )


class _UnhandledMinerTypeErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="TypeError",
            detail_error="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
        )


class _MissingEntrypointOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox entrypoint missing",
            status_code=404,
            detail_code="MissingEntrypoint",
            detail_exception="KeyError",
            detail_error="'query'",
        )


class _PreloadContractFailureOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload contract failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="TypeError",
            detail_error="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
        )


class _PreloadRuntimeErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload runtime failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="RuntimeError",
            detail_error="agent import failed",
        )


class _PreloadImportErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload import failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="ImportError",
            detail_error="missing miner dependency",
        )


class _PreloadInfrastructureFailureOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload infrastructure failed",
            status_code=500,
            detail_code="PreloadInfrastructureFailed",
            detail_exception="RuntimeError",
            detail_error="AGENT_PATH is required",
        )


class _EntrypointUnavailableOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "entrypoint unavailable",
            status_code=500,
            detail_code="EntrypointUnavailable",
            detail_exception="KeyError",
            detail_error="'query'",
        )


class _MinerResponseValidationOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise MinerResponseValidationError("miner returned invalid response payload")


class _ScoringRetryExhaustedOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise LlmRetryExhaustedError("embedding retries exhausted")


class _ScoringRetryExhaustedWithJudgeUsageOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        exc = LlmRetryExhaustedError("embedding retries exhausted")
        exc.__dict__["judge_usage"] = _judge_usage(actual_cost_usd=0.031)
        exc.__dict__["evaluation_trace"] = EvaluationTrace(
            entrypoint_invocation_ms=120.0,
            scoring_ms=250.0,
            orchestration_ms=400.0,
            scoring_judge_selected_routes=("chutes/judge-model",),
            scoring_judge_attempt_count=2,
            scoring_judge_retry_count=1,
            scoring_judge_retry_reasons=("transport_error",),
            scoring_judge_duration_ms=200.0,
            scoring_judge_status="exhausted",
        )
        raise exc


class _SandboxTimeoutOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox timed out",
            status_code=408,
            detail_code="SandboxTimeout",
            detail_exception="TimeoutError",
            detail_error="sandbox timed out",
        )


class _EmbeddingRetryExhaustedOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise VertexEmbeddingRetryExhaustedError("embedding retries exhausted")


class _SuccessfulOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        return _successful_outcome(request)


async def test_evaluation_runner_records_exhausted_submission(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="budget test"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(
            TaskRunOrchestrator,
            _ExhaustingOrchestrator(
                sessions=session_registry,
                receipt_log=receipt_log,
            ),
        ),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.response is None
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "session_budget_exhausted"
    assert submission.run.details.error.message == "session exhausted during entrypoint invocation"
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.25)
    assert tuple(receipt.receipt_id for receipt in submission.execution_log) == ("receipt-1",)
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_assigned_final_timeout_returns_terminal_failed_result(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 3, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="assigned timeout"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=12,
        elapsed_ms=500.0,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=2,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 1
    assert result.result is not None
    assert result.result.run.details.error is not None
    assert result.result.run.details.error.code == "timeout_miner_owned"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert result.terminal_attempt.max_attempts == 2
    assert evaluation_store.records == [result.result]


async def test_evaluation_runner_assigned_task_returns_platform_result_when_submission_progress_recording_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    progress = _FailingSubmissionProgress(storage_root=tmp_path / "run-progress")
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        progress=progress,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is not None
    assert result.result.run.details.error is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.SUCCEEDED
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="progress.record",
    )


async def test_evaluation_runner_assigned_task_returns_platform_result_when_evaluation_recording_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    evaluation_store = _FailOnNthRecordEvaluationStore(fail_on_call=1)
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        evaluation_records=evaluation_store,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is not None
    assert result.result.run.details.error is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.SUCCEEDED
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="evaluation_records.record",
    )


async def test_evaluation_runner_assigned_task_returns_platform_result_when_terminal_attempt_recording_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    progress = _FailingTerminatedAttemptProgress(storage_root=tmp_path / "run-progress")
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        progress=progress,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert result.result is not None
    assert result.result.run.details.error is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.SUCCEEDED
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="progress.record_terminated_attempt",
    )


async def test_evaluation_runner_assigned_setup_failure_returns_platform_result_when_progress_recording_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    progress = _FailingSubmissionProgress(storage_root=tmp_path / "run-progress")
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        progress=progress,
    )

    results = await runner.record_assigned_task_setup_failures(
        batch_id=batch_id,
        artifact=artifact,
        assignments=(
            MinerTaskWorkAssignment(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                attempt_number=1,
                max_attempts=1,
                assignment_token=_ASSIGNMENT_TOKEN,
            ),
        ),
        error_code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
        error_message="script validation failed",
    )

    assert len(results) == 1
    result = results[0]
    assert result.result is not None
    assert result.result.run.details.error is not None
    assert result.result.run.details.error.code == "script_validation_failed"
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="progress.record",
    )


async def test_evaluation_runner_assigned_task_result_survives_local_recording_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    evaluation_store = _FailOnNthRecordEvaluationStore(fail_on_call=1)
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        evaluation_records=evaluation_store,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "scoring_llm_retry_exhausted"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="evaluation_records.record",
    )


async def test_evaluation_runner_assigned_retry_result_survives_local_recording_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    progress = _FailingTerminatedAttemptProgress(storage_root=tmp_path / "run-progress")
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        progress=progress,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _SandboxTimeoutOrchestrator()),
    )

    assert result.result is None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "SandboxTimeout"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY
    assert result.terminal_attempt.terminal_effect is None
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="progress.record_terminated_attempt",
    )


async def test_evaluation_runner_assigned_final_timeout_result_survives_local_recording_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_validator.scheduler")
    evaluation_store = _FailOnNthRecordEvaluationStore(fail_on_call=1)
    clock = _ClockSequence(
        datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        datetime(2025, 10, 17, 12, 3, tzinfo=UTC),
    )
    runner, batch_id, artifact, task, session_registry, receipt_log, _, _ = _assigned_task_test_context(
        tmp_path,
        evaluation_records=evaluation_store,
        clock=clock,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=12,
        elapsed_ms=500.0,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=2,
        max_attempts=2,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert result.result is not None
    assert result.result.run.details.error is not None
    assert result.result.run.details.error.code == "timeout_miner_owned"
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "timeout_miner_owned"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    _assert_local_recording_warning(
        caplog,
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        result=result,
        exception_type="RuntimeError",
        local_write_target="evaluation_records.record",
    )


async def test_evaluation_runner_whole_batch_evaluation_still_records_local_submissions(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, _, _, progress, evaluation_store = _assigned_task_test_context(tmp_path)

    outcome = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )

    assert outcome.submissions
    page = progress.completed_run_page(batch_id=batch_id, after_sequence=0, limit=10)
    assert [item["submission"] for item in page["items"] if item["submission"] is not None] == list(outcome.submissions)
    assert evaluation_store.records == list(outcome.submissions)


async def test_evaluation_runner_assigned_task_final_scoring_failure_stays_task_result(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="assigned scoring retry exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "scoring_llm_retry_exhausted"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert evaluation_store.records[0].run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )


async def test_evaluation_runner_assigned_task_final_scoring_failure_records_partial_judge_usage(
    tmp_path: Path,
) -> None:
    runner, batch_id, artifact, task, _, _, _, evaluation_store = _assigned_task_test_context(tmp_path)

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedWithJudgeUsageOrchestrator()),
    )

    assert result.result is not None
    assert result.terminal_attempt.error_code == "scoring_llm_retry_exhausted"
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert evaluation_store.records[0].run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )
    judge_usage = evaluation_store.records[0].run.details.scoring_judge_usage
    assert judge_usage is not None
    assert judge_usage.call_count == 1
    assert judge_usage.total_tokens == 120
    assert judge_usage.actual_cost_usd == pytest.approx(0.031)
    trace = evaluation_store.records[0].run.details.trace
    assert trace is not None
    assert trace.entrypoint_invocation_ms == pytest.approx(120.0)
    assert trace.scoring_ms == pytest.approx(250.0)
    assert trace.orchestration_ms == pytest.approx(400.0)
    assert trace.scoring_judge_selected_routes == ("chutes/judge-model",)
    assert trace.scoring_judge_attempt_count == 2
    assert trace.scoring_judge_retry_count == 1
    assert trace.scoring_judge_retry_reasons == ("transport_error",)
    assert trace.scoring_judge_duration_ms == pytest.approx(200.0)
    assert trace.scoring_judge_status == "exhausted"


async def test_evaluation_runner_assigned_task_final_sandbox_invocation_failure_stays_task_result(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="assigned sandbox invocation failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=uuid4(),
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(
            TaskRunOrchestrator,
            _AlwaysMinerTimeoutOrchestrator(
                sessions=session_registry,
                receipt_log=receipt_log,
                status_code=500,
                detail_exception="RuntimeError",
            ),
        ),
    )

    assert result.result is not None
    assert result.terminal_attempt.status is MinerTaskAttemptStatus.FAILED
    assert result.terminal_attempt.error_code == "sandbox_invocation_failed"
    assert result.terminal_attempt.retry_decision is MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert evaluation_store.records[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="sandbox entrypoint request timed out",
    )


async def test_evaluation_runner_identifies_proxy_control_failure_before_later_budget_exhaustion(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="proxy control before budget exhaustion"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ExhaustingAfterPlatformToolProxyControlReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    with pytest.raises(ValidatorBatchFailedError, match="platform tool proxy control failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert exc_info.value.failure_detail.error_code == "platform_tool_proxy_denied"
    assert exc_info.value.failure_detail.exception_type == "PlatformToolProxyInvocationError"
    assert exc_info.value.failure_detail.error_message == (
        "platform tool proxy control failure: platform_tool_proxy_denied"
    )
    assert evaluation_store.records == []


async def test_evaluation_runner_retries_transient_invocation_with_new_session_and_accumulated_usage(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="retry test"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _RetryThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[-1]
    assert submission.score == pytest.approx(0.75)
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.25)
    assert sorted(receipt.receipt_id for receipt in submission.execution_log) == ["receipt-2"]
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert attempt_logs == [("receipt-1",), ("receipt-2",)]


async def test_evaluation_runner_fails_batch_on_generic_post_invoke_failure(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="generic failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _GenericFailureOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    with pytest.raises(ValidatorBatchFailedError, match="scoring failed") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


async def test_evaluation_runner_sandbox_timeout_terminalizes_without_scheduler_requeue(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout retry"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=2000.0,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )
    assert evaluation_store.records == list(result.submissions)
    assert session_registry.get(orchestrator.session_ids[0]) is None
    assert tuple(receipt_log.for_session(orchestrator.session_ids[0])) == ()


async def test_evaluation_runner_fails_batch_after_scoring_timeout_retry_exhaustion(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _AlwaysScoringTimeoutOrchestrator(
        sessions=session_registry,
    )

    with pytest.raises(ValidatorBatchFailedError, match="embedding timed out") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "validator_internal_timeout"
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert evaluation_store.records == []


async def test_evaluation_runner_records_sandbox_boundary_timeout_as_miner_owned(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout within threshold"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=1200.0,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_treats_http_client_timeoutexception_as_sandbox_timeout(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="http client timeout exception"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=1200.0,
        detail_exception="TimeoutException",
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_does_not_treat_non_504_timeouterror_as_sandbox_timeout(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="non-boundary timeout-like sandbox error"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        status_code=500,
        detail_exception="TimeoutError",
    )

    with pytest.raises(
        ValidatorBatchFailedError,
        match="sandbox entrypoint request timed out",
    ) as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.completed_submissions is not None
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="sandbox entrypoint request timed out",
    )
    assert exc.remaining_tasks == ()


async def test_evaluation_runner_terminalizes_sandbox_timeout_without_receipts(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout miner owned without evidence"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_current_attempt_receipts_for_terminal_timeout(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout evidence filter"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=2,
    )
    orchestrator = _MinerTimeoutWithNonQualifyingReceiptsOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_zero_score_for_invalid_miner_response(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="invalid miner response"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _MinerResponseValidationOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="miner_response_invalid",
        message="miner returned invalid response payload",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_zero_score_for_scoring_retry_exhaustion(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="scoring retry exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(ValidatorBatchFailedError, match="embedding retries exhausted") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
        )
    assert exc_info.value.error_code == MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED
    assert exc_info.value.completed_submissions is not None
    submission = exc_info.value.completed_submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_embedding_retry_exhausted_submission(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="budget test"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(ValidatorBatchFailedError, match="embedding retries exhausted") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _EmbeddingRetryExhaustedOrchestrator()),
        )
    assert exc_info.value.error_code == MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED
    assert exc_info.value.completed_submissions is not None
    submission = exc_info.value.completed_submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_budget_exhausted_when_retry_starts_near_limit(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="near budget"),
        reference_answer=ReferenceAnswer(text="reference"),
        budget_usd=0.05,
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _RetryThenExhaustedOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[-1]
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.details.error == EvaluationError(
        code="session_budget_exhausted",
        message="session exhausted during retry",
    )
    assert submission.run.details.total_tool_usage.search_tool.call_count == 0
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.0)
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_keeps_valid_response_when_provider_failure_stays_below_threshold(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure success"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderFailureThenSuccessOrchestrator(
        progress=progress,
    )

    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].score == pytest.approx(0.75)
    assert evaluation_store.records == list(result.submissions)
    assert progress.provider_evidence(batch_id) == (
        {
            "provider": "desearch",
            "model": "search_web",
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "http_402: subscription usage cap exceeded",
        },
    )


@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_denied",
        "platform_tool_proxy_grant_failed",
    ],
)
async def test_evaluation_runner_platform_proxy_control_categories_fail_validator(
    tmp_path: Path,
    error_code: str,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} caught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=True,
    )

    with pytest.raises(ValidatorBatchFailedError, match="platform tool proxy control failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == error_code
    assert exc_info.value.failure_detail.error_code == error_code
    assert exc_info.value.failure_detail.exception_type == "PlatformToolProxyInvocationError"
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_execution_failed",
        "provider_failed",
    ],
)
async def test_evaluation_runner_platform_proxy_execution_categories_do_not_fail_validator_when_caught(
    tmp_path: Path,
    error_code: str,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} caught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=True,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error is None
    assert evaluation_store.records == list(result.submissions)


@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_execution_failed",
        "provider_failed",
    ],
)
async def test_evaluation_runner_platform_proxy_execution_categories_are_pair_scoped_when_uncaught(
    tmp_path: Path,
    error_code: str,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} uncaught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=False,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact_with_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error is not None
    assert result.submissions[0].run.details.error.code is MinerTaskErrorCode.MINER_UNHANDLED_EXCEPTION
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempts = [
        item["attempt"]
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert len(attempts) == 1
    assert attempts[0].error_code == "miner_unhandled_exception"
    assert attempts[0].execution_log[0].details.extra is not None
    assert attempts[0].execution_log[0].details.extra["platform_tool_proxy_error_code"] == error_code
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_keeps_miner_owned_proxy_failure_with_the_miner(
    tmp_path: Path,
) -> None:
    error_code = "provider_failed"
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} caught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=True,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error is None
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_retries_platform_tool_proxy_timeout_with_task_retry_count(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform tool proxy timeout retry"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _PlatformToolProxyTimeoutThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert orchestrator.active_attempts == [1, 1]
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == pytest.approx(0.75)
    assert submission.run.details.error is None
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_errors = [
        item["attempt"].error_code
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert "tool_timeout" in attempt_errors
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert ("platform-tool-proxy-timeout-1",) in attempt_logs
    assert any(item["kind"] == "completed_run" for item in page["items"])
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_timeout_owned_path_ignores_earlier_proxy_control_receipt(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="sandbox timeout after proxy grant failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _SandboxTimeoutAfterPlatformToolProxyControlReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact_with_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    terminated_attempts = [
        item["attempt"]
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert len(terminated_attempts) == 1
    attempt = terminated_attempts[0]
    assert attempt.error_code == "timeout_miner_owned"
    assert attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert {receipt.receipt_id for receipt in attempt.execution_log} == {
        "platform-tool-proxy-grant-failed",
        "platform-tool-proxy-timeout",
    }
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_does_not_rewrite_different_miner_exception_after_timeout_receipt(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform proxy timeout before miner crash"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _PlatformToolProxyTimeoutThenDifferentMinerExceptionOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_success_when_successful_fallback_crosses_provider_threshold(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    _seed_provider_evidence(
        progress,
        batch_id=batch_id,
        provider="desearch",
        model="search_web",
        total_calls=9,
        failed_calls=9,
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure fallback"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderFailureThenSuccessOrchestrator(
        progress=progress,
    )

    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error is None
    assert orchestrator.calls == 1
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_assigned_task_records_task_result_when_execution_crosses_provider_threshold(
    tmp_path: Path,
) -> None:
    progress = _progress(tmp_path)
    runner, batch_id, artifact, task, _, _, _, _ = _assigned_task_test_context(
        tmp_path,
        progress=progress,
    )
    _seed_provider_evidence(
        progress,
        batch_id=batch_id,
        provider="desearch",
        model="search_web",
        total_calls=9,
        failed_calls=9,
    )
    orchestrator = _ProviderFailureThenSuccessOrchestrator(
        progress=progress,
    )

    result = await runner.evaluate_assigned_task(
        batch_id=batch_id,
        artifact=artifact,
        task=task,
        attempt_number=1,
        max_attempts=1,
        assignment_token=_ASSIGNMENT_TOKEN,
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert result.result is not None
    assert result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.TASK_RESULT
    assert result.terminal_attempt.delivery_failure_detail is None
    assert orchestrator.calls == 1


async def test_evaluation_runner_keeps_provider_caused_terminal_failure_pair_scoped(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    _seed_provider_evidence(
        progress,
        batch_id=batch_id,
        provider="desearch",
        model="search_web",
        total_calls=9,
        failed_calls=9,
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider threshold"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=8,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderBatchFailureOrchestrator(
        progress=progress,
    )

    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert orchestrator.calls == 1
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_records_zero_score_for_unhandled_miner_exception(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="miner crash"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _UnhandledMinerCrashOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_keeps_query_runtime_type_error_as_miner_unhandled_exception(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="runtime type error"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _UnhandledMinerTypeErrorOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.details.error == EvaluationError(
        code="miner_unhandled_exception",
        message="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
    )


@pytest.mark.parametrize(
    ("orchestrator", "error_code"),
    ((_MissingEntrypointOrchestrator(), "script_validation_failed"),),
)
async def test_evaluation_runner_records_zero_score_for_script_validation_failures(
    orchestrator: TaskRunOrchestrator,
    error_code: str,
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="script invalid"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=orchestrator,
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code=error_code,
        message=submission.run.details.error.message,
    )
    assert evaluation_store.records == [submission]


@pytest.mark.parametrize(
    "orchestrator",
    (_PreloadInfrastructureFailureOrchestrator(),),
)
async def test_evaluate_artifact_with_state_preserves_sandbox_infrastructure_failures(
    orchestrator: TaskRunOrchestrator,
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(
            token_secret_bytes=8,
            execution_time_limit_seconds=300.0,
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="sandbox infra"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(ValidatorBatchFailedError) as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=orchestrator,
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.failure_detail.error_code == "sandbox_invocation_failed"


async def test_evaluation_runner_does_not_let_stale_provider_marker_poison_later_attempt(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure then sandbox failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _ProviderFailureThenSandboxFailureOrchestrator(progress=progress)

    with pytest.raises(ValidatorBatchFailedError, match="plain sandbox failure") as exc_info:
        await runner.evaluate_artifact(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert orchestrator.calls == 2
    assert exc.completed_submissions is not None
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="plain sandbox failure",
    )
    assert evaluation_store.records == list(exc.completed_submissions)


async def test_evaluation_runner_uses_bounded_continuous_worker_pool(tmp_path: Path) -> None:
    """Future failure: evaluation stays bounded while refilling each open slot."""
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(
            token_secret_bytes=8,
            execution_time_limit_seconds=300.0,
            artifact_task_parallelism=5,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"task-{index}"),
            reference_answer=ReferenceAnswer(text=f"reference-{index}"),
        )
        for index in range(6)
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _ContinuousPoolOrchestrator:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.max_active = 0
            self._active = 0
            self.first_wave_started = asyncio.Event()
            self.sixth_started = asyncio.Event()
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started.append(text)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if len(self.started) == 5:
                self.first_wave_started.set()
            if text == "task-5":
                self.sixth_started.set()
            await self.release_by_text[text].wait()
            self._active -= 1
            return _successful_outcome(request, score=1.0)

    orchestrator = _ContinuousPoolOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.first_wave_started.wait(), timeout=1.0)
        assert set(orchestrator.started) == {f"task-{index}" for index in range(5)}
        assert "task-5" not in orchestrator.started

        orchestrator.release_by_text["task-0"].set()
        await asyncio.wait_for(orchestrator.sixth_started.wait(), timeout=1.0)

        for task in tasks[1:]:
            orchestrator.release_by_text[task.query.text].set()

        result = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()

    assert orchestrator.max_active == 5
    assert [submission.run.task_id for submission in result.submissions] == [task.task_id for task in tasks]
    assert len(evaluation_store.records) == 6


async def test_evaluate_artifact_with_state_preserves_earlier_submissions_for_conclusive_failure(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    earlier_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="earlier success"),
        reference_answer=ReferenceAnswer(text="reference earlier"),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="conclusive later round"),
        reference_answer=ReferenceAnswer(text="reference later"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    earlier_submission = _submission_for_task(
        batch_id=batch_id,
        artifact=artifact,
        task=earlier_task,
    )

    class _AlwaysSandboxFailureOrchestrator:
        async def evaluate(self, _request: MinerTaskRunRequest) -> TaskRunOutcome:
            raise _sandbox_invocation_error("shared sandbox failure")

    with pytest.raises(ValidatorBatchFailedError, match="shared sandbox failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _AlwaysSandboxFailureOrchestrator()),
            earlier_submissions=(earlier_submission,),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.completed_submissions is not None
    assert exc.completed_submissions[0] == earlier_submission
    assert exc.completed_submissions[1].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="shared sandbox failure",
    )
    assert exc.remaining_tasks == ()


async def test_evaluate_artifact_with_state_preserves_partial_submissions_for_validator_batch_failure(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(
            token_secret_bytes=8,
            execution_time_limit_seconds=300.0,
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    completed_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="completed"),
        reference_answer=ReferenceAnswer(text="reference completed"),
    )
    pending_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="pending"),
        reference_answer=ReferenceAnswer(text="reference pending"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    failure_detail = ValidatorBatchFailureDetail(
        error_code="validator_internal_timeout",
        error_message="validator timeout",
        occurred_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        artifact_id=artifact.artifact_id,
        uid=artifact.uid,
        sandbox_diagnostics=SandboxFailureDiagnostics(
            image="harnyx/sandbox:test",
            status="exited",
            exit_code=255,
        ),
    )
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        artifact=artifact,
        task=completed_task,
    )

    async def run_artifact_worker(**kwargs) -> None:
        dispatch = kwargs["dispatch"]
        dispatch.submissions_by_index[0] = completed_submission
        dispatch.validator_failure = ValidatorBatchFailedError(
            error_code="validator_internal_timeout",
            message="validator timeout",
            failure_detail=failure_detail,
        )

    runner._run_artifact_worker = run_artifact_worker  # type: ignore[method-assign]

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(completed_task, pending_task),
            orchestrator=cast(TaskRunOrchestrator, object()),
        )

    exc = exc_info.value
    assert exc.completed_submissions == (completed_submission,)
    assert exc.remaining_tasks == (pending_task,)


async def test_evaluate_task_retry_loop_records_validator_batch_failure_detail(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform-owned batch failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    failure_detail = ValidatorBatchFailureDetail(
        error_code="validator_internal_timeout",
        error_message="validator timeout",
        occurred_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        uid=artifact.uid,
        sandbox_diagnostics=SandboxFailureDiagnostics(
            image="harnyx/sandbox:test",
            status="exited",
            exit_code=255,
        ),
    )

    async def evaluate_task_attempt(**kwargs) -> evaluation_runner_module.TaskAttemptDecision:
        return evaluation_runner_module._validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code="validator_internal_timeout",
                message="validator timeout",
                failure_detail=failure_detail,
            )
        )

    runner._evaluate_task_attempt = evaluate_task_attempt  # type: ignore[method-assign]

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout"):
        await runner._evaluate_task_with_retry(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            orchestrator=cast(TaskRunOrchestrator, object()),
        )

    attempts = progress.completed_run_page(batch_id, after_sequence=0, limit=10)["items"]
    terminal_attempt = attempts[-1]["attempt"]
    assert terminal_attempt.delivery_failure_detail == failure_detail


async def test_evaluate_artifact_with_state_preserves_partial_submissions_for_unexpected_failure(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(
            token_secret_bytes=8,
            execution_time_limit_seconds=300.0,
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    completed_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="completed"),
        reference_answer=ReferenceAnswer(text="reference completed"),
    )
    pending_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="pending"),
        reference_answer=ReferenceAnswer(text="reference pending"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        artifact=artifact,
        task=completed_task,
    )

    async def record_then_fail(**kwargs) -> None:
        dispatch = kwargs["dispatch"]
        dispatch.submissions_by_index[0] = completed_submission
        dispatch.unexpected_failure = RuntimeError("progress store failed")

    runner._run_artifact_worker = record_then_fail  # type: ignore[method-assign]

    with pytest.raises(UnexpectedArtifactExecutionError, match="progress store failed") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(completed_task, pending_task),
            orchestrator=cast(TaskRunOrchestrator, object()),
        )

    exc = exc_info.value
    assert exc.completed_submissions == (completed_submission,)
    assert exc.remaining_tasks == (pending_task,)
    assert isinstance(exc.cause, RuntimeError)


async def test_record_failure_for_artifact_preserves_partial_submissions_when_recording_fails(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _FailOnNthRecordEvaluationStore(fail_on_call=2)
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, execution_time_limit_seconds=300.0),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    first_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="first"),
        reference_answer=ReferenceAnswer(text="reference first"),
    )
    second_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="second"),
        reference_answer=ReferenceAnswer(text="reference second"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(UnexpectedArtifactExecutionError, match="evaluation record write failed") as exc_info:
        await runner.record_failure_for_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(first_task, second_task),
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            error_message="artifact setup failed",
        )

    exc = exc_info.value
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.task_id == first_task.task_id
    assert exc.remaining_tasks == (second_task,)
    assert isinstance(exc.cause, RuntimeError)


async def test_evaluation_runner_supports_serialized_artifact_execution(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(
            token_secret_bytes=8,
            execution_time_limit_seconds=300.0,
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"task-{index}"),
            reference_answer=ReferenceAnswer(text=f"reference-{index}"),
        )
        for index in range(3)
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _SerializedOrchestrator:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.max_active = 0
            self._active = 0
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}
            self.first_started = asyncio.Event()
            self.second_started = asyncio.Event()

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started.append(text)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if len(self.started) == 1:
                self.first_started.set()
            if len(self.started) == 2:
                self.second_started.set()
            await self.release_by_text[text].wait()
            self._active -= 1
            return _successful_outcome(request, score=1.0)

    orchestrator = _SerializedOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.first_started.wait(), timeout=1.0)
        assert orchestrator.started == ["task-0"]
        assert not orchestrator.second_started.is_set()

        orchestrator.release_by_text["task-0"].set()
        await asyncio.wait_for(orchestrator.second_started.wait(), timeout=1.0)

        orchestrator.release_by_text["task-1"].set()
        orchestrator.release_by_text["task-2"].set()
        result = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()
    assert orchestrator.max_active == 1
    assert [submission.run.task_id for submission in result.submissions] == [task.task_id for task in tasks]
    assert len(evaluation_store.records) == 3
