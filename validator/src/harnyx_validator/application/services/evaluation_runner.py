"""Helper to run miner tasks for a single artifact."""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from harnyx_commons.application.dto.session import SessionEnvelope, SessionIssued, SessionTokenRequest
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    EvaluationTrace,
    MinerTask,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.session import SessionStatus
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.miner_task_failure_policy import (
    PLATFORM_TOOL_PROXY_TIMEOUT_ERROR_CODE,
    SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION,
    TERMINAL_TIMEOUT_ERROR_MESSAGE,
    ProviderFailureEvidence,
    is_platform_tool_proxy_timeout_receipt,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    is_uncaught_platform_tool_proxy_timeout_sandbox_invocation,
)
from harnyx_commons.sandbox.client import SandboxClient, SandboxInvokeError
from harnyx_commons.tools.receipt_accounting import usage_from_receipts as _usage_from_receipts
from harnyx_miner_sdk.sandbox_protocol import SANDBOX_OOM_AFFECTED, SandboxAdmission
from harnyx_validator.application.assigned_work import (
    TASK_START_WAIT_ATTEMPTS,
    TASK_START_WAIT_SECONDS,
    AssignedArtifactWork,
    ClaimedAssignedTask,
    PhaseRecorder,
    QueueStopReason,
    task_start_timeout_result,
)
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptDiagnostics,
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
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.domain.evaluation import MinerTaskRun

if TYPE_CHECKING:
    from harnyx_validator.application.scheduler import SchedulerConfig

Clock = Callable[[], datetime]
SubmissionFactory = Callable[[MinerTask, SessionIssued], Awaitable[MinerTaskRunSubmission]]
SessionStartedCallback = Callable[[UUID], None]
TaskSessionLimiter = AbstractAsyncContextManager[None]

logger = logging.getLogger("harnyx_validator.scheduler")
measurement_logger = logging.getLogger("harnyx_validator.measurement")
LOCAL_RETRY_ATTEMPTS = 2
VALIDATOR_OWNED_PLATFORM_TOOL_PROXY_CONTROL_ERROR_CODES = frozenset(
    {
        "platform_tool_proxy_denied",
        "platform_tool_proxy_grant_failed",
    }
)


@asynccontextmanager
async def _limited_task_session(limiter: TaskSessionLimiter | None) -> AsyncIterator[None]:
    if limiter is None:
        yield
        return
    async with limiter:
        yield


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


def _monotonic_elapsed_ms(*, started_at: float, completed_at: float) -> float:
    return round((completed_at - started_at) * 1000.0, 3)


def _log_session_finished(
    *,
    batch_id: UUID,
    session_id: UUID,
    artifact_id: UUID,
    task_id: UUID,
    uid: int,
    attempt_count: int,
    session_ms: float,
    terminal_outcome: str,
    error_code: str | None,
) -> None:
    measurement_logger.info(
        "miner-task session finished",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "session_id": str(session_id),
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "uid": uid,
                "attempt_count": attempt_count,
                "session_ms": session_ms,
                "terminal_outcome": terminal_outcome,
                "error_code": error_code,
            }
        },
    )


class AttemptControlKind(StrEnum):
    # Outer control-flow action for one task-attempt loop step.
    SUBMISSION = "submission"
    RETRY = "retry"
    REVIEW_TIMEOUT = "review_timeout"
    VALIDATOR_BATCH_FAILURE = "validator_batch_failure"


@dataclass(frozen=True, slots=True)
class TaskAttemptDecision:
    kind: AttemptControlKind
    submission: MinerTaskRunSubmission | None = None
    attempt_execution_log: tuple[ToolCall, ...] = ()
    attempt_error_code: MinerTaskErrorCode | str | None = None
    timeout_owner: str | None = None
    retry_exc: Exception | None = None
    timeout_exc: SandboxInvocationError | None = None
    validator_failure: ValidatorBatchFailedError | None = None


@dataclass(frozen=True, slots=True)
class _AssignedTaskFinalization:
    result: PlatformOwnedTaskResult
    local_submission: MinerTaskRunSubmission | None
    terminal_outcome: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class _PlatformToolProxyControlFailure:
    error_code: MinerTaskErrorCode
    error_message: str
    occurred_at: datetime
    exception_type: str | None = None


class ValidatorBatchFailedError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        failure_detail: ValidatorBatchFailureDetail,
        completed_submissions: tuple[MinerTaskRunSubmission, ...] | None = None,
        remaining_tasks: tuple[MinerTask, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks


class ArtifactExecutionFailedError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: MinerTaskErrorCode,
        message: str,
        failure_detail: ValidatorBatchFailureDetail,
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        remaining_tasks: tuple[MinerTask, ...],
        artifact_breaker_tripped: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks
        self.artifact_breaker_tripped = artifact_breaker_tripped


class UnexpectedArtifactExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        cause: Exception,
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        remaining_tasks: tuple[MinerTask, ...],
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks


class _PartialSessionIssueError(RuntimeError):
    def __init__(self, *, issued: SessionIssued, cause: Exception) -> None:
        super().__init__(str(cause))
        self.issued = issued
        self.cause = cause


@dataclass(slots=True)
class _ArtifactDispatchState:
    submissions_by_index: list[MinerTaskRunSubmission | None]
    validator_failure: ValidatorBatchFailedError | None = None
    unexpected_failure: Exception | None = None


@dataclass(frozen=True, slots=True)
class _ActiveAssignedTask:
    assignment: MinerTaskWorkAssignment
    lifecycle: _AssignedAttemptLifecycle | None = None
    session_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class _AssignedAttemptRunOutcome:
    execution_to_queue: PlatformOwnedTaskExecution | None = None
    result_to_queue: PlatformOwnedTaskResult | None = None
    result_already_queued: bool = False


@dataclass(frozen=True, slots=True)
class _IssuedAssignedAttempt:
    assignment: MinerTaskWorkAssignment
    issued: SessionIssued
    attempt_started_at: datetime
    session_started_at: float
    phase_recorder: _AssignedAttemptPhaseRecorder


class _AssignmentStartFailedError(Exception):
    def __init__(self, outcome: _AssignedAttemptRunOutcome) -> None:
        super().__init__("assigned attempt start failed")
        self.outcome = outcome


@dataclass(slots=True)
class _AssignedAttemptPhaseRecorder:
    started_at: float
    phase: str = "session_start"
    platform_tool_activity_observed: bool = False

    def mark(self, phase: str) -> str:
        previous = self.phase
        self.phase = phase
        if phase.startswith("platform_tool_proxy_"):
            self.platform_tool_activity_observed = True
        return previous

    def diagnostics(
        self,
        *,
        timeout_owner: str | None = None,
        failure_owner: str | None = None,
    ) -> MinerTaskAttemptDiagnostics:
        return MinerTaskAttemptDiagnostics(
            phase=self.phase,
            timeout_owner=timeout_owner,
            failure_owner=failure_owner,
            elapsed_ms=_monotonic_elapsed_ms(
                started_at=self.started_at,
                completed_at=time.monotonic(),
            ),
            platform_tool_activity_observed=self.platform_tool_activity_observed,
        )


@dataclass(slots=True)
class _AssignedAttemptLifecycle:
    runner: EvaluationRunner
    batch_id: UUID
    artifact: ScriptArtifactSpec
    claimed: ClaimedAssignedTask
    phase_recorder: _AssignedAttemptPhaseRecorder
    issued: SessionIssued | None = None
    admission: SandboxAdmission | None = None

    def start(self) -> _AssignedAttemptStartContext:
        return _AssignedAttemptStartContext(self)

    def build_failure_result(
        self,
        *,
        exc: Exception,
        attempt_started_at: datetime,
    ) -> PlatformOwnedTaskResult:
        return self.runner._assigned_task_unexpected_failure_result(
            batch_id=self.batch_id,
            artifact=self.artifact,
            task=self.claimed.assignment.task,
            issued=self.issued,
            attempt_number=self.claimed.assignment.attempt_number,
            max_attempts=self.claimed.assignment.max_attempts,
            started_at=attempt_started_at,
            exc=exc,
            diagnostics=self.phase_recorder.diagnostics(
                failure_owner=_exception_type_name(exc),
            ),
        )


@dataclass(slots=True)
class _AssignedAttemptStartContext:
    lifecycle: _AssignedAttemptLifecycle

    async def __aenter__(self) -> _IssuedAssignedAttempt:
        assignment = self.lifecycle.claimed.assignment
        session_started_at = time.monotonic()
        attempt_started_at = self.lifecycle.runner._clock()
        try:
            issued = self.lifecycle.runner._issue_session(
                batch_id=self.lifecycle.batch_id,
                uid=self.lifecycle.artifact.uid,
                artifact_id=self.lifecycle.artifact.artifact_id,
                task=assignment.task,
                attempt_number=assignment.attempt_number,
                assignment_token=assignment.assignment_token,
                phase_recorder=self.lifecycle.phase_recorder,
            )
            self.lifecycle.issued = issued
            issued = self.lifecycle.runner._begin_session_attempt(issued.session.session_id)
            self.lifecycle.issued = issued
            attempt_started_at = self.lifecycle.runner._clock()
            self.lifecycle.claimed.mark_started(
                issued.session.session_id,
                started_at=attempt_started_at,
            )
            return _IssuedAssignedAttempt(
                assignment=assignment,
                issued=issued,
                attempt_started_at=attempt_started_at,
                session_started_at=session_started_at,
                phase_recorder=self.lifecycle.phase_recorder,
            )
        except Exception as exc:
            if isinstance(exc, _PartialSessionIssueError):
                self.lifecycle.issued = exc.issued
                exc = exc.cause
            result = self.lifecycle.build_failure_result(
                exc=exc,
                attempt_started_at=attempt_started_at,
            )
            if self.lifecycle.issued is None:
                published = self.lifecycle.claimed.fail_before_start(result)
                raise _AssignmentStartFailedError(
                    _AssignedAttemptRunOutcome(
                        result_to_queue=result if published else None,
                        result_already_queued=published,
                    )
                ) from exc
            self.lifecycle.runner._cleanup_attempt_session_best_effort(
                session_id=self.lifecycle.issued.session.session_id,
                batch_id=self.lifecycle.batch_id,
                artifact_id=self.lifecycle.artifact.artifact_id,
                task_id=assignment.task.task_id,
            )
            published = self.lifecycle.claimed.fail_before_start(result)
            raise _AssignmentStartFailedError(
                _AssignedAttemptRunOutcome(
                    result_to_queue=result if published else None,
                    result_already_queued=published,
                )
            ) from exc

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
        issued = self.lifecycle.issued
        if issued is not None:
            self.lifecycle.runner._cleanup_attempt_session_best_effort(
                session_id=issued.session.session_id,
                batch_id=self.lifecycle.batch_id,
                artifact_id=self.lifecycle.artifact.artifact_id,
                task_id=self.lifecycle.claimed.assignment.task.task_id,
            )
        return False


@dataclass(frozen=True, slots=True)
class ArtifactFailure:
    error_code: MinerTaskErrorCode
    message: str
    failure_detail: ValidatorBatchFailureDetail
    artifact_breaker_tripped: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactEvaluationOutcome:
    submissions: tuple[MinerTaskRunSubmission, ...]
    artifact_failure: ArtifactFailure | None = None


class EvaluationRunner:
    """Executes miner task runs for artifacts and records submissions."""

    def __init__(
        self,
        *,
        subtensor_client: SubtensorClientPort,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        receipt_log: ReceiptLogPort,
        config: SchedulerConfig,
        clock: Clock,
        progress: ProgressRecorder,
        usage_summarizer: UsageSummarizer | None = None,
        platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry | None = None,
    ) -> None:
        _ = subtensor_client
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._receipts = receipt_log
        self._config = config
        self._clock = clock
        self._progress = progress
        self._usage = usage_summarizer or UsageSummarizer()
        self._platform_tool_proxy_scopes = platform_tool_proxy_scopes

    async def evaluate_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator: TaskRunOrchestrator,
    ) -> ArtifactEvaluationOutcome:
        return await self.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=tasks,
            orchestrator=orchestrator,
        )

    async def evaluate_assigned_task(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        attempt_number: int,
        max_attempts: int,
        assignment_token: str,
        orchestrator: TaskRunOrchestrator,
        on_session_started: SessionStartedCallback | None = None,
    ) -> PlatformOwnedTaskResult:
        session_started_at = time.monotonic()
        issued: SessionIssued | None = None
        attempt_started_at = self._clock()
        try:
            issued = self._issue_session(
                batch_id=batch_id,
                uid=artifact.uid,
                artifact_id=artifact.artifact_id,
                task=task,
                attempt_number=attempt_number,
                assignment_token=assignment_token,
            )
            issued = self._begin_session_attempt(issued.session.session_id)
            if on_session_started is not None:
                on_session_started(issued.session.session_id)
            attempt_started_at = self._clock()
        except Exception as exc:
            if isinstance(exc, _PartialSessionIssueError):
                issued = exc.issued
                exc = exc.cause
            result = self._assigned_task_unexpected_failure_result(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=attempt_started_at,
                exc=exc,
            )
            _log_session_finished(
                batch_id=batch_id,
                session_id=issued.session.session_id if issued is not None else UUID(int=0),
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
                uid=artifact.uid,
                attempt_count=attempt_number,
                session_ms=_monotonic_elapsed_ms(
                    started_at=session_started_at,
                    completed_at=time.monotonic(),
                ),
                terminal_outcome=AttemptControlKind.VALIDATOR_BATCH_FAILURE.value,
                error_code=str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE),
            )
            if issued is not None:
                self._cleanup_attempt_session_best_effort(
                    session_id=issued.session.session_id,
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                )
            return result

        return await self._evaluate_issued_assigned_task(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            issued=issued,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            started_at=attempt_started_at,
            session_started_at=session_started_at,
            orchestrator=orchestrator,
        )

    async def _evaluate_issued_assigned_task(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        session_started_at: float,
        orchestrator: TaskRunOrchestrator,
        phase_recorder: PhaseRecorder | None = None,
        cleanup_on_exit: bool = True,
    ) -> PlatformOwnedTaskResult:
        terminal_outcome = "unexpected"
        error_code: str | None = None
        try:
            decision = await self._evaluate_task_attempt(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                orchestrator=orchestrator,
                final_attempt=attempt_number >= max_attempts,
                phase_recorder=phase_recorder,
            )
            finalization = self._finalize_assigned_task_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                decision=decision,
                phase_recorder=phase_recorder,
            )
            terminal_outcome = finalization.terminal_outcome
            error_code = finalization.error_code
            self._record_assigned_task_local_side_effects_best_effort(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                result=finalization.result,
                local_submission=finalization.local_submission,
            )
            return finalization.result
        except Exception as exc:
            terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
            error_code = str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)
            return self._assigned_task_unexpected_failure_result(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                exc=exc,
                diagnostics=(
                    phase_recorder.diagnostics(failure_owner=_exception_type_name(exc))
                    if isinstance(phase_recorder, _AssignedAttemptPhaseRecorder)
                    else None
                ),
            )
        finally:
            _log_session_finished(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
                uid=artifact.uid,
                attempt_count=attempt_number,
                session_ms=_monotonic_elapsed_ms(
                    started_at=session_started_at,
                    completed_at=time.monotonic(),
                ),
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )
            if cleanup_on_exit:
                self._cleanup_attempt_session_best_effort(
                    session_id=issued.session.session_id,
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                )

    async def evaluate_assigned_task_queue(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        initial_assignments: Sequence[MinerTaskWorkAssignment],
        assigned_work: AssignedArtifactWork,
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult | PlatformOwnedTaskExecution],
        orchestrator: TaskRunOrchestrator,
        sandbox_client: SandboxClient,
        sandbox_identifier: str | None = None,
    ) -> QueueStopReason:
        """Finish owned work while keeping cleanup retirement distinct from positive OOM."""
        active: dict[asyncio.Task[_AssignedAttemptRunOutcome], _ActiveAssignedTask] = {}
        queue_waiter: asyncio.Task[ClaimedAssignedTask] | None = None
        claim_to_start: ClaimedAssignedTask | None = None
        drain_logged = False
        close_waiter = asyncio.create_task(close_requested.wait())
        monitor_stop = asyncio.Event()

        async def watch_resources() -> None:
            while not monitor_stop.is_set() and not sandbox_client.draining:
                status = await sandbox_client.resource_status()
                if monitor_stop.is_set():
                    return
                if status.counter_status == "disabled":
                    await monitor_stop.wait()
                    return
                if not sandbox_client.draining:
                    try:
                        await asyncio.wait_for(monitor_stop.wait(), timeout=0.1)
                    except TimeoutError:
                        pass

        oom_waiter = asyncio.create_task(watch_resources())

        def fail_unstarted_claim(claimed: ClaimedAssignedTask, exc: Exception) -> bool:
            assignment = claimed.assignment
            result = self._assigned_task_unexpected_failure_result(
                batch_id=batch_id,
                artifact=artifact,
                task=assignment.task,
                issued=None,
                attempt_number=assignment.attempt_number,
                max_attempts=assignment.max_attempts,
                started_at=self._clock(),
                exc=exc,
            )
            published = claimed.fail_before_start(result)
            return (
                published and result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
            )

        def start_assignment(claimed: ClaimedAssignedTask) -> None:
            if sandbox_client.draining:
                claimed.release_to_queue_if_live()
                return
            assignment = claimed.assignment
            lifecycle = self._assigned_attempt_lifecycle(
                batch_id=batch_id,
                artifact=artifact,
                claimed=claimed,
            )
            execution_task = asyncio.create_task(
                self._run_admitted_assignment(lifecycle, orchestrator=orchestrator, sandbox_client=sandbox_client)
            )
            active[execution_task] = _ActiveAssignedTask(
                assignment=assignment,
                lifecycle=lifecycle,
            )

        async def cancel_queue_waiter() -> None:
            nonlocal queue_waiter
            if queue_waiter is None:
                return
            if not queue_waiter.done():
                queue_waiter.cancel()
            await asyncio.gather(queue_waiter, return_exceptions=True)
            if not queue_waiter.cancelled() and queue_waiter.exception() is None:
                queue_waiter.result().release_to_queue_if_live()
            queue_waiter = None

        try:
            for assignment in initial_assignments:
                claim_to_start = assigned_work.claim_initial_for_dispatch(assignment)
                if claim_to_start is None:
                    continue
                start_assignment(claim_to_start)
                claim_to_start = None

            while active or not close_requested.is_set():
                if sandbox_client.cleanup_retirement_required:
                    assigned_work.close_intake()
                    close_requested.set()
                    await cancel_queue_waiter()
                    # Started siblings are never replayed or relabeled as OOM.
                    # Existing cancellation/uncertainty handling resolves their attempts.
                    for task in active:
                        if not task.done():
                            task.cancel()
                    if active:
                        done, _ = await asyncio.wait(active)
                        await self._drain_assigned_results(
                            active=active,
                            done=done,
                            batch_id=batch_id,
                            artifact=artifact,
                            result_queue=result_queue,
                            oom_draining=False,
                            cleanup_retiring=True,
                        )
                    return QueueStopReason.CLEANUP_UNCONFIRMED
                if sandbox_client.draining:
                    if not drain_logged:
                        logger.warning(
                            "sandbox OOM draining started",
                            extra={
                                "batch_id": str(batch_id),
                                "artifact_id": str(artifact.artifact_id),
                                "container": sandbox_identifier,
                                "active_attempts": [
                                    {
                                        "task_id": str(owned.assignment.task.task_id),
                                        "session_id": (
                                            str(owned.lifecycle.issued.session.session_id)
                                            if owned.lifecycle is not None and owned.lifecycle.issued is not None
                                            else None
                                        ),
                                        "compiler_generation": (
                                            owned.lifecycle.admission.generation
                                            if owned.lifecycle is not None and owned.lifecycle.admission is not None
                                            else None
                                        ),
                                    }
                                    for owned in active.values()
                                ],
                            },
                        )
                        drain_logged = True
                    await cancel_queue_waiter()
                    for task, owned in active.items():
                        if owned.lifecycle is not None and owned.lifecycle.issued is None:
                            task.cancel()
                    if not active:
                        return QueueStopReason.OOM_DRAINED
                while not close_requested.is_set() and not sandbox_client.draining:
                    try:
                        claim_to_start = assigned_work.claim_nowait_for_dispatch()
                        start_assignment(claim_to_start)
                        claim_to_start = None
                    except asyncio.QueueEmpty:
                        break

                wait_for: set[asyncio.Task[object]] = set(active)
                if not sandbox_client.draining:
                    wait_for.add(oom_waiter)
                if not close_requested.is_set() and not sandbox_client.draining:
                    if queue_waiter is None:
                        queue_waiter = asyncio.create_task(assigned_work.claim_for_dispatch())
                    wait_for.add(queue_waiter)
                    wait_for.add(close_waiter)
                elif not active:
                    return QueueStopReason.OOM_DRAINED if sandbox_client.draining else QueueStopReason.COMPLETED
                done, _pending = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                if oom_waiter in done:
                    oom_waiter.result()
                if await self._drain_assigned_results(
                    active=active,
                    done=done,
                    batch_id=batch_id,
                    artifact=artifact,
                    result_queue=result_queue,
                    oom_draining=sandbox_client.draining,
                ):
                    close_requested.set()
                    await cancel_queue_waiter()
                    continue

                if queue_waiter is not None and queue_waiter in done:
                    claim_to_start = queue_waiter.result()
                    start_assignment(claim_to_start)
                    claim_to_start = None
                    queue_waiter = None
            return QueueStopReason.OOM_DRAINED if sandbox_client.draining else QueueStopReason.COMPLETED
        finally:
            # HTTP connection cleanup can consume cancellation; stop cooperatively too.
            monitor_stop.set()
            if claim_to_start is not None and sandbox_client.draining:
                claim_to_start.release_to_queue_if_live()
                claim_to_start = None
            if claim_to_start is not None:
                fail_unstarted_claim(
                    claim_to_start,
                    RuntimeError("assigned work cancelled before validator session start"),
                )
            if queue_waiter is not None and not queue_waiter.done():
                queue_waiter.cancel()
            await cancel_queue_waiter()
            close_waiter.cancel()
            oom_waiter.cancel()
            for task in active:
                task.cancel()
            await asyncio.gather(
                *(task for task in (queue_waiter, close_waiter, oom_waiter) if task is not None),
                return_exceptions=True,
            )
            if active:
                await asyncio.gather(*active, return_exceptions=True)

    async def _drain_assigned_results(
        self,
        *,
        active: dict[asyncio.Task[_AssignedAttemptRunOutcome], _ActiveAssignedTask],
        done: set[asyncio.Task[object]],
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult | PlatformOwnedTaskExecution],
        oom_draining: bool,
        cleanup_retiring: bool = False,
    ) -> bool:
        delivery_failure_seen = False
        for task in tuple(active):
            if task not in done:
                continue
            active_task = active.pop(task)
            try:
                outcome = task.result()
            except asyncio.CancelledError:
                if cleanup_retiring:
                    exc = SandboxInvokeError(
                        "sandbox retired after cleanup uncertainty",
                        status_code=503,
                        detail_code="SandboxCleanupUnconfirmed",
                        detail_exception=None,
                        detail_error=None,
                        remote_state_uncertain=True,
                    )
                    result = (
                        active_task.lifecycle.build_failure_result(exc=exc, attempt_started_at=self._clock())
                        if active_task.lifecycle is not None
                        else self._platform_result_from_assignment_task_exception(
                            batch_id=batch_id,
                            artifact=artifact,
                            assignment=active_task.assignment,
                            exc=exc,
                        )
                    )
                    outcome = _AssignedAttemptRunOutcome(result_to_queue=result)
                elif not oom_draining:
                    raise
                else:
                    continue
            except Exception as exc:
                result = self._platform_result_from_assignment_task_exception(
                    batch_id=batch_id,
                    artifact=artifact,
                    assignment=active_task.assignment,
                    exc=exc,
                )
                outcome = _AssignedAttemptRunOutcome(result_to_queue=result)
            if outcome.execution_to_queue is not None:
                await result_queue.put(outcome.execution_to_queue)
            result = outcome.result_to_queue
            if oom_draining:
                logger.info(
                    "sandbox OOM attempt settled",
                    extra={
                        "batch_id": str(batch_id),
                        "artifact_id": str(artifact.artifact_id),
                        "task_id": str(active_task.assignment.task.task_id),
                        "session_id": (
                            str(active_task.lifecycle.issued.session.session_id)
                            if active_task.lifecycle is not None and active_task.lifecycle.issued is not None
                            else None
                        ),
                        "compiler_generation": (
                            active_task.lifecycle.admission.generation
                            if active_task.lifecycle is not None and active_task.lifecycle.admission is not None
                            else None
                        ),
                        "execution_preserved": outcome.execution_to_queue is not None,
                        "result_preserved": result is not None
                        and result.terminal_attempt.status is MinerTaskAttemptStatus.SUCCEEDED,
                        "error_code": result.terminal_attempt.error_code if result is not None else None,
                    },
                )
            if result is None:
                continue
            if result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE:
                delivery_failure_seen = True
            if outcome.result_already_queued:
                continue
            await result_queue.put(result)
        return delivery_failure_seen

    def _assigned_attempt_lifecycle(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        claimed: ClaimedAssignedTask,
    ) -> _AssignedAttemptLifecycle:
        return _AssignedAttemptLifecycle(
            runner=self,
            batch_id=batch_id,
            artifact=artifact,
            claimed=claimed,
            phase_recorder=_AssignedAttemptPhaseRecorder(started_at=time.monotonic()),
        )

    async def _run_admitted_assignment(
        self,
        lifecycle: _AssignedAttemptLifecycle,
        *,
        orchestrator: TaskRunOrchestrator,
        sandbox_client: SandboxClient,
    ) -> _AssignedAttemptRunOutcome:
        claimed = lifecycle.claimed
        try:
            for _ in range(TASK_START_WAIT_ATTEMPTS):
                admission_wait = asyncio.timeout(min(claimed.remaining_dispatch_seconds(), TASK_START_WAIT_SECONDS))
                try:
                    async with admission_wait:
                        while claimed.is_live() and not sandbox_client.draining:
                            try:
                                async with sandbox_client.admission(
                                    self._config.execution_limit_for(claimed.assignment.task),
                                    claimed.assignment.assignment_token,
                                ) as admission:
                                    lifecycle.admission = admission
                                    if not claimed.is_live() or sandbox_client.draining:
                                        return _AssignedAttemptRunOutcome()
                                    # The waiting deadline must not shorten a started execution.
                                    admission_wait.reschedule(None)
                                    return await self._run_assigned_attempt(
                                        lifecycle, orchestrator=orchestrator, admission=admission
                                    )
                            except SandboxInvokeError as exc:
                                if (
                                    lifecycle.admission is not None
                                    or exc.remote_state_uncertain
                                    or exc.status_code != 504
                                    or exc.detail_exception != "TimeoutError"
                                    or exc.detail_error != "admission deadline exceeded"
                                ):
                                    raise
                            # An early admission timeout has not spent this waiting window.
                            # Pace retries even if the sandbox responds immediately.
                            await asyncio.sleep(0.1)
                except TimeoutError:
                    if not admission_wait.expired():
                        raise
                if sandbox_client.draining:
                    return _AssignedAttemptRunOutcome()
                if not claimed.is_live():
                    break
            result = task_start_timeout_result(claimed.assignment, self._clock())
            published = claimed.fail_before_start(result)
            return _AssignedAttemptRunOutcome(
                result_to_queue=result if published else None, result_already_queued=published
            )
        except Exception as exc:
            if sandbox_client.draining:
                return _AssignedAttemptRunOutcome()
            result = lifecycle.build_failure_result(exc=exc, attempt_started_at=self._clock())
            published = claimed.fail_before_start(result)
            return _AssignedAttemptRunOutcome(
                result_to_queue=result if published else None, result_already_queued=published
            )
        finally:
            claimed.release_to_queue_if_live()

    async def _run_assigned_attempt(
        self,
        lifecycle: _AssignedAttemptLifecycle,
        *,
        orchestrator: TaskRunOrchestrator,
        admission: SandboxAdmission | None = None,
    ) -> _AssignedAttemptRunOutcome:
        try:
            async with lifecycle.start() as attempt:
                try:
                    if "_evaluate_issued_assigned_task" in self.__dict__:
                        result = await self._evaluate_issued_assigned_task(
                            batch_id=lifecycle.batch_id,
                            artifact=lifecycle.artifact,
                            task=attempt.assignment.task,
                            issued=attempt.issued,
                            attempt_number=attempt.assignment.attempt_number,
                            max_attempts=attempt.assignment.max_attempts,
                            started_at=attempt.attempt_started_at,
                            session_started_at=attempt.session_started_at,
                            orchestrator=orchestrator,
                            phase_recorder=attempt.phase_recorder,
                            cleanup_on_exit=False,
                        )
                        return _AssignedAttemptRunOutcome(result_to_queue=result)
                    return await self._execute_issued_assigned_task(
                        batch_id=lifecycle.batch_id,
                        artifact=lifecycle.artifact,
                        task=attempt.assignment.task,
                        issued=attempt.issued,
                        attempt_number=attempt.assignment.attempt_number,
                        max_attempts=attempt.assignment.max_attempts,
                        started_at=attempt.attempt_started_at,
                        session_started_at=attempt.session_started_at,
                        orchestrator=orchestrator,
                        phase_recorder=attempt.phase_recorder,
                        admission=admission,
                    )
                except Exception as exc:
                    result = lifecycle.build_failure_result(
                        exc=exc,
                        attempt_started_at=attempt.attempt_started_at,
                    )
                return _AssignedAttemptRunOutcome(result_to_queue=result)
        except _AssignmentStartFailedError as exc:
            return exc.outcome

    async def _execute_issued_assigned_task(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        session_started_at: float,
        orchestrator: TaskRunOrchestrator,
        phase_recorder: PhaseRecorder | None = None,
        admission: SandboxAdmission | None = None,
    ) -> _AssignedAttemptRunOutcome:
        terminal_outcome = "execution_completed"
        error_code: str | None = None
        try:
            execution_or_decision = await self._execute_task_attempt(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                orchestrator=orchestrator,
                final_attempt=attempt_number >= max_attempts,
                phase_recorder=phase_recorder,
                admission=admission,
            )
            if isinstance(execution_or_decision, TaskAttemptDecision):
                finalization = self._finalize_assigned_task_decision(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    issued=issued,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    started_at=started_at,
                    decision=execution_or_decision,
                    phase_recorder=phase_recorder,
                )
                terminal_outcome = finalization.terminal_outcome
                error_code = finalization.error_code
                self._record_assigned_task_local_side_effects_best_effort(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    result=finalization.result,
                    local_submission=finalization.local_submission,
                )
                return _AssignedAttemptRunOutcome(result_to_queue=finalization.result)

            execution = self._build_platform_execution(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                outcome=execution_or_decision,
            )
            return _AssignedAttemptRunOutcome(execution_to_queue=execution)
        except Exception as exc:
            terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
            error_code = str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)
            result = self._assigned_task_unexpected_failure_result(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                exc=exc,
                diagnostics=(
                    phase_recorder.diagnostics(failure_owner=_exception_type_name(exc))
                    if isinstance(phase_recorder, _AssignedAttemptPhaseRecorder)
                    else None
                ),
            )
            return _AssignedAttemptRunOutcome(result_to_queue=result)
        finally:
            _log_session_finished(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
                uid=artifact.uid,
                attempt_count=attempt_number,
                session_ms=_monotonic_elapsed_ms(
                    started_at=session_started_at,
                    completed_at=time.monotonic(),
                ),
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )
            self._cleanup_attempt_session_best_effort(
                session_id=issued.session.session_id,
                batch_id=batch_id,
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
            )

    async def record_assigned_task_setup_failures(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        assignments: Sequence[MinerTaskWorkAssignment],
        error_code: MinerTaskErrorCode,
        error_message: str,
    ) -> tuple[PlatformOwnedTaskResult, ...]:
        results: list[PlatformOwnedTaskResult] = []
        for assignment in assignments:
            attempt_started_at = self._clock()
            issued: SessionIssued | None = None
            try:
                try:
                    issued = self._begin_session_attempt(
                        self._issue_session(
                            batch_id=batch_id,
                            uid=artifact.uid,
                            artifact_id=artifact.artifact_id,
                            task=assignment.task,
                            attempt_number=assignment.attempt_number,
                            assignment_token=assignment.assignment_token,
                        ).session.session_id
                    )
                    submission = self._build_task_failure(
                        batch_id=batch_id,
                        task=assignment.task,
                        artifact=artifact,
                        session_id=issued.session.session_id,
                        error_code=error_code,
                        error_message=error_message,
                        log_message="assigned miner task setup failed",
                        exc=RuntimeError(error_message),
                    )
                    decision = _submission_decision(submission)
                    attempt = self._build_terminated_attempt_record(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=assignment.task,
                        issued=issued,
                        attempt_number=assignment.attempt_number,
                        max_attempts=assignment.max_attempts,
                        started_at=attempt_started_at,
                        decision=decision,
                        retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                        terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
                    )
                    result = PlatformOwnedTaskResult(
                        batch_id=batch_id,
                        artifact_id=artifact.artifact_id,
                        task_id=assignment.task.task_id,
                        attempt_number=assignment.attempt_number,
                        result=submission,
                        terminal_attempt=attempt,
                    )
                    self._record_assigned_task_local_side_effects_best_effort(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=assignment.task,
                        result=result,
                        local_submission=submission,
                    )
                except Exception as exc:
                    if isinstance(exc, _PartialSessionIssueError):
                        issued = exc.issued
                        exc = exc.cause
                    logger.exception(
                        "assigned miner task setup-failure result construction failed unexpectedly",
                        extra={
                            "batch_id": str(batch_id),
                            "artifact_id": str(artifact.artifact_id),
                            "task_id": str(assignment.task.task_id),
                            "attempt_number": assignment.attempt_number,
                        },
                    )
                    result = self._platform_result_from_assigned_unexpected_failure(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=assignment.task,
                        issued=issued,
                        attempt_number=assignment.attempt_number,
                        max_attempts=assignment.max_attempts,
                        started_at=attempt_started_at,
                        exc=exc,
                    )
                    self._record_assigned_task_local_side_effects_best_effort(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=assignment.task,
                        result=result,
                        local_submission=None,
                    )
                results.append(result)
            finally:
                if issued is not None:
                    self._cleanup_attempt_session_best_effort(
                        session_id=issued.session.session_id,
                        batch_id=batch_id,
                        artifact_id=artifact.artifact_id,
                        task_id=assignment.task.task_id,
                    )
        return tuple(results)

    def _platform_result_from_assignment_task_exception(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        assignment: MinerTaskWorkAssignment,
        exc: Exception,
    ) -> PlatformOwnedTaskResult:
        logger.exception(
            "assigned miner task future failed unexpectedly",
            extra={
                "batch_id": str(batch_id),
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(assignment.task.task_id),
                "attempt_number": assignment.attempt_number,
            },
        )
        return self._platform_result_from_assigned_unexpected_failure(
            batch_id=batch_id,
            artifact=artifact,
            task=assignment.task,
            issued=None,
            attempt_number=assignment.attempt_number,
            max_attempts=assignment.max_attempts,
            started_at=self._clock(),
            exc=exc,
        )

    def _assigned_task_unexpected_failure_result(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued | None,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        exc: Exception,
        diagnostics: MinerTaskAttemptDiagnostics | None = None,
    ) -> PlatformOwnedTaskResult:
        logger.exception(
            "assigned miner task failed unexpectedly",
            extra={
                "batch_id": str(batch_id),
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
                "attempt_number": attempt_number,
            },
        )
        result = self._platform_result_from_assigned_unexpected_failure(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            issued=issued,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            started_at=started_at,
            exc=exc,
            diagnostics=diagnostics,
        )
        self._record_assigned_task_local_side_effects_best_effort(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            result=result,
            local_submission=None,
        )
        return result

    def _platform_result_from_assigned_unexpected_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued | None,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        exc: Exception,
        diagnostics: MinerTaskAttemptDiagnostics | None = None,
    ) -> PlatformOwnedTaskResult:
        session_id = issued.session.session_id if issued is not None else uuid4()
        execution_log = tuple(self._receipts.for_session(session_id)) if issued is not None else ()
        error_code = str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)
        retry_decision = MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
        terminal_effect = MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
        if issued is not None:
            if attempt_number < max_attempts:
                retry_decision = MinerTaskAttemptRetryDecision.WILL_RETRY
                terminal_effect = None
            else:
                terminal_effect = MinerTaskAttemptTerminalEffect.ATTEMPT_FAILURE
        attempt = MinerTaskAttemptAuditRecord(
            validator_session_id=session_id,
            batch_id=batch_id,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            uid=artifact.uid,
            miner_hotkey_ss58=artifact.miner_hotkey_ss58 or "unknown-miner-hotkey",
            started_at=started_at,
            finished_at=self._clock(),
            status=MinerTaskAttemptStatus.FAILED,
            error_code=error_code,
            error_summary_code=error_code,
            retry_decision=retry_decision,
            terminal_effect=terminal_effect,
            max_attempts=max_attempts,
            execution_log=execution_log,
            diagnostics=diagnostics,
        )
        logger.debug(
            "built assigned miner task unexpected-failure result",
            extra={
                "batch_id": str(batch_id),
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
                "attempt_number": attempt_number,
                "error_type": type(exc).__name__,
            },
        )
        return PlatformOwnedTaskResult(
            batch_id=batch_id,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            result=None,
            terminal_attempt=attempt,
        )

    async def evaluate_artifact_with_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator: TaskRunOrchestrator,
        earlier_submissions: tuple[MinerTaskRunSubmission, ...] = (),
        task_session_limiter: TaskSessionLimiter | None = None,
    ) -> ArtifactEvaluationOutcome:
        indexed_tasks = tuple(enumerate(tasks))
        if not indexed_tasks:
            return ArtifactEvaluationOutcome(
                submissions=earlier_submissions,
            )

        dispatch = _ArtifactDispatchState(
            submissions_by_index=[None] * len(indexed_tasks),
        )
        pending_tasks: asyncio.Queue[tuple[int, MinerTask]] = asyncio.Queue()
        for indexed_task in indexed_tasks:
            pending_tasks.put_nowait(indexed_task)

        workers = [
            asyncio.create_task(
                self._run_artifact_worker(
                    batch_id=batch_id,
                    artifact=artifact,
                    orchestrator=orchestrator,
                    pending_tasks=pending_tasks,
                    dispatch=dispatch,
                    task_session_limiter=task_session_limiter,
                )
            )
            for _ in range(
                min(
                    max(1, self._config.artifact_task_parallelism),
                    len(indexed_tasks),
                )
            )
        ]
        await asyncio.gather(*workers)
        completed_submissions = tuple(
            submission for submission in dispatch.submissions_by_index if submission is not None
        )
        all_completed_submissions = (*earlier_submissions, *completed_submissions)
        if dispatch.validator_failure is not None:
            remaining_tasks = tuple(
                task for index, task in indexed_tasks if dispatch.submissions_by_index[index] is None
            )
            raise ValidatorBatchFailedError(
                error_code=dispatch.validator_failure.error_code,
                message=str(dispatch.validator_failure),
                failure_detail=dispatch.validator_failure.failure_detail,
                completed_submissions=all_completed_submissions,
                remaining_tasks=remaining_tasks,
            ) from dispatch.validator_failure
        if dispatch.unexpected_failure is not None:
            remaining_tasks = tuple(
                task for index, task in indexed_tasks if dispatch.submissions_by_index[index] is None
            )
            raise UnexpectedArtifactExecutionError(
                cause=dispatch.unexpected_failure,
                completed_submissions=all_completed_submissions,
                remaining_tasks=remaining_tasks,
            ) from dispatch.unexpected_failure

        return ArtifactEvaluationOutcome(
            submissions=all_completed_submissions,
        )

    async def _run_artifact_worker(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        orchestrator: TaskRunOrchestrator,
        pending_tasks: asyncio.Queue[tuple[int, MinerTask]],
        dispatch: _ArtifactDispatchState,
        task_session_limiter: TaskSessionLimiter | None = None,
    ) -> None:
        while True:
            if dispatch.validator_failure is not None or dispatch.unexpected_failure is not None:
                return
            try:
                task_index, task = pending_tasks.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                async with _limited_task_session(task_session_limiter):
                    decision = await self._evaluate_task_with_retry(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        orchestrator=orchestrator,
                    )
                if decision.kind is AttemptControlKind.SUBMISSION:
                    submission = _require_submission(decision)
                    dispatch.submissions_by_index[task_index] = submission
                    error_code = _submission_error_code_or_none(submission)
                    if (
                        error_code is not None
                        and is_delivery_disqualifying_validator_pair_error(error_code)
                        and dispatch.validator_failure is None
                    ):
                        dispatch.validator_failure = _validator_batch_failed_from_existing_submission(
                            submission=submission,
                            artifact=artifact,
                            task=task,
                            occurred_at=self._clock(),
                        )
                    continue

                if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                    if dispatch.validator_failure is None:
                        dispatch.validator_failure = _require_validator_failure(decision)
                    continue

                raise RuntimeError("unexpected non-terminal decision from task retry loop")
            except ValidatorBatchFailedError as exc:
                if exc.completed_submissions:
                    dispatch.submissions_by_index[task_index] = _require_single_completed_submission(exc)
                if dispatch.validator_failure is None:
                    dispatch.validator_failure = exc
            except Exception as exc:
                if dispatch.unexpected_failure is None:
                    dispatch.unexpected_failure = exc

    async def _run_tasks_with_sessions(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        create_submission: SubmissionFactory,
    ) -> list[MinerTaskRunSubmission]:
        submissions: list[MinerTaskRunSubmission] = []
        for index, task in enumerate(tasks):
            issued = self._issue_session(
                batch_id=batch_id,
                uid=artifact.uid,
                artifact_id=artifact.artifact_id,
                task=task,
            )
            try:
                submissions.append(await create_submission(task, issued))
            except Exception as exc:
                raise UnexpectedArtifactExecutionError(
                    cause=exc,
                    completed_submissions=tuple(submissions),
                    remaining_tasks=tuple(tasks[index:]),
                ) from exc
            finally:
                self._clear_task_session(issued.session.session_id)
                self._clear_platform_tool_proxy_session(issued.session.session_id)
                self._sessions.revoke(issued.session.session_id)
                self._receipts.clear_session(issued.session.session_id)
        return submissions

    async def _evaluate_task_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        orchestrator: TaskRunOrchestrator,
    ) -> TaskAttemptDecision:
        session_started_at = time.monotonic()
        attempt_count = 0
        terminal_outcome = "unexpected"
        error_code: str | None = None
        issued: SessionIssued | None = None
        try:
            max_attempts = 1 + artifact.task_retry_count
            attempt_number = self._progress.next_attempt_number(
                batch_id,
                artifact.artifact_id,
                task.task_id,
            )
            if attempt_number > max_attempts:
                return _validator_batch_failure_decision(
                    ValidatorBatchFailedError(
                        error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                        message="next attempt number exceeds configured retry budget",
                        failure_detail=ValidatorBatchFailureDetail(
                            error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                            error_message="next attempt number exceeds configured retry budget",
                            occurred_at=self._clock(),
                            artifact_id=artifact.artifact_id,
                            task_id=task.task_id,
                            uid=artifact.uid,
                            exception_type=None,
                        ),
                    )
                )
            while attempt_number <= max_attempts:
                attempt_count = attempt_number
                issued = self._begin_session_attempt(
                    self._issue_session(
                        batch_id=batch_id,
                        uid=artifact.uid,
                        artifact_id=artifact.artifact_id,
                        task=task,
                        attempt_number=attempt_number,
                    ).session.session_id
                )
                attempt_started_at = self._clock()
                decision = await self._evaluate_task_attempt(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    issued=issued,
                    orchestrator=orchestrator,
                    final_attempt=attempt_number >= max_attempts,
                )
                if decision.kind is AttemptControlKind.SUBMISSION:
                    terminal_outcome = AttemptControlKind.SUBMISSION.value
                    submission = _require_submission(decision)
                    error_code = _submission_error_code_or_none(submission)
                    self._record_submission(submission)
                    self._record_terminated_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        issued=issued,
                        attempt_number=attempt_number,
                        max_attempts=max_attempts,
                        started_at=attempt_started_at,
                        decision=decision,
                        retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                        terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
                    )
                    return decision

                if decision.kind is AttemptControlKind.REVIEW_TIMEOUT:
                    timeout_exc = _require_timeout_exc(decision)
                    if attempt_number >= max_attempts:
                        timeout_resolution = self._record_terminal_timeout_submission(
                            batch_id=batch_id,
                            artifact=artifact,
                            task=task,
                            issued=issued,
                            timeout_exc=timeout_exc,
                        )
                        terminal_outcome = AttemptControlKind.SUBMISSION.value
                        submission = _require_submission(timeout_resolution)
                        error_code = _submission_error_code_or_none(submission)
                        self._record_terminated_attempt(
                            batch_id=batch_id,
                            artifact=artifact,
                            task=task,
                            issued=issued,
                            attempt_number=attempt_number,
                            max_attempts=max_attempts,
                            started_at=attempt_started_at,
                            decision=timeout_resolution,
                            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                            terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
                        )
                        return timeout_resolution
                    retry_decision = _review_timeout_decision(
                        timeout_exc,
                        attempt_error_code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                    )
                    self._log_retry_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        attempt_number=attempt_number,
                        exc=timeout_exc,
                    )
                    self._record_terminated_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        issued=issued,
                        attempt_number=attempt_number,
                        max_attempts=max_attempts,
                        started_at=attempt_started_at,
                        decision=retry_decision,
                        retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
                        terminal_effect=None,
                    )
                    self._cleanup_attempt_session(issued.session.session_id)
                    attempt_number = self._progress.next_attempt_number(
                        batch_id,
                        artifact.artifact_id,
                        task.task_id,
                    )
                    continue

                if decision.kind is AttemptControlKind.RETRY:
                    self._log_retry_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        attempt_number=attempt_number,
                        exc=_require_retry_exc(decision),
                    )
                    self._record_terminated_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        issued=issued,
                        attempt_number=attempt_number,
                        max_attempts=max_attempts,
                        started_at=attempt_started_at,
                        decision=decision,
                        retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
                        terminal_effect=None,
                    )
                    self._cleanup_attempt_session(issued.session.session_id)
                    attempt_number = self._progress.next_attempt_number(
                        batch_id,
                        artifact.artifact_id,
                        task.task_id,
                    )
                    continue

                if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                    validator_failure = _require_validator_failure(decision)
                    terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
                    error_code = str(validator_failure.error_code)
                    self._record_terminated_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        issued=issued,
                        attempt_number=attempt_number,
                        max_attempts=max_attempts,
                        started_at=attempt_started_at,
                        decision=decision,
                        retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                        terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
                        delivery_failure_detail=validator_failure.failure_detail,
                    )
                    raise validator_failure

                raise RuntimeError("task retry loop returned unexpected decision")
        finally:
            _log_session_finished(
                batch_id=batch_id,
                session_id=issued.session.session_id if issued is not None else UUID(int=0),
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
                uid=artifact.uid,
                attempt_count=attempt_count,
                session_ms=_monotonic_elapsed_ms(
                    started_at=session_started_at,
                    completed_at=time.monotonic(),
                ),
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )
            if issued is not None:
                self._cleanup_attempt_session(issued.session.session_id)

        raise RuntimeError("task retry loop exited without returning")

    async def _evaluate_task_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        orchestrator: TaskRunOrchestrator,
        final_attempt: bool,
        phase_recorder: PhaseRecorder | None = None,
    ) -> TaskAttemptDecision:
        request = MinerTaskRunRequest(
            batch_id=batch_id,
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            execution_time_limit_seconds=self._config.execution_limit_for(task),
        )
        try:
            if phase_recorder is None or not _orchestrator_accepts_phase_recorder(orchestrator):
                outcome = await orchestrator.evaluate(request)
            else:
                outcome = await orchestrator.evaluate(request, phase_recorder=phase_recorder)
        except SessionBudgetExhaustedError as exc:
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            return _submission_decision(
                self._build_exhausted(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=issued.session.session_id,
                    error_message=str(exc),
                )
            )
        except SandboxInvocationError as exc:
            timeout_decision = self._platform_tool_proxy_timeout_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )
            if timeout_decision is not None:
                return timeout_decision
            if is_timeout_sandbox_invocation(
                status_code=exc.status_code,
                detail_exception=exc.detail_exception,
            ):
                return _review_timeout_decision(exc)
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )
        except httpx.TimeoutException as exc:
            if not final_attempt:
                return _retry_decision(exc)
            return _validator_batch_failure_decision(
                ValidatorBatchFailedError(
                    error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                    message=str(exc) or type(exc).__name__,
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                        error_message=str(exc) or type(exc).__name__,
                        occurred_at=self._clock(),
                        artifact_id=artifact.artifact_id,
                        task_id=task.task_id,
                        uid=artifact.uid,
                        exception_type=_exception_type_name(exc),
                    ),
                )
            )
        except Exception as exc:
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )

        unpriceable = self._unpriceable_attempt_decision(
            batch_id=batch_id, artifact=artifact, task=task, session_id=issued.session.session_id
        )
        if unpriceable is not None:
            return unpriceable
        control_failure_decision = self._platform_tool_proxy_control_failure_decision(
            artifact=artifact,
            task=task,
            session_id=issued.session.session_id,
        )
        if control_failure_decision is not None:
            return control_failure_decision
        self._consume_provider_failures(issued.session.session_id)
        return _submission_decision(
            self._build_success_submission(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                outcome=outcome,
            ),
        )

    async def _execute_task_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        orchestrator: TaskRunOrchestrator,
        final_attempt: bool,
        phase_recorder: PhaseRecorder | None = None,
        admission: SandboxAdmission | None = None,
    ) -> TaskExecutionOutcome | TaskAttemptDecision:
        if not hasattr(orchestrator, "execute"):
            return await self._evaluate_task_attempt(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                orchestrator=orchestrator,
                final_attempt=final_attempt,
                phase_recorder=phase_recorder,
            )
        request = MinerTaskRunRequest(
            batch_id=batch_id,
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            execution_time_limit_seconds=self._config.execution_limit_for(task),
        )
        try:
            if admission is not None:
                outcome = await orchestrator.execute(request, phase_recorder=phase_recorder, admission=admission)
            elif phase_recorder is None:
                outcome = await orchestrator.execute(request)
            else:
                outcome = await orchestrator.execute(request, phase_recorder=phase_recorder)
        except SessionBudgetExhaustedError as exc:
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            return _submission_decision(
                self._build_exhausted(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=issued.session.session_id,
                    error_message=str(exc),
                )
            )
        except SandboxInvocationError as exc:
            timeout_decision = self._platform_tool_proxy_timeout_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )
            if timeout_decision is not None:
                return timeout_decision
            if is_timeout_sandbox_invocation(
                status_code=exc.status_code,
                detail_exception=exc.detail_exception,
            ):
                return _review_timeout_decision(exc)
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )
        except httpx.TimeoutException as exc:
            if not final_attempt:
                return _retry_decision(exc)
            return _validator_batch_failure_decision(
                ValidatorBatchFailedError(
                    error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                    message=str(exc) or type(exc).__name__,
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                        error_message=str(exc) or type(exc).__name__,
                        occurred_at=self._clock(),
                        artifact_id=artifact.artifact_id,
                        task_id=task.task_id,
                        uid=artifact.uid,
                        exception_type=_exception_type_name(exc),
                    ),
                )
            )
        except Exception as exc:
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                final_attempt=final_attempt,
            )

        unpriceable = self._unpriceable_attempt_decision(
            batch_id=batch_id, artifact=artifact, task=task, session_id=issued.session.session_id
        )
        if unpriceable is not None:
            return unpriceable
        control_failure_decision = self._platform_tool_proxy_control_failure_decision(
            artifact=artifact,
            task=task,
            session_id=issued.session.session_id,
        )
        if control_failure_decision is not None:
            return control_failure_decision
        self._consume_provider_failures(issued.session.session_id)
        return outcome

    async def record_failure_for_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
    ) -> list[MinerTaskRunSubmission]:
        async def create_submission(task: MinerTask, issued: SessionIssued) -> MinerTaskRunSubmission:
            return self._record_failure(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                uid=artifact.uid,
                artifact_id=artifact.artifact_id,
                task=task,
                error_code=error_code,
                error_message=error_message,
            )

        return await self._run_tasks_with_sessions(
            batch_id=batch_id,
            artifact=artifact,
            tasks=tasks,
            create_submission=create_submission,
        )

    def _record_success(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        outcome: TaskRunOutcome,
    ) -> MinerTaskRunSubmission:
        submission = self._build_success_submission(
            batch_id=batch_id,
            session_id=session_id,
            outcome=outcome,
        )
        self._record_submission(submission)
        return submission

    def _build_success_submission(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        outcome: TaskRunOutcome,
    ) -> MinerTaskRunSubmission:
        breakdown = outcome.run.details.score_breakdown
        if breakdown is None:
            raise RuntimeError("successful task runs require score breakdown details")
        envelope = self._sessions.mark_status(session_id, SessionStatus.COMPLETED)
        return MinerTaskRunSubmission(
            batch_id=batch_id,
            run=outcome.run,
            score=breakdown.total_score,
            execution_log=outcome.tool_receipts,
            usage=outcome.usage,
            session=envelope.session,
        )

    def _build_platform_execution(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        outcome: TaskExecutionOutcome,
    ) -> PlatformOwnedTaskExecution:
        envelope = self._sessions.mark_status(issued.session.session_id, SessionStatus.COMPLETED)
        return PlatformOwnedTaskExecution(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            validator_session_id=issued.session.session_id,
            uid=artifact.uid,
            miner_hotkey_ss58=artifact.miner_hotkey_ss58 or "unknown-miner-hotkey",
            started_at=started_at,
            execution_completed_at=outcome.execution_completed_at,
            response=outcome.response,
            session=envelope.session,
            usage=outcome.usage,
            total_tool_usage=outcome.total_tool_usage,
            execution_log=outcome.execution_log,
            trace=outcome.trace,
        )

    def _record_failure(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        return self._record_failed_submission(
            batch_id=batch_id,
            session_id=session_id,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
        )

    def _record_failed_submission(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        total_tool_usage: ToolUsageSummary | None = None,
        usage: TokenUsageSummary | None = None,
        execution_log: tuple[ToolCall, ...] | None = None,
        elapsed_ms: float | None = None,
    ) -> MinerTaskRunSubmission:
        envelope = self._sessions.mark_status(session_id, SessionStatus.ERROR)
        submission = self._build_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
            total_tool_usage=total_tool_usage,
            usage=usage,
            execution_log=execution_log,
            elapsed_ms=elapsed_ms,
        )
        self._record_submission(submission)
        return submission

    def _build_failed_submission(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        rejected_response: str | None = None,
        total_tool_usage: ToolUsageSummary | None = None,
        usage: TokenUsageSummary | None = None,
        execution_log: tuple[ToolCall, ...] | None = None,
        elapsed_ms: float | None = None,
        scoring_judge_usage: JudgeUsageSummary | None = None,
        trace: EvaluationTrace | None = None,
    ) -> MinerTaskRunSubmission:
        envelope = self._sessions.mark_status(session_id, SessionStatus.ERROR)
        return self._build_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
            rejected_response=rejected_response,
            total_tool_usage=total_tool_usage,
            usage=usage,
            execution_log=execution_log,
            elapsed_ms=elapsed_ms,
            scoring_judge_usage=scoring_judge_usage,
            trace=trace,
        )

    def _record_exhausted(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        submission = self._build_exhausted(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            session_id=session_id,
            error_message=error_message,
        )
        self._record_submission(submission)
        return submission

    def _build_exhausted(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        envelope = self._sessions.inspect(session_id)
        if envelope.session.status is not SessionStatus.EXHAUSTED:
            raise RuntimeError("exhausted task runs require exhausted session status")
        return self._build_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=MinerTaskErrorCode.SESSION_BUDGET_EXHAUSTED,
            error_message=error_message,
        )

    def _record_terminal_failure(
        self,
        *,
        batch_id: UUID,
        envelope: SessionEnvelope,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        total_tool_usage: ToolUsageSummary | None = None,
        usage: TokenUsageSummary | None = None,
        execution_log: tuple[ToolCall, ...] | None = None,
        elapsed_ms: float | None = None,
        scoring_judge_usage: JudgeUsageSummary | None = None,
        trace: EvaluationTrace | None = None,
    ) -> MinerTaskRunSubmission:
        submission = self._build_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
            total_tool_usage=total_tool_usage,
            usage=usage,
            execution_log=execution_log,
            elapsed_ms=elapsed_ms,
            scoring_judge_usage=scoring_judge_usage,
            trace=trace,
        )
        self._record_submission(submission)
        return submission

    def _build_terminal_failure(
        self,
        *,
        batch_id: UUID,
        envelope: SessionEnvelope,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        rejected_response: str | None = None,
        total_tool_usage: ToolUsageSummary | None = None,
        usage: TokenUsageSummary | None = None,
        execution_log: tuple[ToolCall, ...] | None = None,
        elapsed_ms: float | None = None,
        scoring_judge_usage: JudgeUsageSummary | None = None,
        trace: EvaluationTrace | None = None,
    ) -> MinerTaskRunSubmission:
        session_id = envelope.session.session_id
        completed_at = self._clock()
        summarized_usage, summarized_tool_usage = self._summarize_session(envelope)
        receipt_log = tuple(self._receipts.for_session(session_id)) if execution_log is None else execution_log
        details = EvaluationDetails(
            error=EvaluationError(code=error_code, message=error_message),
            scoring_judge_usage=scoring_judge_usage,
            trace=trace,
            total_tool_usage=total_tool_usage or summarized_tool_usage,
            elapsed_ms=elapsed_ms or _elapsed_ms(issued_at=envelope.session.issued_at, completed_at=completed_at),
        )
        run = MinerTaskRun(
            session_id=session_id,
            uid=uid,
            artifact_id=artifact_id,
            task_id=task.task_id,
            response=None,
            rejected_response=rejected_response,
            details=details,
            completed_at=completed_at,
        )
        self._receipts.clear_session(session_id)
        submission = MinerTaskRunSubmission(
            batch_id=batch_id,
            run=run,
            score=0.0,
            execution_log=receipt_log,
            usage=summarized_usage if usage is None else usage,
            session=envelope.session,
        )
        return submission

    def _record_task_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_code: MinerTaskErrorCode,
        error_message: str,
        log_message: str,
        exc: Exception,
    ) -> MinerTaskRunSubmission:
        submission = self._build_task_failure(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            session_id=session_id,
            error_code=error_code,
            error_message=error_message,
            log_message=log_message,
            exc=exc,
        )
        self._record_submission(submission)
        return submission

    def _build_task_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_code: MinerTaskErrorCode,
        error_message: str,
        log_message: str,
        exc: Exception,
        rejected_response: str | None = None,
        scoring_judge_usage: JudgeUsageSummary | None = None,
    ) -> MinerTaskRunSubmission:
        logger.error(
            log_message,
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
            },
            exc_info=exc,
        )
        evaluation_trace = getattr(exc, "evaluation_trace", None)
        return self._build_failed_submission(
            batch_id=batch_id,
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
            rejected_response=rejected_response,
            scoring_judge_usage=scoring_judge_usage or getattr(exc, "judge_usage", None),
            trace=evaluation_trace if isinstance(evaluation_trace, EvaluationTrace) else None,
        )

    def _record_terminal_timeout_submission(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        timeout_exc: SandboxInvocationError,
    ) -> TaskAttemptDecision:
        decision = self._build_terminal_timeout_decision(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            issued=issued,
            timeout_exc=timeout_exc,
        )
        self._record_submission(_require_submission(decision))
        return decision

    def _build_terminal_timeout_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        timeout_exc: SandboxInvocationError,
    ) -> TaskAttemptDecision:
        current_attempt_receipts = self._current_attempt_receipts(
            session_id=issued.session.session_id,
            active_attempt=issued.session.active_attempt,
        )
        envelope = self._sessions.inspect(issued.session.session_id)
        usage, total_tool_usage = self._summarize_receipts(
            envelope=envelope,
            receipts=current_attempt_receipts,
        )
        submission = self._build_failed_submission(
            batch_id=batch_id,
            session_id=issued.session.session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
            error_message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
            total_tool_usage=total_tool_usage,
            usage=usage,
            execution_log=current_attempt_receipts,
            elapsed_ms=_elapsed_ms(
                issued_at=envelope.session.issued_at,
                completed_at=self._clock(),
            ),
        )
        logger.error(
            "miner task timed out after retry budget exhaustion",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
            },
            exc_info=timeout_exc,
        )
        return _submission_decision(
            submission,
            attempt_execution_log=current_attempt_receipts,
        )

    def _platform_tool_proxy_timeout_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: SandboxInvocationError,
        final_attempt: bool,
    ) -> TaskAttemptDecision | None:
        latest_receipt = self._latest_current_attempt_platform_tool_proxy_receipt(session_id)
        if latest_receipt is None:
            return None
        if not is_uncaught_platform_tool_proxy_timeout_sandbox_invocation(
            detail_code=exc.detail_code,
            detail_exception=exc.detail_exception,
            detail_error=exc.detail_error,
            latest_current_attempt_platform_tool_proxy_receipt_is_timeout=(
                is_platform_tool_proxy_timeout_receipt(latest_receipt)
            ),
        ):
            return None
        if not final_attempt:
            return _retry_decision(
                exc,
                timeout_owner="platform_tool_proxy_execute",
                attempt_error_code=PLATFORM_TOOL_PROXY_TIMEOUT_ERROR_CODE,
            )
        return _submission_decision(
            self._build_task_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=session_id,
                error_code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                error_message="platform tool proxy execution timed out",
                log_message="miner-owned platform tool proxy execution timed out",
                exc=exc,
            ),
            timeout_owner="platform_tool_proxy_execute",
        )

    def _latest_current_attempt_platform_tool_proxy_receipt(self, session_id: UUID) -> ToolCall | None:
        envelope = self._sessions.inspect(session_id)
        current_attempt_receipts = self._current_attempt_receipts(
            session_id=session_id,
            active_attempt=envelope.session.active_attempt,
        )
        for receipt in reversed(current_attempt_receipts):
            if receipt.details.extra is None:
                continue
            if receipt.details.extra.get("platform_tool_proxy_error_code") is not None:
                return receipt
        return None

    def _unpriceable_attempt_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
    ) -> TaskAttemptDecision | None:
        envelope = self._sessions.inspect(session_id)
        receipts = self._current_attempt_receipts(session_id=session_id, active_attempt=envelope.session.active_attempt)
        if not any(r.is_successful() and (r.details.extra or {}).get("reference_cost_unavailable") for r in receipts):
            return None
        return _submission_decision(
            self._build_failed_submission(
                batch_id=batch_id,
                session_id=session_id,
                uid=artifact.uid,
                artifact_id=artifact.artifact_id,
                task=task,
                error_code=MinerTaskErrorCode.TOOL_PROVIDER_FAILED,
                error_message="OpenRouter regular reference cost unavailable after dispatch",
            )
        )

    def _non_timeout_failure_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: Exception,
        final_attempt: bool,
    ) -> TaskAttemptDecision:
        unpriceable = self._unpriceable_attempt_decision(
            batch_id=batch_id, artifact=artifact, task=task, session_id=session_id
        )
        if unpriceable is not None:
            return unpriceable
        if isinstance(exc, LlmRetryExhaustedError):
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
                    error_message=str(exc),
                    log_message="validator scoring provider retries exhausted",
                    exc=exc,
                )
            )

        if isinstance(exc, MinerResponseValidationError):
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.MINER_RESPONSE_INVALID,
                    error_message=str(exc),
                    rejected_response=exc.rejected_response,
                    log_message="miner returned invalid response payload",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and exc.detail_code == SANDBOX_OOM_AFFECTED:
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SANDBOX_OOM_AFFECTED,
                    error_message="execution lost its result in an OOM-affected sandbox",
                    log_message="sandbox OOM affected miner task execution",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and is_script_validation_sandbox_invocation(
            detail_code=exc.detail_code,
        ):
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
                    error_message=exc.detail_error or str(exc),
                    log_message="miner script failed validation during sandbox preload",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and exc.detail_code == SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION:
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.MINER_UNHANDLED_EXCEPTION,
                    error_message=exc.detail_error or str(exc),
                    log_message="miner entrypoint raised unhandled exception",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError):
            if not final_attempt:
                return _retry_decision(exc)
            return _submission_decision(
                self._build_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                    error_message=str(exc),
                    log_message="sandbox invocation failed during miner task run",
                    exc=exc,
                )
            )

        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                message=str(exc),
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                    error_message=str(exc),
                    occurred_at=self._clock(),
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=_exception_type_name(exc),
                ),
            )
        )

    def _platform_tool_proxy_control_failure_decision(
        self,
        *,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
    ) -> TaskAttemptDecision | None:
        failure = self._latest_platform_tool_proxy_control_failure(session_id)
        if failure is None:
            return None
        message = f"platform tool proxy control failure: {failure.error_code}"
        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=failure.error_code,
                message=message,
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=failure.error_code,
                    error_message=message,
                    occurred_at=failure.occurred_at,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=failure.exception_type,
                ),
            )
        )

    def _latest_platform_tool_proxy_control_failure(
        self,
        session_id: UUID,
    ) -> _PlatformToolProxyControlFailure | None:
        for receipt in reversed(tuple(self._receipts.for_session(session_id))):
            extra = receipt.details.extra
            if extra is None:
                continue
            if is_platform_tool_proxy_timeout_receipt(receipt):
                continue
            error_code = extra.get("platform_tool_proxy_error_code")
            if error_code not in VALIDATOR_OWNED_PLATFORM_TOOL_PROXY_CONTROL_ERROR_CODES:
                continue
            error_message = extra.get("error_message") or f"platform tool proxy error: {error_code}"
            occurred_at = receipt.details.execution.finished_at if receipt.details.execution is not None else None
            return _PlatformToolProxyControlFailure(
                error_code=MinerTaskErrorCode(error_code),
                error_message=error_message,
                occurred_at=occurred_at or self._clock(),
                exception_type=extra.get("error_type"),
            )
        return None

    def _finalize_assigned_task_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        decision: TaskAttemptDecision,
        phase_recorder: PhaseRecorder | None = None,
    ) -> _AssignedTaskFinalization:
        diagnostics = (
            phase_recorder.diagnostics(
                timeout_owner=_timeout_owner_from_decision(decision),
                failure_owner=_failure_owner_from_decision(decision),
            )
            if isinstance(phase_recorder, _AssignedAttemptPhaseRecorder)
            else None
        )
        if decision.kind is AttemptControlKind.SUBMISSION:
            terminal_outcome = AttemptControlKind.SUBMISSION.value
            submission = _require_submission(decision)
            error_code = _submission_error_code_or_none(submission)
            terminal_effect = MinerTaskAttemptTerminalEffect.TASK_RESULT
            attempt = self._build_terminated_attempt_record(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                decision=decision,
                retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                terminal_effect=terminal_effect,
                diagnostics=diagnostics,
            )
            return _AssignedTaskFinalization(
                result=PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=submission,
                    terminal_attempt=attempt,
                ),
                local_submission=submission,
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )

        if decision.kind is AttemptControlKind.REVIEW_TIMEOUT and attempt_number >= max_attempts:
            timeout_resolution = self._build_terminal_timeout_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                timeout_exc=_require_timeout_exc(decision),
            )
            terminal_outcome = AttemptControlKind.SUBMISSION.value
            submission = _require_submission(timeout_resolution)
            error_code = _submission_error_code_or_none(submission)
            attempt = self._build_terminated_attempt_record(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                decision=timeout_resolution,
                retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
                diagnostics=diagnostics,
            )
            return _AssignedTaskFinalization(
                result=PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=submission,
                    terminal_attempt=attempt,
                ),
                local_submission=submission,
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )

        if decision.kind in {AttemptControlKind.RETRY, AttemptControlKind.REVIEW_TIMEOUT}:
            terminal_outcome = decision.kind.value
            error_code = _attempt_error_code(decision)
            attempt = self._build_terminated_attempt_record(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                decision=decision,
                retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
                terminal_effect=None,
                diagnostics=diagnostics,
            )
            return _AssignedTaskFinalization(
                result=PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=None,
                    terminal_attempt=attempt,
                ),
                local_submission=None,
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )

        if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
            validator_failure = _require_validator_failure(decision)
            terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
            error_code = str(validator_failure.error_code)
            retry_decision = (
                MinerTaskAttemptRetryDecision.WILL_RETRY
                if attempt_number < max_attempts
                else MinerTaskAttemptRetryDecision.WILL_NOT_RETRY
            )
            try:
                validator_error_code = MinerTaskErrorCode(validator_failure.error_code)
            except ValueError:
                validator_error_code = None
            is_delivery_disqualifying = (
                validator_error_code is not None
                and is_delivery_disqualifying_validator_pair_error(validator_error_code)
            )
            delivery_failure_detail = None
            if retry_decision is MinerTaskAttemptRetryDecision.WILL_RETRY:
                terminal_effect = None
            elif is_delivery_disqualifying:
                terminal_effect = MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
                delivery_failure_detail = validator_failure.failure_detail
            else:
                terminal_effect = MinerTaskAttemptTerminalEffect.ATTEMPT_FAILURE
            attempt = self._build_terminated_attempt_record(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=started_at,
                decision=decision,
                retry_decision=retry_decision,
                terminal_effect=terminal_effect,
                diagnostics=diagnostics,
                delivery_failure_detail=delivery_failure_detail,
            )
            return _AssignedTaskFinalization(
                result=PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=None,
                    terminal_attempt=attempt,
                ),
                local_submission=None,
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )

        raise RuntimeError("assigned task attempt returned unexpected decision")

    def _record_assigned_task_local_side_effects_best_effort(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        result: PlatformOwnedTaskResult,
        local_submission: MinerTaskRunSubmission | None,
    ) -> None:
        if local_submission is not None:
            try:
                self._progress.record(local_submission)
            except Exception as exc:
                self._log_assigned_task_local_recording_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    result=result,
                    local_write_target="progress.record",
                    exc=exc,
                )
            try:
                self._evaluation_records.record(local_submission)
            except Exception as exc:
                self._log_assigned_task_local_recording_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    result=result,
                    local_write_target="evaluation_records.record",
                    exc=exc,
                )
        try:
            self._progress.record_terminated_attempt(result.terminal_attempt)
        except Exception as exc:
            self._log_assigned_task_local_recording_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                result=result,
                local_write_target="progress.record_terminated_attempt",
                exc=exc,
            )

    def _log_assigned_task_local_recording_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        result: PlatformOwnedTaskResult,
        local_write_target: str,
        exc: Exception,
    ) -> None:
        logger.warning(
            "assigned miner task local recording failed",
            extra={
                "data": {
                    "batch_id": str(batch_id),
                    "artifact_id": str(artifact.artifact_id),
                    "task_id": str(task.task_id),
                    "uid": artifact.uid,
                    "validator_session_id": str(result.terminal_attempt.validator_session_id),
                    "attempt_number": result.attempt_number,
                    "max_attempts": result.terminal_attempt.max_attempts,
                    "error_code": result.terminal_attempt.error_code,
                    "local_write_target": local_write_target,
                    "exception_type": _exception_type_name(exc),
                }
            },
            exc_info=exc,
        )

    def _record_terminated_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        decision: TaskAttemptDecision,
        retry_decision: MinerTaskAttemptRetryDecision,
        terminal_effect: MinerTaskAttemptTerminalEffect | None,
        diagnostics: MinerTaskAttemptDiagnostics | None = None,
        delivery_failure_detail: ValidatorBatchFailureDetail | None = None,
    ) -> MinerTaskAttemptAuditRecord:
        record = self._build_terminated_attempt_record(
            batch_id=batch_id,
            artifact=artifact,
            task=task,
            issued=issued,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            started_at=started_at,
            decision=decision,
            retry_decision=retry_decision,
            terminal_effect=terminal_effect,
            diagnostics=diagnostics,
            delivery_failure_detail=delivery_failure_detail,
        )
        self._progress.record_terminated_attempt(record)
        return record

    def _build_terminated_attempt_record(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        decision: TaskAttemptDecision,
        retry_decision: MinerTaskAttemptRetryDecision,
        terminal_effect: MinerTaskAttemptTerminalEffect | None,
        diagnostics: MinerTaskAttemptDiagnostics | None = None,
        delivery_failure_detail: ValidatorBatchFailureDetail | None = None,
    ) -> MinerTaskAttemptAuditRecord:
        status = MinerTaskAttemptStatus.FAILED
        attempt_receipts = decision.attempt_execution_log or tuple(
            self._receipts.for_session(issued.session.session_id)
        )
        error_code = _attempt_error_code(decision)
        if decision.submission is not None and decision.submission.run.details.error is None:
            status = MinerTaskAttemptStatus.SUCCEEDED
            error_code = None
        record = MinerTaskAttemptAuditRecord(
            validator_session_id=issued.session.session_id,
            batch_id=batch_id,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            uid=artifact.uid,
            miner_hotkey_ss58=artifact.miner_hotkey_ss58 or "unknown-miner-hotkey",
            started_at=started_at,
            finished_at=self._clock(),
            status=status,
            error_code=error_code,
            error_summary_code=error_code,
            retry_decision=retry_decision,
            terminal_effect=terminal_effect,
            max_attempts=max_attempts,
            execution_log=attempt_receipts,
            diagnostics=diagnostics,
            delivery_failure_detail=delivery_failure_detail,
        )
        return record

    def _log_retry_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        attempt_number: int,
        exc: Exception,
    ) -> None:
        logger.warning(
            "miner task run attempt failed; retrying once",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
                "attempt_number": attempt_number,
            },
            exc_info=exc,
        )

    def _summarize_session(self, envelope: SessionEnvelope) -> tuple[TokenUsageSummary, ToolUsageSummary]:
        receipts = tuple(self._receipts.for_session(envelope.session.session_id))
        return self._usage.summarize(envelope.session, receipts)

    def _record_submission(self, submission: MinerTaskRunSubmission) -> None:
        self._progress.record(submission)
        self._evaluation_records.record(submission)

    def _consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        return self._progress.consume_provider_failures(session_id)

    def _clear_task_session(self, session_id: UUID) -> None:
        self._progress.clear_task_session(session_id)

    def _issue_session(
        self,
        *,
        batch_id: UUID,
        uid: int,
        artifact_id: UUID | None = None,
        task: MinerTask,
        attempt_number: int = 1,
        assignment_token: str | None = None,
        phase_recorder: PhaseRecorder | None = None,
    ) -> SessionIssued:
        issued_at = self._clock()
        expires_at = issued_at + self._config.session_ttl_for(task)
        token = secrets.token_urlsafe(self._config.token_secret_bytes)
        request = SessionTokenRequest(
            session_id=uuid4(),
            uid=uid,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=expires_at,
            budget_usd=task.budget_usd,
            token=token,
        )
        issued = self._sessions.issue(request)
        try:
            self._progress.register_task_session(
                batch_id=batch_id,
                session_id=issued.session.session_id,
            )
            if (
                artifact_id is not None
                and assignment_token is not None
                and self._platform_tool_proxy_scopes is not None
            ):
                self._platform_tool_proxy_scopes.register_session(
                    batch_id=batch_id,
                    session_id=issued.session.session_id,
                    artifact_id=artifact_id,
                    task_id=task.task_id,
                    assignment_token=assignment_token,
                    attempt_number=attempt_number,
                    phase_recorder=phase_recorder,
                )
        except Exception as exc:
            try:
                self._cleanup_attempt_session(issued.session.session_id)
            except Exception:
                logger.exception(
                    "session cleanup failed after partial issue",
                    extra={
                        "batch_id": str(batch_id),
                        "session_id": str(issued.session.session_id),
                        "artifact_id": str(artifact_id) if artifact_id is not None else None,
                        "task_id": str(task.task_id),
                    },
                )
            raise _PartialSessionIssueError(issued=issued, cause=exc) from exc
        return issued

    def _cleanup_attempt_session(self, session_id: UUID) -> None:
        self._clear_task_session(session_id)
        self._clear_platform_tool_proxy_session(session_id)
        self._sessions.revoke(session_id)
        self._receipts.clear_session(session_id)

    def _cleanup_attempt_session_best_effort(
        self,
        *,
        session_id: UUID,
        batch_id: UUID,
        artifact_id: UUID,
        task_id: UUID,
    ) -> None:
        try:
            self._cleanup_attempt_session(session_id)
        except Exception:
            logger.exception(
                "assigned miner task cleanup failed after result construction",
                extra={
                    "batch_id": str(batch_id),
                    "session_id": str(session_id),
                    "artifact_id": str(artifact_id),
                    "task_id": str(task_id),
                },
            )

    def _clear_platform_tool_proxy_session(self, session_id: UUID) -> None:
        if self._platform_tool_proxy_scopes is not None:
            self._platform_tool_proxy_scopes.clear_session(session_id)

    def _begin_session_attempt(self, session_id: UUID) -> SessionIssued:
        token = secrets.token_urlsafe(self._config.token_secret_bytes)
        return self._sessions.begin_attempt(session_id, token=token)

    def _current_attempt_receipts(
        self,
        *,
        session_id: UUID,
        active_attempt: int,
    ) -> tuple[ToolCall, ...]:
        return tuple(
            receipt
            for receipt in self._receipts.for_session(session_id)
            if receipt.details.extra is not None
            and receipt.details.extra.get("session_active_attempt") == str(active_attempt)
        )

    def _summarize_receipts(
        self,
        *,
        envelope: SessionEnvelope,
        receipts: tuple[ToolCall, ...],
    ) -> tuple[TokenUsageSummary, ToolUsageSummary]:
        session = envelope.session.with_usage(_usage_from_receipts(receipts))
        return self._usage.summarize(session, receipts)


def _submission_decision(
    submission: MinerTaskRunSubmission,
    *,
    attempt_execution_log: tuple[ToolCall, ...] | None = None,
    timeout_owner: str | None = None,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.SUBMISSION,
        submission=submission,
        attempt_execution_log=submission.execution_log if attempt_execution_log is None else attempt_execution_log,
        timeout_owner=timeout_owner,
    )


def _retry_decision(
    exc: Exception,
    *,
    timeout_owner: str | None = None,
    attempt_error_code: MinerTaskErrorCode | str | None = None,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.RETRY,
        attempt_error_code=attempt_error_code,
        timeout_owner=timeout_owner,
        retry_exc=exc,
    )


def _review_timeout_decision(
    exc: SandboxInvocationError,
    *,
    attempt_error_code: MinerTaskErrorCode | str | None = None,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.REVIEW_TIMEOUT,
        timeout_exc=exc,
        attempt_error_code=attempt_error_code,
    )


def _validator_batch_failure_decision(
    validator_failure: ValidatorBatchFailedError,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.VALIDATOR_BATCH_FAILURE,
        validator_failure=validator_failure,
    )


def _require_submission(decision: TaskAttemptDecision) -> MinerTaskRunSubmission:
    if decision.submission is None:
        raise RuntimeError("attempt decision requires submission")
    return decision.submission


def _require_timeout_exc(decision: TaskAttemptDecision) -> SandboxInvocationError:
    if decision.timeout_exc is None:
        raise RuntimeError("attempt decision requires timeout exception")
    return decision.timeout_exc


def _require_retry_exc(decision: TaskAttemptDecision) -> Exception:
    if decision.retry_exc is None:
        raise RuntimeError("attempt decision requires retry exception")
    return decision.retry_exc


def _require_validator_failure(decision: TaskAttemptDecision) -> ValidatorBatchFailedError:
    if decision.validator_failure is None:
        raise RuntimeError("attempt decision requires validator batch failure")
    return decision.validator_failure


def _require_single_completed_submission(
    validator_failure: ValidatorBatchFailedError,
) -> MinerTaskRunSubmission:
    completed_submissions = validator_failure.completed_submissions
    if completed_submissions is None or len(completed_submissions) != 1:
        raise RuntimeError("validator batch failure must provide exactly one completed submission")
    return completed_submissions[0]


def _exception_type_name(exc: Exception | None) -> str | None:
    if exc is None:
        return None
    return type(exc).__name__


def _submission_error_code(submission: MinerTaskRunSubmission) -> MinerTaskErrorCode:
    error = submission.run.details.error
    if error is None:
        raise RuntimeError("artifact failure submission requires error code")
    return error.code


def _submission_error_message(submission: MinerTaskRunSubmission) -> str:
    error = submission.run.details.error
    if error is None:
        raise RuntimeError("artifact failure submission requires error message")
    return error.message


def _submission_error_code_or_none(submission: MinerTaskRunSubmission) -> MinerTaskErrorCode | None:
    error = submission.run.details.error
    if error is None:
        return None
    return error.code


def _attempt_error_code(
    decision: TaskAttemptDecision,
) -> str | None:
    if decision.submission is not None:
        error_code = _submission_error_code_or_none(decision.submission)
        return None if error_code is None else str(error_code)
    if decision.validator_failure is not None:
        return str(decision.validator_failure.error_code)
    if decision.attempt_error_code is not None:
        return str(decision.attempt_error_code)
    if decision.retry_exc is not None:
        if isinstance(decision.retry_exc, SandboxInvocationError):
            return decision.retry_exc.detail_code or str(MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED)
        if isinstance(decision.retry_exc, httpx.TimeoutException):
            return str(MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT)
        return type(decision.retry_exc).__name__[:128]
    if decision.timeout_exc is not None:
        return decision.timeout_exc.detail_code or str(MinerTaskErrorCode.TIMEOUT_MINER_OWNED)
    return None


def _timeout_owner_from_decision(decision: TaskAttemptDecision) -> str | None:
    if decision.timeout_owner is not None:
        return decision.timeout_owner
    if decision.kind is AttemptControlKind.REVIEW_TIMEOUT:
        return "sandbox_entrypoint"
    if isinstance(decision.retry_exc, httpx.TimeoutException):
        return "validator_internal"
    if (
        decision.validator_failure is not None
        and decision.validator_failure.error_code is MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT
    ):
        return "validator_internal"
    return None


def _failure_owner_from_decision(decision: TaskAttemptDecision) -> str | None:
    if decision.retry_exc is not None:
        return _exception_type_name(decision.retry_exc)
    if decision.timeout_exc is not None:
        return _exception_type_name(decision.timeout_exc)
    if decision.validator_failure is not None:
        return str(decision.validator_failure.error_code)
    if decision.submission is not None:
        error_code = _submission_error_code_or_none(decision.submission)
        return None if error_code is None else str(error_code)
    return None


def _orchestrator_accepts_phase_recorder(orchestrator: TaskRunOrchestrator) -> bool:
    parameters = inspect.signature(orchestrator.evaluate).parameters
    return "phase_recorder" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _validator_batch_failed_from_existing_submission(
    *,
    submission: MinerTaskRunSubmission,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    occurred_at: datetime,
) -> ValidatorBatchFailedError:
    error_code = _submission_error_code(submission)
    return ValidatorBatchFailedError(
        error_code=error_code,
        message=_submission_error_message(submission),
        failure_detail=ValidatorBatchFailureDetail(
            error_code=error_code,
            error_message=_submission_error_message(submission),
            occurred_at=occurred_at,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            uid=artifact.uid,
            exception_type=(
                "SandboxInvocationError" if error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED else None
            ),
        ),
        completed_submissions=(submission,),
        remaining_tasks=(),
    )


__all__ = [
    "ArtifactExecutionFailedError",
    "ArtifactEvaluationOutcome",
    "ArtifactFailure",
    "EvaluationRunner",
    "SandboxFailureDiagnostics",
    "LOCAL_RETRY_ATTEMPTS",
    "TERMINAL_TIMEOUT_ERROR_MESSAGE",
    "TaskAttemptDecision",
    "ValidatorBatchFailureDetail",
    "ValidatorBatchFailedError",
]
