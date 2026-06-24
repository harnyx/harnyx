"""Helper to run miner tasks for a single artifact."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isclose
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from harnyx_commons.application.dto.session import SessionEnvelope, SessionIssued, SessionTokenRequest
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.session import LlmUsageTotals, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.miner_task_failure_policy import (
    SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION,
    TERMINAL_TIMEOUT_ERROR_MESSAGE,
    ProviderFailureEvidence,
    is_platform_tool_proxy_timeout_receipt,
    is_provider_caused_terminal_failure,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    is_uncaught_platform_tool_proxy_timeout_sandbox_invocation,
    provider_batch_failure_evidence,
    provider_batch_failure_message,
)
from harnyx_commons.tools.types import is_search_tool
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
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
TaskSessionLimiter = AbstractAsyncContextManager[None]

logger = logging.getLogger("harnyx_validator.scheduler")
measurement_logger = logging.getLogger("harnyx_validator.measurement")
LOCAL_RETRY_ATTEMPTS = 2
DIAGNOSTIC_ID_MAX_LENGTH = 512
DIAGNOSTIC_STATE_ERROR_MAX_LENGTH = 2048
DIAGNOSTIC_TEXT_MAX_LENGTH = 4096
DIAGNOSTIC_LOG_TAIL_MAX_LENGTH = 8192
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
    retry_exc: Exception | None = None
    timeout_exc: SandboxInvocationError | None = None
    validator_failure: ValidatorBatchFailedError | None = None


@dataclass(frozen=True, slots=True)
class SandboxFailureDiagnostics:
    image: str | None = None
    pull_policy: str | None = None
    container_name: str | None = None
    container_id: str | None = None
    status: str | None = None
    exit_code: int | None = None
    oom_killed: bool | None = None
    state_error: str | None = None
    error_text: str | None = None
    docker_logs_tail: str | None = None
    pull_returncode: int | None = None
    pull_stdout_tail: str | None = None
    pull_stderr_tail: str | None = None
    run_returncode: int | None = None
    run_stdout_tail: str | None = None
    run_stderr_tail: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatorBatchFailureDetail:
    error_code: str
    error_message: str
    occurred_at: datetime
    artifact_id: UUID | None = None
    task_id: UUID | None = None
    uid: int | None = None
    exception_type: str | None = None
    traceback: str | None = None
    sandbox_diagnostics: SandboxFailureDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class _PlatformToolProxyControlFailure:
    error_code: str
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


@dataclass(slots=True)
class _ArtifactDispatchState:
    submissions_by_index: list[MinerTaskRunSubmission | None]
    validator_failure: ValidatorBatchFailedError | None = None
    unexpected_failure: Exception | None = None


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
        self._subtensor = subtensor_client
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._receipts = receipt_log
        self._config = config
        self._clock = clock
        self._progress = progress
        self._usage = usage_summarizer or UsageSummarizer()
        self._platform_tool_proxy_scopes = platform_tool_proxy_scopes
        self._validator_uid: int | None = None

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
    ) -> PlatformOwnedTaskResult:
        session_started_at = time.monotonic()
        terminal_outcome = "unexpected"
        error_code: str | None = None
        issued: SessionIssued | None = None
        try:
            issued = self._begin_session_attempt(
                self._issue_session(
                    batch_id=batch_id,
                    uid=artifact.uid,
                    artifact_id=artifact.artifact_id,
                    task=task,
                    attempt_number=attempt_number,
                    assignment_token=assignment_token,
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
                terminal_effect = MinerTaskAttemptTerminalEffect.TASK_RESULT
                platform_result: MinerTaskRunSubmission | None = submission
                if error_code is not None and is_delivery_disqualifying_validator_pair_error(error_code):
                    terminal_effect = MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE
                    platform_result = None
                attempt = self._record_terminated_attempt(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    issued=issued,
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    started_at=attempt_started_at,
                    decision=decision,
                    retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                    terminal_effect=terminal_effect,
                )
                return PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=platform_result,
                    terminal_attempt=attempt,
                )

            if decision.kind is AttemptControlKind.REVIEW_TIMEOUT and attempt_number >= max_attempts:
                timeout_exc = _require_timeout_exc(decision)
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
                attempt = self._record_terminated_attempt(
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
                return PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=submission,
                    terminal_attempt=attempt,
                )

            if decision.kind in {AttemptControlKind.RETRY, AttemptControlKind.REVIEW_TIMEOUT}:
                terminal_outcome = decision.kind.value
                error_code = _attempt_error_code(
                    decision,
                    execution_log=decision.attempt_execution_log
                    or tuple(self._receipts.for_session(issued.session.session_id)),
                )
                attempt = self._record_terminated_attempt(
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
                return PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=None,
                    terminal_attempt=attempt,
                )

            if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                validator_failure = _require_validator_failure(decision)
                terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
                error_code = str(validator_failure.error_code)
                attempt = self._record_terminated_attempt(
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
                )
                return PlatformOwnedTaskResult(
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    attempt_number=attempt_number,
                    result=None,
                    terminal_attempt=attempt,
                )

            raise RuntimeError("assigned task attempt returned unexpected decision")
        finally:
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
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )
            if issued is not None:
                self._cleanup_attempt_session(issued.session.session_id)

    async def evaluate_assigned_task_queue(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        initial_assignments: Sequence[MinerTaskWorkAssignment],
        assignment_queue: asyncio.Queue[MinerTaskWorkAssignment],
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
        orchestrator: TaskRunOrchestrator,
    ) -> None:
        active: dict[asyncio.Task[PlatformOwnedTaskResult], MinerTaskWorkAssignment] = {}
        queue_waiter: asyncio.Task[MinerTaskWorkAssignment] | None = None
        close_waiter = asyncio.create_task(close_requested.wait())

        def start_assignment(assignment: MinerTaskWorkAssignment) -> None:
            active[
                asyncio.create_task(
                    self.evaluate_assigned_task(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=assignment.task,
                        attempt_number=assignment.attempt_number,
                        max_attempts=assignment.max_attempts,
                        assignment_token=assignment.assignment_token,
                        orchestrator=orchestrator,
                    )
                )
            ] = assignment

        for assignment in initial_assignments:
            start_assignment(assignment)

        try:
            while active or not close_requested.is_set():
                while not close_requested.is_set():
                    try:
                        start_assignment(assignment_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if not active:
                    if close_requested.is_set():
                        break
                    queue_waiter = asyncio.create_task(assignment_queue.get())
                    done, _pending = await asyncio.wait(
                        {queue_waiter, close_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if queue_waiter in done:
                        start_assignment(queue_waiter.result())
                        queue_waiter = None
                    continue

                wait_for: set[asyncio.Task[object]] = set(active)
                if not close_requested.is_set():
                    if queue_waiter is None:
                        queue_waiter = asyncio.create_task(assignment_queue.get())
                    wait_for.add(queue_waiter)
                    wait_for.add(close_waiter)
                done, _pending = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                if queue_waiter is not None and queue_waiter in done:
                    start_assignment(queue_waiter.result())
                    queue_waiter = None

                delivery_failure_seen = False
                for task in tuple(active):
                    if task not in done:
                        continue
                    active.pop(task)
                    result = task.result()
                    await result_queue.put(result)
                    if result.terminal_attempt.terminal_effect is MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE:
                        delivery_failure_seen = True
                if delivery_failure_seen:
                    for remaining in active:
                        remaining.cancel()
                    if active:
                        await asyncio.gather(*active, return_exceptions=True)
                    return
        finally:
            if queue_waiter is not None and not queue_waiter.done():
                queue_waiter.cancel()
            close_waiter.cancel()
            await asyncio.gather(
                *(task for task in (queue_waiter, close_waiter) if task is not None),
                return_exceptions=True,
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
            attempt_started_at = self._clock()
            try:
                submission = self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=assignment.task,
                    session_id=issued.session.session_id,
                    error_code=error_code,
                    error_message=error_message,
                    log_message="assigned miner task setup failed",
                    exc=RuntimeError(error_message),
                )
                decision = _submission_decision(submission)
                attempt = self._record_terminated_attempt(
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
                results.append(
                    PlatformOwnedTaskResult(
                        batch_id=batch_id,
                        artifact_id=artifact.artifact_id,
                        task_id=assignment.task.task_id,
                        attempt_number=assignment.attempt_number,
                        result=submission,
                        terminal_attempt=attempt,
                    )
                )
            finally:
                self._cleanup_attempt_session(issued.session.session_id)
        return tuple(results)

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
    ) -> TaskAttemptDecision:
        request = MinerTaskRunRequest(
            batch_id=batch_id,
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
        )
        try:
            outcome = await orchestrator.evaluate(request)
        except SessionBudgetExhaustedError as exc:
            control_failure_decision = self._platform_tool_proxy_control_failure_decision(
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
            )
            if control_failure_decision is not None:
                return control_failure_decision
            return _submission_decision(
                self._record_exhausted(
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
            provider_failures = self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                provider_failures=provider_failures,
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
            provider_failures = self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                provider_failures=provider_failures,
                final_attempt=final_attempt,
            )

        control_failure_decision = self._platform_tool_proxy_control_failure_decision(
            artifact=artifact,
            task=task,
            session_id=issued.session.session_id,
        )
        if control_failure_decision is not None:
            return control_failure_decision
        provider_failures = self._consume_provider_failures(issued.session.session_id)
        provider_failure_decision = self._provider_batch_failure_decision(
            artifact=artifact,
            task=task,
            provider_failures=provider_failures,
            exception_type=None,
        )
        if provider_failure_decision is not None:
            return provider_failure_decision
        return _submission_decision(
            self._record_success(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                outcome=outcome,
            ),
        )

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
        breakdown = outcome.run.details.score_breakdown
        if breakdown is None:
            raise RuntimeError("successful task runs require score breakdown details")
        envelope = self._sessions.mark_status(session_id, SessionStatus.COMPLETED)
        submission = MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=self._validator_uid_value(),
            run=outcome.run,
            score=breakdown.total_score,
            execution_log=outcome.tool_receipts,
            usage=outcome.usage,
            session=envelope.session,
        )
        self._record_submission(submission)
        return submission

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
        return self._record_terminal_failure(
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

    def _record_exhausted(
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
        return self._record_terminal_failure(
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
    ) -> MinerTaskRunSubmission:
        session_id = envelope.session.session_id
        completed_at = self._clock()
        summarized_usage, summarized_tool_usage = self._summarize_session(envelope)
        receipt_log = tuple(self._receipts.for_session(session_id)) if execution_log is None else execution_log
        details = EvaluationDetails(
            error=EvaluationError(code=error_code, message=error_message),
            total_tool_usage=total_tool_usage or summarized_tool_usage,
            elapsed_ms=elapsed_ms or _elapsed_ms(issued_at=envelope.session.issued_at, completed_at=completed_at),
        )
        run = MinerTaskRun(
            session_id=session_id,
            uid=uid,
            artifact_id=artifact_id,
            task_id=task.task_id,
            response=None,
            details=details,
            completed_at=completed_at,
        )
        self._receipts.clear_session(session_id)
        submission = MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=self._validator_uid_value(),
            run=run,
            score=0.0,
            execution_log=receipt_log,
            usage=summarized_usage if usage is None else usage,
            session=envelope.session,
        )
        self._record_submission(submission)
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
        return self._record_failure(
            batch_id=batch_id,
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
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
        current_attempt_receipts = self._current_attempt_receipts(
            session_id=issued.session.session_id,
            active_attempt=issued.session.active_attempt,
        )
        envelope = self._sessions.inspect(issued.session.session_id)
        usage, total_tool_usage = self._summarize_receipts(
            envelope=envelope,
            receipts=current_attempt_receipts,
        )
        submission = self._record_failed_submission(
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
            return _retry_decision(exc)
        return _submission_decision(
            self._record_task_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=session_id,
                error_code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                error_message="platform tool proxy execution timed out",
                log_message="miner-owned platform tool proxy execution timed out",
                exc=exc,
            )
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

    def _non_timeout_failure_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: Exception,
        provider_failures: tuple[ProviderFailureEvidence, ...],
        final_attempt: bool,
    ) -> TaskAttemptDecision:
        if _is_provider_caused_terminal_failure(exc):
            provider_failure_decision = self._provider_batch_failure_decision(
                artifact=artifact,
                task=task,
                provider_failures=provider_failures,
                exception_type=_exception_type_name(exc),
            )
            if provider_failure_decision is not None:
                return provider_failure_decision

        if isinstance(exc, LlmRetryExhaustedError):
            return _submission_decision(
                self._record_task_failure(
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
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.MINER_RESPONSE_INVALID,
                    error_message=str(exc),
                    log_message="miner returned invalid response payload",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and is_script_validation_sandbox_invocation(
            detail_code=exc.detail_code,
        ):
            return _submission_decision(
                self._record_task_failure(
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
                self._record_task_failure(
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
                self._record_task_failure(
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

    def _provider_batch_failure_decision(
        self,
        *,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        provider_failures: tuple[ProviderFailureEvidence, ...],
        exception_type: str | None,
    ) -> TaskAttemptDecision | None:
        provider_batch_evidence = provider_batch_failure_evidence(provider_failures)
        if provider_batch_evidence is None:
            return None
        message = provider_batch_failure_message(provider_batch_evidence)
        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.PROVIDER_BATCH_FAILURE,
                message=message,
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.PROVIDER_BATCH_FAILURE,
                    error_message=message,
                    occurred_at=self._clock(),
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=exception_type,
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
                error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                message=message,
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
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
                error_code=error_code,
                error_message=error_message,
                occurred_at=occurred_at or self._clock(),
                exception_type=extra.get("error_type"),
            )
        return None

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
    ) -> MinerTaskAttemptAuditRecord:
        status = MinerTaskAttemptStatus.FAILED
        attempt_receipts = decision.attempt_execution_log or tuple(
            self._receipts.for_session(issued.session.session_id)
        )
        error_code = _attempt_error_code(decision, execution_log=attempt_receipts)
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
        )
        self._progress.record_terminated_attempt(record)
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

    def _validator_uid_value(self) -> int:
        if self._validator_uid is None:
            info = self._subtensor.validator_info()
            self._validator_uid = int(info.uid)
        return self._validator_uid

    def _issue_session(
        self,
        *,
        batch_id: UUID,
        uid: int,
        artifact_id: UUID | None = None,
        task: MinerTask,
        attempt_number: int = 1,
        assignment_token: str | None = None,
    ) -> SessionIssued:
        issued_at = self._clock()
        expires_at = issued_at + self._config.session_ttl
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
            )
        return issued

    def _cleanup_attempt_session(self, session_id: UUID) -> None:
        self._clear_task_session(session_id)
        self._clear_platform_tool_proxy_session(session_id)
        self._sessions.revoke(session_id)
        self._receipts.clear_session(session_id)

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


def _usage_from_receipts(receipts: tuple[ToolCall, ...]) -> SessionUsage:
    llm_usage_totals: dict[str, dict[str, LlmUsageTotals]] = {}
    cost_by_provider: dict[str, float] = {}
    actual_cost_by_provider: dict[str, float] = {}
    total_cost_usd = 0.0
    actual_total_cost_usd: float | None = 0.0
    llm_tokens_last_call = 0

    for receipt in receipts:
        session_cost = _receipt_session_cost(receipt)
        if session_cost is not None:
            provider = _receipt_session_cost_provider(receipt)
            total_cost_usd += session_cost
            cost_by_provider[provider] = cost_by_provider.get(provider, 0.0) + session_cost
        actual_total_cost_usd = _accumulate_receipt_actual_cost(
            actual_total_cost_usd,
            cost_usd=session_cost,
        )
        if session_cost is not None:
            actual_provider = _receipt_session_cost_provider(receipt)
            actual_cost_by_provider[actual_provider] = (
                actual_cost_by_provider.get(actual_provider, 0.0) + session_cost
            )
        if not receipt.is_successful() or receipt.tool != "llm_chat":
            continue
        model = _receipt_llm_model(receipt)
        usage = _receipt_llm_usage(receipt)
        if model is None:
            continue
        if usage is None and session_cost is not None:
            usage = _ReceiptLlmUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                reasoning_tokens=0,
            )
        if usage is None:
            continue
        provider = _receipt_llm_provider(receipt)
        provider_totals = llm_usage_totals.setdefault(provider, {})
        existing = provider_totals.get(model, LlmUsageTotals())
        provider_totals[model] = existing.accumulate(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )
        llm_tokens_last_call = usage.total_tokens

    return SessionUsage(
        total_cost_usd=total_cost_usd,
        cost_by_provider=cost_by_provider,
        reference_total_cost_usd=total_cost_usd,
        reference_cost_by_provider=cost_by_provider,
        actual_total_cost_usd=actual_total_cost_usd,
        actual_cost_by_provider=actual_cost_by_provider,
        llm_tokens_last_call=llm_tokens_last_call,
        llm_usage_totals=llm_usage_totals,
    )


@dataclass(frozen=True, slots=True)
class _ReceiptLlmUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int


def _receipt_llm_usage(receipt: ToolCall) -> _ReceiptLlmUsage | None:
    response_payload = receipt.details.response_payload
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _non_negative_int(usage.get("prompt_tokens"))
    completion_tokens = _non_negative_int(usage.get("completion_tokens"))
    total_tokens = _non_negative_int(usage.get("total_tokens"))
    reasoning_tokens = _non_negative_int(usage.get("reasoning_tokens"))
    resolved_total = (
        total_tokens
        if total_tokens is not None
        else (prompt_tokens or 0) + (completion_tokens or 0) + (reasoning_tokens or 0)
    )
    return _ReceiptLlmUsage(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
        total_tokens=resolved_total,
        reasoning_tokens=reasoning_tokens or 0,
    )


def _receipt_llm_model(receipt: ToolCall) -> str | None:
    request_payload = receipt.details.request_payload
    if not isinstance(request_payload, dict):
        return None
    direct_model = request_payload.get("model")
    if isinstance(direct_model, str):
        return direct_model
    kwargs = request_payload.get("kwargs")
    if isinstance(kwargs, dict):
        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str):
            return kwargs_model
    args = request_payload.get("args")
    if isinstance(args, list) and args:
        first_arg = args[0]
        if isinstance(first_arg, dict):
            arg_model = first_arg.get("model")
            if isinstance(arg_model, str):
                return arg_model
    return None


def _receipt_llm_provider(receipt: ToolCall) -> str:
    provider = _receipt_request_provider(receipt)
    if provider is None or not provider.strip():
        raise ValueError("llm_chat receipt requires request provider")
    return provider.strip()


def _receipt_actual_cost(receipt: ToolCall) -> float | None:
    return None if receipt.details.actual_cost_usd is None else float(receipt.details.actual_cost_usd)


def _receipt_session_cost(receipt: ToolCall) -> float | None:
    cost_usd = None if receipt.details.cost_usd is None else float(receipt.details.cost_usd)
    actual_cost_usd = _receipt_actual_cost(receipt)
    if cost_usd is None:
        if actual_cost_usd is not None:
            raise ValueError("receipt actual_cost_usd requires matching cost_usd")
        return None
    if actual_cost_usd is not None and not isclose(actual_cost_usd, cost_usd, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("receipt cost_usd and actual_cost_usd must match")
    return cost_usd


def _accumulate_receipt_actual_cost(
    current: float | None,
    *,
    cost_usd: float | None,
) -> float | None:
    if cost_usd is None:
        return current
    if current is None:
        return None
    return current + cost_usd


def _receipt_session_cost_provider(receipt: ToolCall) -> str:
    if receipt.tool == "llm_chat":
        _receipt_llm_provider(receipt)
    if receipt.details.actual_cost_provider is not None:
        return receipt.details.actual_cost_provider
    return _receipt_request_cost_provider(receipt)


def _receipt_request_cost_provider(receipt: ToolCall) -> str:
    provider = _receipt_request_provider(receipt)
    if provider is not None and provider.strip():
        return provider.strip()
    if receipt.tool == "llm_chat":
        raise ValueError("llm_chat receipt requires request provider")
    return "desearch" if is_search_tool(receipt.tool) else "chutes"


def _receipt_request_provider(receipt: ToolCall) -> str | None:
    request_payload = receipt.details.request_payload
    if isinstance(request_payload, dict):
        direct_provider = request_payload.get("provider")
        if isinstance(direct_provider, str):
            return direct_provider
        kwargs = request_payload.get("kwargs")
        if isinstance(kwargs, dict):
            kwargs_provider = kwargs.get("provider")
            if isinstance(kwargs_provider, str):
                return kwargs_provider
        args = request_payload.get("args")
        if isinstance(args, list) and args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                arg_provider = first_arg.get("provider")
                if isinstance(arg_provider, str):
                    return arg_provider
    return None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _is_provider_caused_terminal_failure(exc: Exception) -> bool:
    if not isinstance(exc, SandboxInvocationError):
        return False
    return is_provider_caused_terminal_failure(
        detail_code=exc.detail_code,
        detail_exception=exc.detail_exception,
        detail_error=exc.detail_error,
    )


def _submission_decision(
    submission: MinerTaskRunSubmission,
    *,
    attempt_execution_log: tuple[ToolCall, ...] | None = None,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.SUBMISSION,
        submission=submission,
        attempt_execution_log=submission.execution_log if attempt_execution_log is None else attempt_execution_log,
    )


def _retry_decision(exc: Exception) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.RETRY,
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
    *,
    execution_log: Sequence[ToolCall] = (),
) -> str | None:
    proxy_error_code = _platform_tool_proxy_error_code_or_none(execution_log)
    if proxy_error_code is not None:
        return proxy_error_code
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


def _platform_tool_proxy_error_code_or_none(execution_log: Sequence[ToolCall]) -> str | None:
    for receipt in reversed(tuple(execution_log)):
        extra = receipt.details.extra
        if extra is None:
            continue
        error_code = extra.get("platform_tool_proxy_error_code")
        if error_code is not None:
            return str(error_code)
    return None


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
