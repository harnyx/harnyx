"""Helper to run miner tasks for a single artifact."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from harnyx_commons.application.dto.session import SessionEnvelope, SessionIssued, SessionTokenRequest
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import EvaluationDetails, EvaluationError, MinerTask
from harnyx_commons.domain.session import SessionFailureCode, SessionStatus
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, UsageSummarizer
from harnyx_validator.application.invoke_entrypoint import (
    MinerResponseValidationError,
    SandboxInvocationError,
)
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.domain.evaluation import MinerTaskRun

if TYPE_CHECKING:
    from harnyx_validator.application.scheduler import SchedulerConfig

Clock = Callable[[], datetime]
SubmissionFactory = Callable[[MinerTask, SessionIssued], Awaitable[MinerTaskRunSubmission]]

logger = logging.getLogger("harnyx_validator.scheduler")
LOCAL_RETRY_ATTEMPTS = 2
ARTIFACT_TASK_PARALLELISM = 5


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


class FailureKind(StrEnum):
    MINER_TASK_FAILURE = "miner_task_failure"
    VALIDATOR_BATCH_FAILURE = "validator_batch_failure"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    kind: FailureKind
    error_code: str
    error_message: str
    log_message: str
    retryable: bool = False
    exc: Exception | None = None


class AttemptDecisionKind(StrEnum):
    SUBMISSION = "submission"
    VALIDATOR_BATCH_FAILURE = "validator_batch_failure"


@dataclass(frozen=True, slots=True)
class TaskAttemptDecision:
    kind: AttemptDecisionKind
    submission: MinerTaskRunSubmission | None = None
    classification: FailureClassification | None = None


@dataclass(frozen=True, slots=True)
class ValidatorBatchFailureDetail:
    error_code: str
    error_message: str
    occurred_at: datetime
    artifact_id: UUID | None = None
    task_id: UUID | None = None
    uid: int | None = None
    exception_type: str | None = None


class ValidatorBatchFailedError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        failure_detail: ValidatorBatchFailureDetail,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail


@dataclass(slots=True)
class _ArtifactDispatchState:
    submissions_by_index: list[MinerTaskRunSubmission | None]
    validator_failure: ValidatorBatchFailedError | None = None
    unexpected_failure: Exception | None = None


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
        progress: ProgressRecorder | None = None,
        usage_summarizer: UsageSummarizer | None = None,
    ) -> None:
        self._subtensor = subtensor_client
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._receipts = receipt_log
        self._config = config
        self._clock = clock
        self._progress = progress
        self._usage = usage_summarizer or UsageSummarizer()
        self._validator_uid: int | None = None

    async def evaluate_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator: TaskRunOrchestrator,
    ) -> list[MinerTaskRunSubmission]:
        indexed_tasks = tuple(enumerate(tasks))
        if not indexed_tasks:
            return []

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
                )
            )
            for _ in range(min(ARTIFACT_TASK_PARALLELISM, len(indexed_tasks)))
        ]
        await asyncio.gather(*workers)

        if dispatch.validator_failure is not None:
            raise dispatch.validator_failure
        if dispatch.unexpected_failure is not None:
            raise dispatch.unexpected_failure
        return [submission for submission in dispatch.submissions_by_index if submission is not None]

    async def _run_artifact_worker(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        orchestrator: TaskRunOrchestrator,
        pending_tasks: asyncio.Queue[tuple[int, MinerTask]],
        dispatch: _ArtifactDispatchState,
    ) -> None:
        while True:
            if dispatch.validator_failure is not None or dispatch.unexpected_failure is not None:
                return
            try:
                task_index, task = pending_tasks.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                dispatch.submissions_by_index[task_index] = await self._evaluate_task_with_retry(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    orchestrator=orchestrator,
                )
            except ValidatorBatchFailedError as exc:
                if dispatch.validator_failure is None:
                    dispatch.validator_failure = exc
            except Exception as exc:
                if dispatch.unexpected_failure is None:
                    dispatch.unexpected_failure = exc

    async def _run_tasks_with_sessions(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        create_submission: SubmissionFactory,
    ) -> list[MinerTaskRunSubmission]:
        submissions: list[MinerTaskRunSubmission] = []
        for task in tasks:
            issued = self._issue_session(uid=artifact.uid, task=task)
            try:
                submissions.append(await create_submission(task, issued))
            finally:
                self._sessions.revoke(issued.session.session_id)
        return submissions

    async def _evaluate_task_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        orchestrator: TaskRunOrchestrator,
    ) -> MinerTaskRunSubmission:
        issued = self._issue_session(uid=artifact.uid, task=task)
        try:
            for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
                self._sessions.begin_attempt(issued.session.session_id)
                decision = await self._evaluate_task_attempt(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    issued=issued,
                    orchestrator=orchestrator,
                )
                if decision.kind is AttemptDecisionKind.SUBMISSION:
                    return _require_submission(decision)

                classification = _require_classification(decision)
                if classification.retryable and attempt_number < LOCAL_RETRY_ATTEMPTS:
                    self._log_retry_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        attempt_number=attempt_number,
                        exc=_retry_exception_for_classification(classification),
                    )
                    continue

                raise ValidatorBatchFailedError(
                    error_code=classification.error_code,
                    message=classification.error_message,
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code=classification.error_code,
                        error_message=classification.error_message,
                        occurred_at=self._clock(),
                        artifact_id=artifact.artifact_id,
                        task_id=task.task_id,
                        uid=artifact.uid,
                        exception_type=_exception_type_name(classification.exc),
                    ),
                ) from classification.exc
        finally:
            self._sessions.revoke(issued.session.session_id)

        raise RuntimeError("task retry loop exited without returning")

    async def _evaluate_task_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        orchestrator: TaskRunOrchestrator,
    ) -> TaskAttemptDecision:
        request = MinerTaskRunRequest(
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
        )
        try:
            outcome = await orchestrator.evaluate(request)
        except SessionBudgetExhaustedError as exc:
            return _submission_decision(
                self._record_exhausted(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=issued.session.session_id,
                    error_message=str(exc),
                )
            )
        except Exception as exc:
            failure_code = self._sessions.consume_failure_code(issued.session.session_id)
            return self._resolve_attempt_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                failure_code=failure_code,
            )

        failure_code = self._sessions.consume_failure_code(issued.session.session_id)
        if failure_code is not None:
            return self._resolve_attempt_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                failure_code=failure_code,
            )

        return _submission_decision(
            self._record_success(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                outcome=outcome,
            )
        )

    async def record_failure_for_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: str,
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
        error_code: str,
        error_message: str,
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
            error_code="session_budget_exhausted",
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
        error_code: str,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        session_id = envelope.session.session_id
        completed_at = self._clock()
        usage, total_tool_usage = self._summarize_session(envelope)
        details = EvaluationDetails(
            error=EvaluationError(code=error_code, message=error_message),
            total_tool_usage=total_tool_usage,
            elapsed_ms=_elapsed_ms(issued_at=envelope.session.issued_at, completed_at=completed_at),
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
            usage=usage,
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
        error_code: str,
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

    def _resolve_attempt_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: Exception | None = None,
        failure_code: SessionFailureCode | None = None,
    ) -> TaskAttemptDecision:
        classification = self._classify_attempt_failure(
            exc=exc,
            failure_code=failure_code,
        )
        if classification.kind is FailureKind.MINER_TASK_FAILURE:
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=classification.error_code,
                    error_message=classification.error_message,
                    log_message=classification.log_message,
                    exc=_retry_exception_for_classification(classification),
                )
            )
        return _validator_failure_decision(classification)

    def _classify_attempt_failure(
        self,
        *,
        exc: Exception | None,
        failure_code: SessionFailureCode | None = None,
    ) -> FailureClassification:
        if failure_code is SessionFailureCode.TOOL_PROVIDER_FAILED:
            return FailureClassification(
                kind=FailureKind.VALIDATOR_BATCH_FAILURE,
                error_code=failure_code.value,
                error_message=_failure_message_for_session_code(failure_code),
                log_message="validator-hosted tool provider failed during miner task run",
                retryable=True,
                exc=exc,
            )
        if isinstance(exc, LlmRetryExhaustedError):
            return FailureClassification(
                kind=FailureKind.VALIDATOR_BATCH_FAILURE,
                error_code="scoring_llm_retry_exhausted",
                error_message=str(exc),
                log_message="validator scoring provider retries exhausted",
                exc=exc,
            )
        if isinstance(exc, MinerResponseValidationError):
            return FailureClassification(
                kind=FailureKind.MINER_TASK_FAILURE,
                error_code="miner_response_invalid",
                error_message=str(exc),
                log_message="miner returned invalid response payload",
                exc=exc,
            )
        if isinstance(exc, SandboxInvocationError) and exc.detail_code == "UnhandledException":
            return FailureClassification(
                kind=FailureKind.MINER_TASK_FAILURE,
                error_code="miner_unhandled_exception",
                error_message=exc.detail_error or str(exc),
                log_message="miner entrypoint raised unhandled exception",
                exc=exc,
            )
        if isinstance(exc, SandboxInvocationError):
            return FailureClassification(
                kind=FailureKind.VALIDATOR_BATCH_FAILURE,
                error_code="sandbox_invocation_failed",
                error_message=str(exc),
                log_message="sandbox invocation failed during miner task run",
                retryable=True,
                exc=exc,
            )
        if exc is None:
            raise RuntimeError("attempt failure classification requires exc or failure_code")
        return FailureClassification(
            kind=FailureKind.VALIDATOR_BATCH_FAILURE,
            error_code="unexpected_validator_failure",
            error_message=str(exc),
            log_message="validator task execution failed unexpectedly",
            exc=exc,
        )

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
        self._evaluation_records.record(submission)
        if self._progress is not None:
            self._progress.record(submission)

    def _validator_uid_value(self) -> int:
        if self._validator_uid is None:
            info = self._subtensor.validator_info()
            self._validator_uid = int(info.uid)
        return self._validator_uid

    def _issue_session(self, *, uid: int, task: MinerTask) -> SessionIssued:
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
        return self._sessions.issue(request)


def _failure_message_for_session_code(failure_code: SessionFailureCode) -> str:
    if failure_code is SessionFailureCode.TOOL_PROVIDER_FAILED:
        return "tool provider failed"
    raise RuntimeError(f"unsupported session failure code: {failure_code}")


def _submission_decision(submission: MinerTaskRunSubmission) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptDecisionKind.SUBMISSION,
        submission=submission,
    )


def _validator_failure_decision(classification: FailureClassification) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptDecisionKind.VALIDATOR_BATCH_FAILURE,
        classification=classification,
    )


def _require_submission(decision: TaskAttemptDecision) -> MinerTaskRunSubmission:
    if decision.submission is None:
        raise RuntimeError("attempt decision requires submission")
    return decision.submission


def _require_classification(decision: TaskAttemptDecision) -> FailureClassification:
    if decision.classification is None:
        raise RuntimeError("attempt decision requires failure classification")
    return decision.classification


def _retry_exception_for_classification(classification: FailureClassification) -> Exception:
    if classification.exc is not None:
        return classification.exc
    return RuntimeError(classification.error_message)


def _exception_type_name(exc: Exception | None) -> str | None:
    if exc is None:
        return None
    return type(exc).__name__


__all__ = [
    "AttemptDecisionKind",
    "EvaluationRunner",
    "FailureClassification",
    "FailureKind",
    "LOCAL_RETRY_ATTEMPTS",
    "TaskAttemptDecision",
    "ValidatorBatchFailureDetail",
    "ValidatorBatchFailedError",
]
