"""Helper to run miner tasks for a single artifact."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from caster_commons.application.dto.session import SessionEnvelope, SessionIssued, SessionTokenRequest
from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.miner_task import EvaluationDetails, EvaluationError, MinerTask
from caster_commons.domain.session import SessionStatus
from caster_commons.domain.tool_usage import ToolUsageSummary
from caster_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from caster_validator.application.evaluate_task_run import TaskRunOrchestrator, UsageSummarizer
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.domain.evaluation import MinerTaskRun

if TYPE_CHECKING:
    from caster_validator.application.scheduler import SchedulerConfig

Clock = Callable[[], datetime]
SubmissionFactory = Callable[[MinerTask, SessionIssued], Awaitable[MinerTaskRunSubmission]]

logger = logging.getLogger("caster_validator.scheduler")


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


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
        async def create_submission(task: MinerTask, issued: SessionIssued) -> MinerTaskRunSubmission:
            return await self._evaluate_task(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                issued=issued,
                orchestrator=orchestrator,
            )

        return await self._run_tasks_with_sessions(
            artifact=artifact,
            tasks=tasks,
            create_submission=create_submission,
        )

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

    async def _evaluate_task(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        orchestrator: TaskRunOrchestrator,
    ) -> MinerTaskRunSubmission:
        request = MinerTaskRunRequest(
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
        )
        try:
            outcome = await orchestrator.evaluate(request)
            return self._record_success(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                outcome=outcome,
            )
        except SandboxInvocationError as exc:
            return self._record_task_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                error_code="sandbox_invocation_failed",
                error_message=str(exc),
                log_message="sandbox invocation failed during miner task run",
                exc=exc,
            )
        except Exception as exc:
            return self._record_task_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                error_code="task_run_failed",
                error_message=str(exc),
                log_message="miner task run failed after sandbox invocation",
                exc=exc,
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


__all__ = ["EvaluationRunner"]
