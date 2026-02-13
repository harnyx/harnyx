"""Helper to run evaluations for a single miner."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from caster_commons.application.dto.session import SessionIssued, SessionTokenRequest
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.claim import MinerTaskClaim
from caster_commons.domain.session import SessionStatus
from caster_commons.json_types import JsonObject
from caster_validator.application.dto.evaluation import (
    EvaluationOutcome,
    EvaluationRequest,
    MinerTaskResult,
    ScoredEvaluation,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.services.evaluation_scoring import EvaluationScore
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation

if TYPE_CHECKING:
    from caster_validator.application.scheduler import SchedulerConfig

Clock = Callable[[], datetime]

logger = logging.getLogger("caster_validator.scheduler")


class EvaluationRunner:
    """Executes evaluations for miners and records outcomes."""

    def __init__(
        self,
        *,
        subtensor_client: SubtensorClientPort,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        config: SchedulerConfig,
        clock: Clock,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._subtensor = subtensor_client
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._config = config
        self._clock = clock
        self._progress = progress
        self._validator_uid: int | None = None

    async def evaluate_candidate(
        self,
        *,
        batch_id: UUID,
        candidate: ScriptArtifactSpec,
        claims: Sequence[MinerTaskClaim],
        orchestrator: EvaluationOrchestrator,
    ) -> list[ScoredEvaluation]:
        uid = candidate.uid
        artifact_id = candidate.artifact_id
        evaluations: list[ScoredEvaluation] = []
        for claim in claims:
            issued = self._issue_session(uid=uid, claim=claim)
            try:
                scored = await self._run_evaluation(
                    batch_id=batch_id,
                    uid=uid,
                    artifact_id=artifact_id,
                    claim=claim,
                    issued=issued,
                    orchestrator=orchestrator,
                )
                if scored is not None:
                    evaluations.append(scored)
            finally:
                self._sessions.revoke(issued.session.session_id)
        return evaluations

    async def _run_evaluation(
        self,
        *,
        batch_id: UUID,
        uid: int,
        artifact_id: UUID,
        claim: MinerTaskClaim,
        issued: SessionIssued,
        orchestrator: EvaluationOrchestrator,
    ) -> ScoredEvaluation | None:
        request = self._build_request(
            session_id=issued.session.session_id,
            token=issued.token,
            uid=uid,
            artifact_id=artifact_id,
            claim=claim,
        )
        outcome, error_code, error_message = await self._execute_orchestrator(
            batch_id, uid, claim, orchestrator, request
        )
        return self._record_result(batch_id, issued.session.session_id, outcome, error_code, error_message)

    async def _execute_orchestrator(
        self,
        batch_id: UUID,
        uid: int,
        claim: MinerTaskClaim,
        orchestrator: EvaluationOrchestrator,
        request: EvaluationRequest,
    ) -> tuple[EvaluationOutcome, str | None, str | None]:
        try:
            return await orchestrator.evaluate(request), None, None
        except SandboxInvocationError as exc:
            logger.error(
                "Sandbox invocation failed during evaluation",
                extra={
                    "batch_id": str(batch_id),
                    "uid": uid,
                    "artifact_id": str(request.artifact_id),
                    "claim_id": str(claim.claim_id),
                    "entrypoint": request.entrypoint,
                },
                exc_info=exc,
            )
            self._sessions.mark_status(request.session_id, SessionStatus.ERROR)
            return (
                self._failure_outcome(uid=uid, artifact_id=request.artifact_id, claim=claim, request=request),
                "sandbox_invocation_failed",
                str(exc),
            )

    def _record_result(
        self,
        batch_id: UUID,
        session_id: UUID,
        outcome: EvaluationOutcome,
        error_code: str | None,
        error_message: str | None,
    ) -> ScoredEvaluation | None:
        scored = ScoredEvaluation(
            criterion_evaluation=outcome.criterion_evaluation,
            score=outcome.score,
            usage=outcome.usage,
            total_tool_usage=outcome.total_tool_usage,
        )
        status = SessionStatus.COMPLETED if error_code is None else SessionStatus.ERROR
        envelope = self._sessions.mark_status(session_id, status)
        result = MinerTaskResult(
            batch_id=batch_id,
            validator_uid=self._validator_uid_value(),
            outcome=outcome,
            session=envelope.session,
            error_code=error_code,
            error_message=error_message,
        )
        self._evaluation_records.record(result)
        if self._progress is not None:
            self._progress.record(result)
        if error_code is None:
            return scored
        return None

    async def record_failure_for_candidate(
        self,
        *,
        batch_id: UUID,
        candidate: ScriptArtifactSpec,
        claims: Sequence[MinerTaskClaim],
        error_code: str,
        error_message: str,
    ) -> None:
        uid = candidate.uid
        artifact_id = candidate.artifact_id
        for claim in claims:
            issued = self._issue_session(uid=uid, claim=claim)
            try:
                request = self._build_request(
                    session_id=issued.session.session_id,
                    token=issued.token,
                    uid=uid,
                    artifact_id=artifact_id,
                    claim=claim,
                )
                self._sessions.mark_status(request.session_id, SessionStatus.ERROR)
                outcome = self._failure_outcome(uid=uid, artifact_id=artifact_id, claim=claim, request=request)
                _ = self._record_result(batch_id, issued.session.session_id, outcome, error_code, error_message)
            finally:
                self._sessions.revoke(issued.session.session_id)

    def _failure_outcome(
        self,
        *,
        uid: int,
        artifact_id: UUID,
        claim: MinerTaskClaim,
        request: EvaluationRequest,
    ) -> EvaluationOutcome:
        answer = MinerAnswer(verdict=-1, justification="execution failed", citations=())
        evaluation = MinerCriterionEvaluation(
            criterion_evaluation_id=request.criterion_evaluation_id,
            session_id=request.session_id,
            uid=uid,
            artifact_id=artifact_id,
            claim_id=claim.claim_id,
            rubric=claim.rubric,
            miner_answer=answer,
            completed_at=self._clock(),
        )
        score = EvaluationScore(
            verdict_score=0.0,
            support_score=0.0,
            justification_pass=False,
            failed_citation_ids=(),
            grader_rationale=None,
        )
        return EvaluationOutcome(
            criterion_evaluation=evaluation,
            score=score,
            tool_receipts=(),
            usage=TokenUsageSummary.empty(),
        )

    def _validator_uid_value(self) -> int:
        if self._validator_uid is None:
            info = self._subtensor.validator_info()
            self._validator_uid = int(info.uid)
        return self._validator_uid

    def _issue_session(self, *, uid: int, claim: MinerTaskClaim) -> SessionIssued:
        issued_at = self._clock()
        expires_at = issued_at + self._config.session_ttl
        token = secrets.token_urlsafe(self._config.token_secret_bytes)
        request = SessionTokenRequest(
            session_id=uuid4(),
            uid=uid,
            claim_id=claim.claim_id,
            issued_at=issued_at,
            expires_at=expires_at,
            budget_usd=claim.budget_usd,
            token=token,
        )
        return self._sessions.issue(request)

    def _build_request(
        self,
        *,
        session_id: UUID,
        token: str,
        uid: int,
        artifact_id: UUID,
        claim: MinerTaskClaim,
    ) -> EvaluationRequest:
        payload: JsonObject = {
            "claim_text": claim.text,
            "rubric_title": claim.rubric.title,
            "rubric_description": claim.rubric.description,
            "verdict_options": [
                {"value": entry.value, "description": entry.description}
                for entry in claim.rubric.verdict_options.options
            ],
        }
        if claim.context is not None:
            payload["context"] = {
                "feed_id": str(claim.context.feed_id),
                "enqueue_seq": claim.context.enqueue_seq,
            }
        return EvaluationRequest(
            session_id=session_id,
            token=token,
            uid=uid,
            artifact_id=artifact_id,
            entrypoint=self._config.entrypoint,
            payload=payload,
            context={
                "claim_id": str(claim.claim_id),
            },
            claim=claim,
            criterion_evaluation_id=uuid4(),
        )


__all__ = ["EvaluationRunner"]
