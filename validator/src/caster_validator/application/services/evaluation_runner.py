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
from caster_commons.domain.claim import EvaluationClaim
from caster_commons.domain.session import SessionStatus
from caster_validator.application.dto.evaluation import (
    EvaluationCloseout,
    EvaluationOutcome,
    EvaluationRequest,
    ScoredEvaluation,
)
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.invoke_entrypoint import SandboxInvocationError
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort

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

    async def evaluate_miner(
        self,
        *,
        run_id: UUID,
        uid: int,
        claims: Sequence[EvaluationClaim],
        orchestrator: EvaluationOrchestrator,
    ) -> list[ScoredEvaluation]:
        evaluations: list[ScoredEvaluation] = []
        for claim in claims:
            issued = self._issue_session(uid=uid, claim_id=claim.claim_id)
            try:
                outcome = await self._run_evaluation(
                    run_id=run_id,
                    uid=uid,
                    claim=claim,
                    issued=issued,
                    orchestrator=orchestrator,
                )
                if outcome is not None:
                    evaluations.append(outcome)
            finally:
                self._sessions.revoke(issued.session.session_id)
        return evaluations

    async def _run_evaluation(
        self,
        *,
        run_id: UUID,
        uid: int,
        claim: EvaluationClaim,
        issued: SessionIssued,
        orchestrator: EvaluationOrchestrator,
    ) -> ScoredEvaluation | None:
        request = self._build_request(
            session_id=issued.session.session_id,
            token=issued.token,
            uid=uid,
            claim=claim,
        )
        outcome = await self._execute_orchestrator(run_id, uid, claim, orchestrator, request)
        if outcome is None:
            return None
        return self._record_outcome(run_id, issued.session.session_id, outcome)

    async def _execute_orchestrator(
        self,
        run_id: UUID,
        uid: int,
        claim: EvaluationClaim,
        orchestrator: EvaluationOrchestrator,
        request: EvaluationRequest,
    ) -> EvaluationOutcome | None:
        try:
            return await orchestrator.evaluate(request)
        except SandboxInvocationError as exc:
            logger.error(
                "Sandbox invocation failed during evaluation",
                extra={
                    "run_id": str(run_id),
                    "uid": uid,
                    "claim_id": str(claim.claim_id),
                    "entrypoint": request.entrypoint,
                },
                exc_info=exc,
            )
            self._sessions.mark_status(request.session_id, SessionStatus.ERROR)
            return None

    def _record_outcome(
        self,
        run_id: UUID,
        session_id: UUID,
        outcome: EvaluationOutcome,
    ) -> ScoredEvaluation:
        scored = ScoredEvaluation(
            evaluation=outcome.evaluation,
            score=outcome.score,
            usage=outcome.usage,
            total_tool_usage=outcome.total_tool_usage,
        )
        envelope = self._sessions.mark_status(session_id, SessionStatus.COMPLETED)
        closeout = EvaluationCloseout(
            run_id=run_id,
            validator_uid=self._validator_uid_value(),
            outcome=outcome,
            session=envelope.session,
        )
        self._evaluation_records.record(closeout)
        if self._progress is not None:
            self._progress.record(closeout)
        return scored

    def _validator_uid_value(self) -> int:
        if self._validator_uid is None:
            info = self._subtensor.validator_info()
            self._validator_uid = int(info.uid)
        return self._validator_uid

    def _issue_session(self, *, uid: int, claim_id: UUID) -> SessionIssued:
        issued_at = self._clock()
        expires_at = issued_at + self._config.session_ttl
        token = secrets.token_urlsafe(self._config.token_secret_bytes)
        request = SessionTokenRequest(
            session_id=uuid4(),
            uid=uid,
            claim_id=claim_id,
            issued_at=issued_at,
            expires_at=expires_at,
            token=token,
        )
        return self._sessions.issue(request)

    def _build_request(
        self,
        *,
        session_id: UUID,
        token: str,
        uid: int,
        claim: EvaluationClaim,
    ) -> EvaluationRequest:
        return EvaluationRequest(
            session_id=session_id,
            token=token,
            uid=uid,
            entrypoint=self._config.entrypoint,
            payload={
                "claim_text": claim.text,
                "rubric_title": claim.rubric.title,
                "rubric_description": claim.rubric.description,
                "verdict_options": [
                    {"value": entry.value, "description": entry.description}
                    for entry in claim.rubric.verdict_options.options
                ],
            },
            context={
                "claim_id": str(claim.claim_id),
            },
            claim=claim,
            evaluation_id=uuid4(),
        )


__all__ = ["EvaluationRunner"]
