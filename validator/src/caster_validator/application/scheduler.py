"""Batch scheduler orchestrating claim evaluations across miners."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from caster_commons.application.session_manager import SessionManager
from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.manager import SandboxManager
from caster_commons.sandbox.options import SandboxOptions
from caster_validator.application.dto.evaluation import MinerTaskBatchResult, ScriptArtifactSpec
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.ports.claims import ClaimsProviderPort
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.services.evaluation_runner import EvaluationRunner

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
EvaluationOrchestratorFactory = Callable[[SandboxClient], EvaluationOrchestrator]
Clock = Callable[[], datetime]

logger = logging.getLogger("caster_validator.scheduler")


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    entrypoint: str
    token_secret_bytes: int
    session_ttl: timedelta
    budget_usd: float


class EvaluationScheduler:
    """Coordinates issuing sessions and evaluating claims across miners."""

    def __init__(
        self,
        *,
        claims_provider: ClaimsProviderPort,
        subtensor_client: SubtensorClientPort,
        sandbox_manager: SandboxManager,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        orchestrator_factory: EvaluationOrchestratorFactory,
        sandbox_options_factory: SandboxOptionsFactory,
        clock: Clock,
        config: SchedulerConfig,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._claims = claims_provider
        self._subtensor = subtensor_client
        self._sandboxes = sandbox_manager
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._make_orchestrator = orchestrator_factory
        self._sandbox_options = sandbox_options_factory
        self._clock = clock
        self._config = config
        self._progress = progress
        self._runner = EvaluationRunner(
            subtensor_client=subtensor_client,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            config=config,
            clock=clock,
            progress=progress,
        )

    async def run(
        self,
        *,
        batch_id: UUID,
        requested_candidates: Sequence[ScriptArtifactSpec],
    ) -> MinerTaskBatchResult:
        """Run evaluations for the supplied miner-task batch identifier."""
        claims = tuple(self._claims.fetch(batch_id=batch_id))
        if not claims:
            raise ValueError("claims provider returned no entries")

        candidates = tuple(requested_candidates)
        if not candidates:
            raise ValueError("no candidates supplied for evaluation batch")
        evaluations = []

        for candidate in candidates:
            logger.debug(
                "starting evaluation for candidate",
                extra={"uid": candidate.uid, "artifact_id": str(candidate.artifact_id)},
            )
            try:
                options = self._sandbox_options(candidate)
            except Exception as exc:
                logger.error(
                    "Failed to prepare sandbox options",
                    extra={"batch_id": str(batch_id), "uid": candidate.uid, "artifact_id": str(candidate.artifact_id)},
                    exc_info=exc,
                )
                await self._runner.record_failure_for_candidate(
                    batch_id=batch_id,
                    candidate=candidate,
                    claims=claims,
                    error_code="agent_unavailable",
                    error_message=str(exc),
                )
                continue
            try:
                deployment = await asyncio.to_thread(self._sandboxes.start, options)
            except Exception as exc:
                logger.error(
                    "Failed to start sandbox",
                    extra={"batch_id": str(batch_id), "uid": candidate.uid, "artifact_id": str(candidate.artifact_id)},
                    exc_info=exc,
                )
                await self._runner.record_failure_for_candidate(
                    batch_id=batch_id,
                    candidate=candidate,
                    claims=claims,
                    error_code="sandbox_start_failed",
                    error_message=str(exc),
                )
                continue
            try:
                orchestrator = self._make_orchestrator(deployment.client)
                evaluations.extend(
                    await self._runner.evaluate_candidate(
                        batch_id=batch_id,
                        candidate=candidate,
                        claims=claims,
                        orchestrator=orchestrator,
                    ),
                )
            finally:
                await asyncio.to_thread(self._sandboxes.stop, deployment)
            logger.debug(
                "finished evaluation for candidate",
                extra={"uid": candidate.uid, "artifact_id": str(candidate.artifact_id)},
            )

        return MinerTaskBatchResult(
            batch_id=batch_id,
            claims=claims,
            evaluations=tuple(evaluations),
            candidate_uids=tuple(candidate.uid for candidate in candidates),
        )

__all__ = ["EvaluationScheduler", "SchedulerConfig"]
