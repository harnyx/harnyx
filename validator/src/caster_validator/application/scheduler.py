"""Batch scheduler orchestrating claim evaluations across miners."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from caster_commons.application.session_manager import SessionManager
from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.manager import SandboxManager
from caster_commons.sandbox.options import SandboxOptions
from caster_validator.application.dto.evaluation import EvaluationBatchResult
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.ports.claims import ClaimsProviderPort
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.services.evaluation_runner import EvaluationRunner

SandboxOptionsFactory = Callable[[int], SandboxOptions]
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
        run_id: UUID,
        requested_uids: Sequence[int] | None = None,
    ) -> EvaluationBatchResult:
        """Run evaluations for the supplied run identifier."""
        claims = tuple(self._claims.fetch(run_id=run_id))
        if not claims:
            raise ValueError("claims provider returned no entries")

        uids = self._resolve_uids(requested_uids)
        evaluations = []

        for uid in uids:
            logger.debug("starting evaluation for miner", extra={"uid": uid})
            deployment = self._sandboxes.start(self._sandbox_options(uid))
            try:
                orchestrator = self._make_orchestrator(deployment.client)
                evaluations.extend(
                    await self._runner.evaluate_miner(
                        run_id=run_id,
                        uid=uid,
                        claims=claims,
                        orchestrator=orchestrator,
                    ),
                )
            finally:
                self._sandboxes.stop(deployment)
            logger.debug("finished evaluation for miner", extra={"uid": uid})

        return EvaluationBatchResult(
            run_id=run_id,
            claims=claims,
            evaluations=tuple(evaluations),
            uids=uids,
        )

    def _resolve_uids(self, requested: Sequence[int] | None) -> tuple[int, ...]:
        if requested:
            return tuple(dict.fromkeys(requested))
        snapshot = self._subtensor.fetch_metagraph()
        if not snapshot.uids:
            raise ValueError("metagraph did not return any miner UIDs")
        return tuple(int(uid) for uid in snapshot.uids)

__all__ = ["EvaluationScheduler", "SchedulerConfig"]
