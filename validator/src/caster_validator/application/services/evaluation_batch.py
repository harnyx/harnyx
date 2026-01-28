"""Service for processing miner-task batches."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.claim import MinerTaskClaim, Rubric
from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.docker import DockerSandboxManager
from caster_commons.sandbox.options import SandboxOptions
from caster_validator.application.dto.evaluation import (
    MinerTaskBatchResult,
    MinerTaskBatchSpec,
    ScoredEvaluation,
    ScriptArtifactSpec,
)
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.platform import PlatformPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.scheduler import EvaluationScheduler
from caster_validator.application.services.evaluation_batch_prep import (
    AgentResolver,
    BatchExecutionPlanner,
    EvaluationBatchConfig,
    RunContext,
)
from caster_validator.application.status import StatusProvider
from caster_validator.domain.evaluation import MinerAnswer, MinerCriterionEvaluation

logger = logging.getLogger("caster_validator.miner_task_batch")

_LOG_SNIPPET_LIMIT = 512


def _truncate(value: str | None, *, limit: int = _LOG_SNIPPET_LIMIT) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_evaluation_log(
    *,
    batch_id: UUID,
    evaluation: MinerCriterionEvaluation,
    claim: MinerTaskClaim | None,
    rubric: Rubric,
    miner_answer: MinerAnswer,
    total_score: float,
) -> str:
    parts = [
        f"batch_id={batch_id}",
        f"uid={evaluation.uid}",
        f"claim_id={evaluation.claim_id}",
        f"verdict={miner_answer.verdict}",
        f"score_total={total_score:.3f}",
        f"citations={len(miner_answer.citations)}",
    ]
    lines = ["Miner criterion evaluation result " + " ".join(parts)]
    claim_text = _truncate(claim.text if claim is not None else None)
    if claim_text:
        lines.append(f"  claim: {claim_text}")
    lines.append(f"  rubric: {rubric.title} — {_truncate(rubric.description)}")
    justification = _truncate(miner_answer.justification)
    if justification:
        lines.append(f"  justification: {justification}")
    return "\n".join(lines)


# Type aliases for factories
class MinerTaskBatchService:
    """Processes miner-task batches by coordinating sandbox and scheduler."""

    def __init__(
        self,
        *,
        platform_client: PlatformPort | None,
        subtensor_client: SubtensorClientPort,
        sandbox_manager: DockerSandboxManager,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        orchestrator_factory: Callable[[SandboxClient], EvaluationOrchestrator],
        sandbox_options_factory: Callable[[], SandboxOptions],
        agent_resolver: AgentResolver,
        status_provider: StatusProvider | None = None,
        config: EvaluationBatchConfig | None = None,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._platform = platform_client
        self._subtensor = subtensor_client
        self._sandbox_manager = sandbox_manager
        self._session_manager = session_manager
        self._evaluation_records = evaluation_records
        self._status = status_provider
        self._config = config or EvaluationBatchConfig()
        self._progress = progress
        self._planner = BatchExecutionPlanner(
            subtensor_client=self._subtensor,
            sandbox_manager=self._sandbox_manager,
            session_manager=self._session_manager,
            evaluation_records=self._evaluation_records,
            orchestrator_factory=orchestrator_factory,
            sandbox_options_factory=sandbox_options_factory,
            agent_resolver=agent_resolver,
            progress=progress,
            config=self._config,
        )

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        """Process a single miner-task batch."""
        self._require_platform()
        run_ctx = self._planner.build_run_context(batch)
        self._mark_status_started(run_ctx.batch_id)

        selected_candidates, scheduler = self._planner.prepare_execution(run_ctx, batch)
        batch_result, elapsed = await self._run_scheduler_async(run_ctx.batch_id, scheduler, selected_candidates)

        self._log_results(run_ctx, batch_result, elapsed)
        self._mark_status_completed()

    def process(self, batch: MinerTaskBatchSpec) -> None:
        asyncio.run(self.process_async(batch))

    def _require_platform(self) -> None:
        if self._platform is None:
            raise RuntimeError("platform client is not configured")

    def _mark_status_started(self, batch_id: UUID) -> None:
        if self._status is None:
            return
        self._status.state.last_batch_id = batch_id
        self._status.state.last_started_at = datetime.now(UTC)
        self._status.state.running = True
        self._status.state.last_error = None

    def _mark_status_completed(self) -> None:
        if self._status is None:
            return
        self._status.state.last_completed_at = datetime.now(UTC)
        self._status.state.running = False

    async def _run_scheduler_async(
        self,
        batch_id: UUID,
        scheduler: EvaluationScheduler,
        selected_candidates: tuple[ScriptArtifactSpec, ...],
    ) -> tuple[MinerTaskBatchResult, float]:
        started = time.monotonic()
        try:
            result = await scheduler.run(batch_id=batch_id, requested_candidates=selected_candidates)
        except Exception as exc:
            if self._status is not None:
                self._status.state.last_error = str(exc)
                self._status.state.running = False
            raise
        elapsed = time.monotonic() - started
        return result, elapsed

    def _log_results(
        self,
        run_ctx: RunContext,
        batch_result: MinerTaskBatchResult,
        elapsed_seconds: float,
    ) -> None:
        self._log_batch_summary(run_ctx, batch_result)
        self._log_each_evaluation(run_ctx, batch_result)
        self._log_completion(run_ctx.batch_id, elapsed_seconds)

    def _log_batch_summary(self, run_ctx: RunContext, batch_result: MinerTaskBatchResult) -> None:
        logger.info(
            "Scheduler returned criterion evaluations",
            extra={
                "batch_id": str(run_ctx.batch_id),
                "claims": len(batch_result.claims),
                "evaluations": len(batch_result.evaluations),
                "candidates": len(batch_result.candidate_uids),
            },
        )

    def _log_each_evaluation(self, run_ctx: RunContext, batch_result: MinerTaskBatchResult) -> None:
        claims_by_id = {claim.claim_id: claim for claim in batch_result.claims}
        for scored in batch_result.evaluations:
            self._log_single_evaluation(run_ctx, scored, claims_by_id)

    def _log_single_evaluation(
        self,
        run_ctx: RunContext,
        scored: ScoredEvaluation,
        claims_by_id: dict[UUID, MinerTaskClaim],
    ) -> None:
        evaluation = scored.criterion_evaluation
        claim = claims_by_id.get(evaluation.claim_id)
        rubric = claim.rubric if claim is not None else evaluation.rubric
        logger.info(
            _format_evaluation_log(
                batch_id=run_ctx.batch_id,
                evaluation=evaluation,
                claim=claim,
                rubric=rubric,
                miner_answer=evaluation.miner_answer,
                total_score=scored.score.total,
            ),
        )

    def _log_completion(self, batch_id: UUID, elapsed_seconds: float) -> None:
        logger.info(
            "Miner-task batch completed",
            extra={
                "batch_id": str(batch_id),
                "elapsed_seconds": round(elapsed_seconds, 2),
            },
        )


__all__ = [
    "EvaluationBatchConfig",
    "MinerTaskBatchService",
]
