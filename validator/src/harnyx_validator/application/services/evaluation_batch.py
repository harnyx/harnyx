"""Service for processing miner-task batches."""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Executor
from datetime import UTC, datetime, timedelta
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    MinerTaskBatchSpec,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.platform import PlatformPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig
from harnyx_validator.application.services.evaluation_batch_prep import (
    AgentResolver,
    BatchExecutionPlanner,
    EvaluationBatchConfig,
    RunContext,
)
from harnyx_validator.application.services.evaluation_runner import (
    EvaluationRunner,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider

logger = logging.getLogger("harnyx_validator.miner_task_batch")


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
        receipt_log: ReceiptLogPort,
        blocking_executor: Executor,
        orchestrator_factory: Callable[[SandboxClient], TaskRunOrchestrator],
        sandbox_options_factory: Callable[[], SandboxOptions],
        agent_resolver: AgentResolver,
        progress: ProgressRecorder,
        status_provider: StatusProvider | None = None,
        activity: BatchActivityTracker | None = None,
        config: EvaluationBatchConfig | None = None,
    ) -> None:
        self._platform = platform_client
        self._status = status_provider
        self._config = config or EvaluationBatchConfig()
        self._failure_recorder = EvaluationRunner(
            subtensor_client=subtensor_client,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            receipt_log=receipt_log,
            config=SchedulerConfig(
                token_secret_bytes=self._config.token_secret_bytes,
                session_ttl=timedelta(minutes=5),
                artifact_parallelism=self._config.artifact_parallelism,
                artifact_task_parallelism=self._config.artifact_task_parallelism,
            ),
            clock=lambda: datetime.now(UTC),
            progress=progress,
        )
        self._planner = BatchExecutionPlanner(
            subtensor_client=subtensor_client,
            sandbox_manager=sandbox_manager,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            receipt_log=receipt_log,
            blocking_executor=blocking_executor,
            orchestrator_factory=orchestrator_factory,
            sandbox_options_factory=sandbox_options_factory,
            agent_resolver=agent_resolver,
            progress=progress,
            activity=activity,
            config=self._config,
        )

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        self._require_platform()
        self._mark_status_started(batch.batch_id)
        try:
            run_ctx = self._planner.build_run_context(batch)
            batch_result, elapsed = await self._execute_batch(run_ctx, batch)
        except ValidatorBatchFailedError:
            raise
        except Exception as exc:
            raise ValidatorBatchFailedError(
                error_code="batch_execution_failed",
                message=str(exc),
                failure_detail=ValidatorBatchFailureDetail(
                    error_code="batch_execution_failed",
                    error_message=str(exc),
                    occurred_at=datetime.now(UTC),
                    exception_type=type(exc).__name__,
                    traceback=traceback.format_exc(),
                ),
            ) from exc
        self._complete_batch(run_ctx.batch_id, batch_result, elapsed)

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
        self._status.state.last_error = None

    async def _execute_batch(
        self,
        run_ctx: RunContext,
        batch: MinerTaskBatchSpec,
    ) -> tuple[MinerTaskBatchRunResult, float]:
        selected_artifacts, scheduler = self._planner.prepare_execution(run_ctx, batch)
        return await self._run_scheduler_async(run_ctx.batch_id, scheduler, selected_artifacts)

    async def _run_scheduler_async(
        self,
        batch_id: UUID,
        scheduler: EvaluationScheduler,
        selected_artifacts: tuple[ScriptArtifactSpec, ...],
    ) -> tuple[MinerTaskBatchRunResult, float]:
        started = time.monotonic()
        result = await scheduler.run(batch_id=batch_id, requested_artifacts=selected_artifacts)
        elapsed = time.monotonic() - started
        return result, elapsed

    def _complete_batch(
        self,
        batch_id: UUID,
        batch_result: MinerTaskBatchRunResult,
        elapsed_seconds: float,
    ) -> None:
        self._log_results(batch_id, batch_result, elapsed_seconds)
        self._mark_status_completed()

    def _log_results(
        self,
        batch_id: UUID,
        batch_result: MinerTaskBatchRunResult,
        elapsed_seconds: float,
    ) -> None:
        self._log_batch_summary(batch_id, batch_result)
        self._log_completion(batch_id, elapsed_seconds)

    def _log_batch_summary(self, batch_id: UUID, batch_result: MinerTaskBatchRunResult) -> None:
        logger.info(
            "Scheduler returned miner task runs",
            extra={
                "batch_id": str(batch_id),
                "tasks": len(batch_result.tasks),
                "runs": batch_result.completed_run_count,
            },
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
