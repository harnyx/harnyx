"""Service for processing miner-task batches."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.platform import PlatformPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.scheduler import EvaluationScheduler
from harnyx_validator.application.services.evaluation_batch_prep import (
    AgentResolver,
    BatchExecutionPlanner,
    EvaluationBatchConfig,
    RunContext,
)
from harnyx_validator.application.status import StatusProvider

logger = logging.getLogger("harnyx_validator.miner_task_batch")

_LOG_SNIPPET_LIMIT = 512


def _truncate(value: str | None, *, limit: int = _LOG_SNIPPET_LIMIT) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_run_log(
    *,
    batch_id: UUID,
    task: MinerTask,
    submission: MinerTaskRunSubmission,
) -> str:
    run = submission.run
    details = run.details
    parts = [
        f"batch_id={batch_id}",
        f"uid={run.uid}",
        f"artifact_id={run.artifact_id}",
        f"task_id={run.task_id}",
        f"score={submission.score:.3f}",
    ]
    if details.error is not None:
        parts.append(f"error_code={details.error.code}")
    lines = ["Miner task run result " + " ".join(parts)]
    query_text = _truncate(task.query.text)
    if query_text:
        lines.append(f"  query: {query_text}")
    response_text = _truncate(run.response.text if run.response is not None else None)
    if response_text:
        lines.append(f"  response: {response_text}")
    if details.error is not None:
        lines.append(f"  error: {_truncate(details.error.message)}")
    return "\n".join(lines)


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
        orchestrator_factory: Callable[[SandboxClient], TaskRunOrchestrator],
        sandbox_options_factory: Callable[[], SandboxOptions],
        agent_resolver: AgentResolver,
        status_provider: StatusProvider | None = None,
        config: EvaluationBatchConfig | None = None,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._platform = platform_client
        self._status = status_provider
        self._config = config or EvaluationBatchConfig()
        self._planner = BatchExecutionPlanner(
            subtensor_client=subtensor_client,
            sandbox_manager=sandbox_manager,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            receipt_log=receipt_log,
            orchestrator_factory=orchestrator_factory,
            sandbox_options_factory=sandbox_options_factory,
            agent_resolver=agent_resolver,
            progress=progress,
            config=self._config,
        )

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        self._require_platform()
        run_ctx = self._planner.build_run_context(batch)
        self._mark_status_started(run_ctx.batch_id)
        batch_result, elapsed = await self._execute_batch(run_ctx, batch)
        self._complete_batch(run_ctx, batch_result, elapsed)

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

    def _mark_status_failed(self, error_message: str) -> None:
        if self._status is None:
            return
        self._status.state.last_error = error_message
        self._status.state.running = False

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
        try:
            result = await scheduler.run(batch_id=batch_id, requested_artifacts=selected_artifacts)
        except Exception as exc:
            self._mark_status_failed(str(exc))
            raise
        elapsed = time.monotonic() - started
        return result, elapsed

    def _complete_batch(
        self,
        run_ctx: RunContext,
        batch_result: MinerTaskBatchRunResult,
        elapsed_seconds: float,
    ) -> None:
        self._log_results(run_ctx, batch_result, elapsed_seconds)
        self._mark_status_completed()

    def _log_results(
        self,
        run_ctx: RunContext,
        batch_result: MinerTaskBatchRunResult,
        elapsed_seconds: float,
    ) -> None:
        self._log_batch_summary(run_ctx, batch_result)
        self._log_each_run(run_ctx, batch_result)
        self._log_completion(run_ctx.batch_id, elapsed_seconds)

    def _log_batch_summary(self, run_ctx: RunContext, batch_result: MinerTaskBatchRunResult) -> None:
        logger.info(
            "Scheduler returned miner task runs",
            extra={
                "batch_id": str(run_ctx.batch_id),
                "tasks": len(batch_result.tasks),
                "runs": len(batch_result.runs),
            },
        )

    def _log_each_run(self, run_ctx: RunContext, batch_result: MinerTaskBatchRunResult) -> None:
        tasks_by_id = {task.task_id: task for task in batch_result.tasks}
        for submission in batch_result.runs:
            task = tasks_by_id.get(submission.run.task_id)
            if task is None:
                raise RuntimeError(f"task {submission.run.task_id} missing from batch result")
            logger.info(
                _format_run_log(
                    batch_id=run_ctx.batch_id,
                    task=task,
                    submission=submission,
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
