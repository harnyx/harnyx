"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_runner import (
    LOCAL_RETRY_ATTEMPTS,
    EvaluationRunner,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]

logger = logging.getLogger("harnyx_validator.scheduler")


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta


class EvaluationScheduler:
    """Coordinates issuing sessions and running tasks across artifacts."""

    def __init__(
        self,
        *,
        tasks: Sequence[MinerTask],
        subtensor_client: SubtensorClientPort,
        sandbox_manager: SandboxManager,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        receipt_log: ReceiptLogPort,
        orchestrator_factory: TaskRunOrchestratorFactory,
        sandbox_options_factory: SandboxOptionsFactory,
        clock: Clock,
        config: SchedulerConfig,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._tasks = tuple(tasks)
        self._sandboxes = sandbox_manager
        self._make_orchestrator = orchestrator_factory
        self._sandbox_options = sandbox_options_factory
        self._progress = progress
        self._clock = clock
        self._runner = EvaluationRunner(
            subtensor_client=subtensor_client,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            receipt_log=receipt_log,
            config=config,
            clock=clock,
            progress=progress,
        )

    async def run(
        self,
        *,
        batch_id: UUID,
        requested_artifacts: Sequence[ScriptArtifactSpec],
    ) -> MinerTaskBatchRunResult:
        tasks = self._tasks
        if not tasks:
            raise ValueError("scheduler requires at least one task")

        artifacts = tuple(requested_artifacts)
        if not artifacts:
            raise ValueError("scheduler requires at least one artifact")

        submissions = []
        recorded_pairs = self._progress.recorded_pairs(batch_id) if self._progress is not None else frozenset()
        for artifact in artifacts:
            remaining_tasks = tuple(
                task
                for task in tasks
                if (artifact.artifact_id, task.task_id) not in recorded_pairs
            )
            if not remaining_tasks:
                continue

            logger.debug(
                "starting miner task run for artifact",
                extra={"uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
            )
            deployment = await self._start_artifact_with_retry(
                batch_id=batch_id,
                artifact=artifact,
            )

            try:
                orchestrator = self._make_orchestrator(deployment.client)
                submissions.extend(
                    await self._runner.evaluate_artifact(
                        batch_id=batch_id,
                        artifact=artifact,
                        tasks=remaining_tasks,
                        orchestrator=orchestrator,
                    ),
                )
            finally:
                await asyncio.to_thread(self._sandboxes.stop, deployment)

            logger.debug(
                "finished miner task run for artifact",
                extra={"uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
            )

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            runs=tuple(submissions),
        )

    async def _start_artifact_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
    ) -> SandboxDeployment:
        last_error_code = ""
        last_error_message = ""
        for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
            try:
                options = self._sandbox_options(artifact)
            except Exception as exc:
                last_error_code = "agent_unavailable"
                last_error_message = str(exc)
                if attempt_number < LOCAL_RETRY_ATTEMPTS:
                    self._log_artifact_retry(
                        batch_id=batch_id,
                        artifact=artifact,
                        attempt_number=attempt_number,
                        stage="sandbox options",
                        exc=exc,
                    )
                    continue
                logger.error(
                    "failed to prepare sandbox options",
                    extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                    exc_info=exc,
                )
                break

            try:
                return await asyncio.to_thread(self._sandboxes.start, options)
            except Exception as exc:
                last_error_code = "sandbox_start_failed"
                last_error_message = str(exc)
                if attempt_number < LOCAL_RETRY_ATTEMPTS:
                    self._log_artifact_retry(
                        batch_id=batch_id,
                        artifact=artifact,
                        attempt_number=attempt_number,
                        stage="sandbox start",
                        exc=exc,
                    )
                    continue
                logger.error(
                    "failed to start sandbox",
                    extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                    exc_info=exc,
                )
                break

        raise ValidatorBatchFailedError(
            error_code=last_error_code or "artifact_setup_failed",
            message=last_error_message or "artifact setup failed",
            failure_detail=ValidatorBatchFailureDetail(
                error_code=last_error_code or "artifact_setup_failed",
                error_message=last_error_message or "artifact setup failed",
                occurred_at=self._clock().astimezone(UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
                exception_type=None,
            ),
        )

    def _log_artifact_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        attempt_number: int,
        stage: str,
        exc: Exception,
    ) -> None:
        logger.warning(
            "artifact setup attempt failed; retrying once",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "attempt_number": attempt_number,
                "stage": stage,
            },
            exc_info=exc,
        )


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
