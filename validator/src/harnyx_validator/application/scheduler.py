"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TypeVar
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask, MinerTaskErrorCode
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_runner import (
    LOCAL_RETRY_ATTEMPTS,
    ArtifactEvaluationOutcome,
    ArtifactExecutionFailedError,
    ArtifactFailure,
    EvaluationRunner,
    TimeoutObservationEvidence,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]
_T = TypeVar("_T")

logger = logging.getLogger("harnyx_validator.scheduler")
BATCH_ARTIFACT_BREAKER_THRESHOLD = 3
_ARTIFACT_PREPARATION_BREAKER_ERROR_CODES: frozenset[MinerTaskErrorCode] = frozenset(
    (
        MinerTaskErrorCode.ARTIFACT_FETCH_FAILED,
        MinerTaskErrorCode.ARTIFACT_STAGING_FAILED,
        MinerTaskErrorCode.ARTIFACT_SETUP_FAILED,
    )
)


@dataclass(frozen=True, slots=True)
class TimeoutRetryState:
    prior_observations: tuple[TimeoutObservationEvidence, ...] = ()


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta
    artifact_task_parallelism: int = 5


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
        blocking_executor: Executor,
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
        self._blocking_executor = blocking_executor
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

        return await self._run_artifacts(
            batch_id=batch_id,
            tasks=tasks,
            artifacts=artifacts,
            blocking_executor=self._blocking_executor,
        )

    async def _run_artifacts(
        self,
        *,
        batch_id: UUID,
        tasks: tuple[MinerTask, ...],
        artifacts: tuple[ScriptArtifactSpec, ...],
        blocking_executor: Executor,
    ) -> MinerTaskBatchRunResult:
        submissions = []
        recorded_pairs = self._progress.recorded_pairs(batch_id) if self._progress is not None else frozenset()
        artifacts_with_breaker: set[UUID] = set()
        successful_baseline_tps: float | None = None
        timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState] = {}
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
            try:
                deployment = await self._start_artifact_with_retry(
                    batch_id=batch_id,
                    artifact=artifact,
                    tasks=remaining_tasks,
                    blocking_executor=blocking_executor,
                )
            except ArtifactExecutionFailedError as exc:
                submissions.extend(
                    await self._record_artifact_failure(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure=exc,
                    )
                )
                if exc.artifact_breaker_tripped:
                    self._raise_if_batch_breaker_tripped(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure_detail=exc.failure_detail,
                        artifacts_with_breaker=artifacts_with_breaker,
                    )
                continue

            batch_breaker_failure: ValidatorBatchFailedError | None = None
            try:
                orchestrator = self._make_orchestrator(deployment.client)
                artifact_result = await self._evaluate_artifact_with_timeout_state(
                    batch_id=batch_id,
                    artifact=artifact,
                    tasks=remaining_tasks,
                    orchestrator=orchestrator,
                    successful_baseline_tps=successful_baseline_tps,
                    timeout_retry_state_by_pair=timeout_retry_state_by_pair,
                )
                submissions.extend(artifact_result.submissions)
                successful_baseline_tps = artifact_result.slowest_successful_tps
                timeout_retry_state_by_pair = {
                    pair_key: TimeoutRetryState(prior_observations=observations)
                    for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
                }
                if artifact_result.artifact_failure is not None:
                    submissions.extend(
                        await self._record_remaining_tasks_for_artifact_failure(
                            batch_id=batch_id,
                            artifact=artifact,
                            failure=artifact_result.artifact_failure,
                            remaining_tasks=artifact_result.unresolved_tasks,
                        )
                    )
                    if artifact_result.artifact_failure.artifact_breaker_tripped:
                        batch_breaker_failure = self._batch_breaker_failure(
                            batch_id=batch_id,
                            artifact=artifact,
                            failure_detail=artifact_result.artifact_failure.failure_detail,
                            artifacts_with_breaker=artifacts_with_breaker,
                        )
            finally:
                await _run_blocking_call(blocking_executor, self._sandboxes.stop, deployment)
            if batch_breaker_failure is not None:
                raise batch_breaker_failure

            logger.debug(
                "finished miner task run for artifact",
                extra={"uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
            )

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            runs=tuple(submissions),
        )

    async def _evaluate_artifact_with_timeout_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        orchestrator: TaskRunOrchestrator,
        successful_baseline_tps: float | None,
        timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState],
    ) -> ArtifactEvaluationOutcome:
        artifact_result = ArtifactEvaluationOutcome(
            submissions=(),
            unresolved_tasks=tasks,
            timeout_observations_by_pair={
                pair_key: state.prior_observations
                for pair_key, state in timeout_retry_state_by_pair.items()
            },
            slowest_successful_tps=successful_baseline_tps,
        )
        current_timeout_states = dict(timeout_retry_state_by_pair)
        while artifact_result.unresolved_tasks:
            artifact_result = await self._runner.evaluate_artifact_with_state(
                batch_id=batch_id,
                artifact=artifact,
                tasks=artifact_result.unresolved_tasks,
                orchestrator=orchestrator,
                successful_baseline_tps=artifact_result.slowest_successful_tps,
                timeout_observations_by_pair={
                    pair_key: state.prior_observations
                    for pair_key, state in current_timeout_states.items()
                },
                earlier_submissions=artifact_result.submissions,
            )
            current_timeout_states = {
                pair_key: TimeoutRetryState(prior_observations=observations)
                for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
            }
            if artifact_result.artifact_failure is not None:
                return artifact_result
        return ArtifactEvaluationOutcome(
            submissions=artifact_result.submissions,
            unresolved_tasks=(),
            timeout_observations_by_pair={
                pair_key: state.prior_observations
                for pair_key, state in current_timeout_states.items()
            },
            slowest_successful_tps=artifact_result.slowest_successful_tps,
        )

    async def _start_artifact_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        blocking_executor: Executor,
    ) -> SandboxDeployment:
        try:
            options = await _run_blocking_call(blocking_executor, self._sandbox_options, artifact)
        except ArtifactPreparationError as exc:
            logger.error(
                "failed to prepare sandbox options",
                extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                exc_info=exc,
            )
            raise self._artifact_execution_failure(
                artifact=artifact,
                tasks=tasks,
                error_code=MinerTaskErrorCode(exc.error_code),
                error_message=str(exc),
                exception_type=exc.exception_type,
                artifact_breaker_tripped=(
                    MinerTaskErrorCode(exc.error_code)
                    in _ARTIFACT_PREPARATION_BREAKER_ERROR_CODES
                ),
            ) from exc
        except Exception as exc:
            logger.error(
                "failed to prepare sandbox options",
                extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                exc_info=exc,
            )
            raise self._artifact_execution_failure(
                artifact=artifact,
                tasks=tasks,
                error_code=MinerTaskErrorCode.ARTIFACT_SETUP_FAILED,
                error_message=str(exc),
                exception_type=type(exc).__name__,
                artifact_breaker_tripped=True,
            ) from exc

        last_error_message = ""
        for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
            try:
                return await _run_blocking_call(blocking_executor, self._sandboxes.start, options)
            except Exception as exc:
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

        raise self._artifact_execution_failure(
            artifact=artifact,
            tasks=tasks,
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            error_message=last_error_message or "artifact setup failed",
            exception_type=None,
            artifact_breaker_tripped=True,
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

    async def _record_artifact_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure: ArtifactExecutionFailedError,
    ) -> list[MinerTaskRunSubmission]:
        submissions = list(failure.completed_submissions)
        submissions.extend(
            await self._runner.record_failure_for_artifact(
                batch_id=batch_id,
                artifact=artifact,
                tasks=failure.remaining_tasks,
                error_code=failure.error_code,
                error_message=str(failure),
            )
        )
        return submissions

    async def _record_remaining_tasks_for_artifact_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure: ArtifactFailure,
        remaining_tasks: tuple[MinerTask, ...],
    ) -> tuple[MinerTaskRunSubmission, ...]:
        return tuple(
            await self._runner.record_failure_for_artifact(
                batch_id=batch_id,
                artifact=artifact,
                tasks=remaining_tasks,
                error_code=failure.error_code,
                error_message=failure.message,
            )
        )

    def _artifact_execution_failure(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
        exception_type: str | None,
        artifact_breaker_tripped: bool = False,
    ) -> ArtifactExecutionFailedError:
        return ArtifactExecutionFailedError(
            error_code=error_code,
            message=error_message,
            failure_detail=ValidatorBatchFailureDetail(
                error_code=error_code,
                error_message=error_message,
                occurred_at=self._clock().astimezone(UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
                exception_type=exception_type,
            ),
            completed_submissions=(),
            remaining_tasks=tuple(tasks),
            artifact_breaker_tripped=artifact_breaker_tripped,
        )

    def _raise_if_batch_breaker_tripped(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure_detail: ValidatorBatchFailureDetail,
        artifacts_with_breaker: set[UUID],
    ) -> None:
        failure = self._batch_breaker_failure(
            batch_id=batch_id,
            artifact=artifact,
            failure_detail=failure_detail,
            artifacts_with_breaker=artifacts_with_breaker,
        )
        if failure is not None:
            raise failure

    def _batch_breaker_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure_detail: ValidatorBatchFailureDetail,
        artifacts_with_breaker: set[UUID],
    ) -> ValidatorBatchFailedError | None:
        artifacts_with_breaker.add(artifact.artifact_id)
        logger.warning(
            "artifact breaker tripped",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "artifacts_with_breaker": len(artifacts_with_breaker),
            },
        )
        if len(artifacts_with_breaker) < BATCH_ARTIFACT_BREAKER_THRESHOLD:
            return None
        return ValidatorBatchFailedError(
            error_code=MinerTaskErrorCode.ARTIFACT_BREAKER_TRIPPED,
            message="validator artifact breaker tripped across 3 artifacts",
            failure_detail=ValidatorBatchFailureDetail(
                error_code=MinerTaskErrorCode.ARTIFACT_BREAKER_TRIPPED,
                error_message="validator artifact breaker tripped across 3 artifacts",
                occurred_at=self._clock().astimezone(UTC),
                artifact_id=failure_detail.artifact_id,
                uid=failure_detail.uid,
                exception_type=failure_detail.exception_type,
            ),
        )


async def _run_blocking_call(
    executor: Executor,
    func: Callable[..., _T],
    /,
    *args: object,
) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args))


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
