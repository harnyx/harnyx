"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from types import TracebackType
from typing import TypeVar
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    MinerTask,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.miner_task_failure_policy import ValidatorModelLlmBaseline
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
    EvaluationRunner,
    TimeoutObservationEvidence,
    UnexpectedArtifactExecutionError,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.status import BatchActivityTracker
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]
_T = TypeVar("_T")

logger = logging.getLogger("harnyx_validator.scheduler")
measurement_logger = logging.getLogger("harnyx_validator.measurement")
_BATCH_FAILURE_PROGRESS_PAGE_SIZE = 500


@dataclass(frozen=True, slots=True)
class TimeoutRetryState:
    prior_observations: tuple[TimeoutObservationEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class _ArtifactWorkItem:
    artifact_index: int
    artifact: ScriptArtifactSpec
    tasks: tuple[MinerTask, ...]
    earlier_submissions: tuple[MinerTaskRunSubmission, ...] = ()
    timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState] = field(default_factory=dict)
    retry_round: int = 0

    @property
    def owned_pair_keys(self) -> frozenset[tuple[UUID, UUID]]:
        return frozenset((self.artifact.artifact_id, task.task_id) for task in self.tasks)


class _ArtifactWorkCoordinator:
    def __init__(self, normal_work: Sequence[_ArtifactWorkItem]) -> None:
        self._normal_queue = deque(normal_work)
        self._retry_queues_by_round: defaultdict[int, deque[_ArtifactWorkItem]] = defaultdict(deque)
        self._active_normal_count = 0
        self._active_retry_count_by_round: Counter[int] = Counter()
        self._stopped = False
        self._condition = asyncio.Condition()

    async def next_work(self) -> _ArtifactWorkItem | None:
        async with self._condition:
            while True:
                if self._stopped:
                    return None
                if self._normal_queue:
                    self._active_normal_count += 1
                    return self._normal_queue.popleft()
                if self._active_normal_count:
                    await self._condition.wait()
                    continue
                lowest_round = self._lowest_queued_retry_round()
                if lowest_round is None:
                    if not self._active_retry_count():
                        return None
                    await self._condition.wait()
                    continue
                if self._has_active_lower_retry_round(lowest_round):
                    await self._condition.wait()
                    continue
                self._active_retry_count_by_round[lowest_round] += 1
                return self._retry_queues_by_round[lowest_round].popleft()

    async def requeue_retry(self, work_item: _ArtifactWorkItem) -> None:
        async with self._condition:
            self._retry_queues_by_round[work_item.retry_round].append(work_item)
            self._condition.notify_all()

    async def complete_work(self, work_item: _ArtifactWorkItem) -> None:
        async with self._condition:
            if work_item.retry_round == 0:
                self._active_normal_count -= 1
            else:
                self._active_retry_count_by_round[work_item.retry_round] -= 1
                if self._active_retry_count_by_round[work_item.retry_round] <= 0:
                    del self._active_retry_count_by_round[work_item.retry_round]
            self._condition.notify_all()

    async def stop(self) -> None:
        async with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _lowest_queued_retry_round(self) -> int | None:
        queued_rounds = (
            retry_round
            for retry_round, queue in self._retry_queues_by_round.items()
            if queue
        )
        return min(queued_rounds, default=None)

    def _has_active_lower_retry_round(self, retry_round: int) -> bool:
        return any(
            active_round < retry_round and count > 0
            for active_round, count in self._active_retry_count_by_round.items()
        )

    def _active_retry_count(self) -> int:
        return sum(self._active_retry_count_by_round.values())


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta
    artifact_parallelism: int = 4
    artifact_task_parallelism: int = 20


class _BatchTaskSessionLimiter:
    def __init__(
        self,
        max_sessions: int,
        *,
        batch_id: UUID,
        activity: BatchActivityTracker | None,
    ) -> None:
        self.max_sessions = max(1, max_sessions)
        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self._active_count = 0
        self._batch_id = batch_id
        self._activity = activity

    async def __aenter__(self) -> None:
        await self._semaphore.acquire()
        self._active_count += 1
        if self._activity is not None:
            self._activity.mark_task_session_started(self._batch_id)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._active_count -= 1
        if self._activity is not None:
            self._activity.mark_task_session_finished(self._batch_id)
        self._semaphore.release()

    @property
    def active_count(self) -> int:
        return self._active_count


@dataclass(frozen=True, slots=True)
class _CompletedArtifactResult:
    artifact_id: UUID
    submissions: tuple[MinerTaskRunSubmission, ...]
    validator_model_llm_baseline: ValidatorModelLlmBaseline
    timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState]
    unresolved_tasks: tuple[MinerTask, ...] = ()
    retry_round: int = 0
    validator_batch_failure: ValidatorBatchFailedError | None = None


@dataclass(slots=True)
class _BatchArtifactDispatchState:
    completed_run_count: int = 0
    validator_model_llm_baseline: ValidatorModelLlmBaseline = field(
        default_factory=ValidatorModelLlmBaseline.empty
    )
    timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState] = field(default_factory=dict)
    stop_dequeuing: bool = False
    published_batch_failure: ValidatorBatchFailedError | None = None
    validator_batch_failures_by_artifact_index: dict[int, ValidatorBatchFailedError] = field(default_factory=dict)
    merge_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _monotonic_elapsed_ms(*, started_at: float, completed_at: float) -> float:
    return round((completed_at - started_at) * 1000.0, 3)


def _count_submission_outcomes(
    submissions: Sequence[MinerTaskRunSubmission],
) -> tuple[int, int]:
    success_count = 0
    failure_count = 0
    for submission in submissions:
        if submission.run.details.error is None:
            success_count += 1
            continue
        failure_count += 1
    return success_count, failure_count


def _completed_submission_delta(
    *,
    submissions: Sequence[MinerTaskRunSubmission],
    earlier_submissions: Sequence[MinerTaskRunSubmission],
) -> int:
    earlier_pairs = {_submission_pair(submission) for submission in earlier_submissions}
    current_pairs = {_submission_pair(submission) for submission in submissions}
    return len(current_pairs - earlier_pairs)


def _submission_pair(submission: MinerTaskRunSubmission) -> tuple[UUID, UUID]:
    return (submission.run.artifact_id, submission.run.task_id)


def _copy_owned_timeout_states(
    *,
    dispatch_states: dict[tuple[UUID, UUID], TimeoutRetryState],
    work_item_states: dict[tuple[UUID, UUID], TimeoutRetryState],
    owned_pair_keys: frozenset[tuple[UUID, UUID]],
) -> dict[tuple[UUID, UUID], TimeoutRetryState]:
    timeout_states: dict[tuple[UUID, UUID], TimeoutRetryState] = {}
    for pair_key in owned_pair_keys:
        state = _newer_timeout_state(
            dispatch_states.get(pair_key),
            work_item_states.get(pair_key),
        )
        if state is None:
            continue
        timeout_states[pair_key] = TimeoutRetryState(prior_observations=state.prior_observations)
    return timeout_states


def _merge_owned_timeout_states(
    *,
    target: dict[tuple[UUID, UUID], TimeoutRetryState],
    updates: dict[tuple[UUID, UUID], TimeoutRetryState],
    owned_pair_keys: frozenset[tuple[UUID, UUID]],
) -> None:
    for pair_key in owned_pair_keys:
        update = updates.get(pair_key)
        if update is None:
            target.pop(pair_key, None)
            continue
        current = target.get(pair_key)
        target[pair_key] = _newer_timeout_state(current, update) or update


def _newer_timeout_state(
    left: TimeoutRetryState | None,
    right: TimeoutRetryState | None,
) -> TimeoutRetryState | None:
    if left is None:
        return right
    if right is None:
        return left
    return right if len(right.prior_observations) > len(left.prior_observations) else left


def _has_primary_artifact_outcome(
    *,
    outcome: str,
    primary_failure_raised: bool,
) -> bool:
    _ = outcome
    return primary_failure_raised


def _log_batch_execution_started(
    *,
    batch_id: UUID,
    artifact_count: int,
    task_count: int,
    artifact_parallelism: int,
    artifact_task_parallelism: int,
    recorded_pair_count: int,
) -> None:
    measurement_logger.info(
        "miner-task batch execution started",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "artifact_count": artifact_count,
                "task_count": task_count,
                "artifact_parallelism": artifact_parallelism,
                "artifact_task_parallelism": artifact_task_parallelism,
                "recorded_pair_count": recorded_pair_count,
            }
        },
    )


def _log_artifact_execution_finished(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    artifact_index: int,
    artifact_count: int,
    planned_task_count: int,
    success_count: int,
    failure_count: int,
    unresolved_count: int,
    setup_ms: float,
    evaluation_ms: float,
    teardown_ms: float,
    total_ms: float,
    outcome: str,
    error_code: str | None,
) -> None:
    measurement_logger.info(
        "miner-task artifact execution finished",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "artifact_id": str(artifact.artifact_id),
                "uid": artifact.uid,
                "artifact_index": artifact_index,
                "artifact_count": artifact_count,
                "planned_task_count": planned_task_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "unresolved_count": unresolved_count,
                "setup_ms": setup_ms,
                "evaluation_ms": evaluation_ms,
                "teardown_ms": teardown_ms,
                "total_ms": total_ms,
                "outcome": outcome,
                "error_code": error_code,
            }
        },
    )


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
        progress: ProgressRecorder,
        activity: BatchActivityTracker | None = None,
    ) -> None:
        self._tasks = tuple(tasks)
        self._sandboxes = sandbox_manager
        self._make_orchestrator = orchestrator_factory
        self._sandbox_options = sandbox_options_factory
        self._progress = progress
        self._clock = clock
        self._blocking_executor = blocking_executor
        self._config = config
        self._activity = activity
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
        recorded_pairs = self._progress.recorded_pairs(batch_id)
        recorded_progress_sequence = self._progress.summary(batch_id)["latest_sequence"]
        artifact_parallelism = min(max(1, self._config.artifact_parallelism), len(artifacts))
        task_session_limiter = _BatchTaskSessionLimiter(
            self._config.artifact_task_parallelism,
            batch_id=batch_id,
            activity=self._activity,
        )
        if self._activity is not None:
            self._activity.mark_batch_started(batch_id)
        _log_batch_execution_started(
            batch_id=batch_id,
            artifact_count=len(artifacts),
            task_count=len(tasks),
            artifact_parallelism=artifact_parallelism,
            artifact_task_parallelism=self._config.artifact_task_parallelism,
            recorded_pair_count=len(recorded_pairs),
        )
        dispatch = _BatchArtifactDispatchState()
        work_coordinator = _ArtifactWorkCoordinator(
            tuple(
                _ArtifactWorkItem(
                    artifact_index=artifact_index,
                    artifact=artifact,
                    tasks=tasks,
                )
                for artifact_index, artifact in enumerate(artifacts, start=1)
            )
        )

        workers = [
            asyncio.create_task(
                self._run_artifact_worker(
                    batch_id=batch_id,
                    artifact_count=len(artifacts),
                    recorded_pairs=recorded_pairs,
                    work_coordinator=work_coordinator,
                    dispatch=dispatch,
                    blocking_executor=blocking_executor,
                    task_session_limiter=task_session_limiter,
                )
            )
            for _ in range(artifact_parallelism)
        ]
        try:
            try:
                await asyncio.gather(*workers)
            except BaseException:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                raise
        finally:
            if self._activity is not None:
                self._activity.mark_batch_finished(batch_id)

        if dispatch.published_batch_failure is not None:
            raise self._build_final_batch_failure(
                batch_id=batch_id,
                dispatch=dispatch,
                completed_runs_after_sequence=recorded_progress_sequence,
                recorded_pairs=recorded_pairs,
            )

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            completed_run_count=dispatch.completed_run_count,
        )

    async def _run_artifact_worker(
        self,
        *,
        batch_id: UUID,
        artifact_count: int,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
        work_coordinator: _ArtifactWorkCoordinator,
        dispatch: _BatchArtifactDispatchState,
        blocking_executor: Executor,
        task_session_limiter: _BatchTaskSessionLimiter,
    ) -> None:
        while True:
            work_item = await work_coordinator.next_work()
            if work_item is None:
                return

            async with dispatch.merge_lock:
                if dispatch.stop_dequeuing or dispatch.published_batch_failure is not None:
                    should_stop_coordinator = True
                    retry_work_item = None
                    timeout_retry_state_snapshot: dict[tuple[UUID, UUID], TimeoutRetryState] | None = None
                else:
                    should_stop_coordinator = False
                    retry_work_item = None
                    timeout_retry_state_snapshot = _copy_owned_timeout_states(
                        dispatch_states=dispatch.timeout_retry_state_by_pair,
                        work_item_states=work_item.timeout_retry_state_by_pair,
                        owned_pair_keys=work_item.owned_pair_keys,
                    )

            if should_stop_coordinator:
                await work_coordinator.stop()
                await work_coordinator.complete_work(work_item)
                return
            if timeout_retry_state_snapshot is None:
                await work_coordinator.complete_work(work_item)
                return

            artifact_result = await self._run_single_artifact(
                batch_id=batch_id,
                artifact_index=work_item.artifact_index,
                artifact_count=artifact_count,
                artifact=work_item.artifact,
                tasks=work_item.tasks,
                recorded_pairs=recorded_pairs,
                blocking_executor=blocking_executor,
                completed_artifact_baseline=lambda: dispatch.validator_model_llm_baseline,
                timeout_retry_state_snapshot=timeout_retry_state_snapshot,
                earlier_submissions=work_item.earlier_submissions,
                retry_round=work_item.retry_round,
                stop_dequeuing=lambda: self._stop_artifact_dequeue(dispatch),
                task_session_limiter=task_session_limiter,
            )

            async with dispatch.merge_lock:
                dispatch.completed_run_count += _completed_submission_delta(
                    submissions=artifact_result.submissions,
                    earlier_submissions=work_item.earlier_submissions,
                )
                dispatch.validator_model_llm_baseline = dispatch.validator_model_llm_baseline.merge(
                    artifact_result.validator_model_llm_baseline,
                )
                _merge_owned_timeout_states(
                    target=dispatch.timeout_retry_state_by_pair,
                    updates=artifact_result.timeout_retry_state_by_pair,
                    owned_pair_keys=work_item.owned_pair_keys,
                )
                if artifact_result.validator_batch_failure is not None:
                    dispatch.validator_batch_failures_by_artifact_index[work_item.artifact_index] = (
                        artifact_result.validator_batch_failure
                    )
                    if dispatch.published_batch_failure is None:
                        dispatch.published_batch_failure = artifact_result.validator_batch_failure
                    dispatch.stop_dequeuing = True
                    should_stop_coordinator = True
                    retry_work_item = None
                elif artifact_result.unresolved_tasks:
                    retry_work_item = _ArtifactWorkItem(
                        artifact_index=work_item.artifact_index,
                        artifact=work_item.artifact,
                        tasks=artifact_result.unresolved_tasks,
                        earlier_submissions=artifact_result.submissions,
                        timeout_retry_state_by_pair={
                            pair_key: state
                            for pair_key, state in artifact_result.timeout_retry_state_by_pair.items()
                            if pair_key in work_item.owned_pair_keys
                        },
                        retry_round=work_item.retry_round + 1,
                    )
                    should_stop_coordinator = False
                else:
                    should_stop_coordinator = False
                    retry_work_item = None

            if should_stop_coordinator:
                await work_coordinator.stop()
            elif retry_work_item is not None:
                await work_coordinator.requeue_retry(retry_work_item)
            await work_coordinator.complete_work(work_item)
            if should_stop_coordinator or retry_work_item is not None:
                continue

    async def _evaluate_artifact_with_timeout_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        orchestrator: TaskRunOrchestrator,
        validator_model_llm_baseline: ValidatorModelLlmBaseline,
        completed_artifact_baseline: Callable[[], ValidatorModelLlmBaseline] | None = None,
        timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState],
        earlier_submissions: tuple[MinerTaskRunSubmission, ...] = (),
        task_session_limiter: _BatchTaskSessionLimiter | None = None,
    ) -> ArtifactEvaluationOutcome:
        artifact_result = await self._runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=tasks,
            orchestrator=orchestrator,
            validator_model_llm_baseline=validator_model_llm_baseline,
            completed_artifact_baseline=completed_artifact_baseline,
            timeout_observations_by_pair={
                pair_key: state.prior_observations
                for pair_key, state in timeout_retry_state_by_pair.items()
            },
            earlier_submissions=earlier_submissions,
            task_session_limiter=task_session_limiter,
        )
        return ArtifactEvaluationOutcome(
            submissions=artifact_result.submissions,
            unresolved_tasks=artifact_result.unresolved_tasks,
            timeout_observations_by_pair=artifact_result.timeout_observations_by_pair,
            validator_model_llm_baseline=artifact_result.validator_model_llm_baseline,
        )

    async def _run_single_artifact(
        self,
        *,
        batch_id: UUID,
        artifact_index: int,
        artifact_count: int,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        recorded_pairs: frozenset[tuple[UUID, UUID]],
        blocking_executor: Executor,
        completed_artifact_baseline: Callable[[], ValidatorModelLlmBaseline],
        timeout_retry_state_snapshot: dict[tuple[UUID, UUID], TimeoutRetryState],
        earlier_submissions: tuple[MinerTaskRunSubmission, ...] = (),
        retry_round: int = 0,
        stop_dequeuing: Callable[[], None] | None = None,
        task_session_limiter: _BatchTaskSessionLimiter | None = None,
    ) -> _CompletedArtifactResult:
        remaining_tasks = tuple(
            task for task in tasks if (artifact.artifact_id, task.task_id) not in recorded_pairs
        )
        if not remaining_tasks:
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=earlier_submissions,
                validator_model_llm_baseline=completed_artifact_baseline(),
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                retry_round=retry_round,
            )

        if self._activity is not None:
            self._activity.mark_artifact_started(batch_id, artifact.artifact_id)
        artifact_started_at = time.monotonic()
        setup_ms = 0.0
        evaluation_ms = 0.0
        teardown_ms = 0.0
        artifact_submissions: tuple[MinerTaskRunSubmission, ...] = earlier_submissions
        unresolved_count = len(remaining_tasks)
        outcome = "completed"
        error_code: str | None = None
        backfill_primary_outcome: tuple[str, str] | None = None
        try:
            if self._activity is not None:
                self._activity.mark_artifact_stage(batch_id, "sandbox_setup_started")
            setup_started_at = time.monotonic()
            deployment = await self._start_artifact_with_retry(
                batch_id=batch_id,
                artifact=artifact,
                tasks=remaining_tasks,
                blocking_executor=blocking_executor,
            )
            if self._activity is not None:
                self._activity.mark_artifact_stage(batch_id, "sandbox_setup_finished")
            setup_ms = _monotonic_elapsed_ms(
                started_at=setup_started_at,
                completed_at=time.monotonic(),
            )
        except ArtifactExecutionFailedError as exc:
            setup_ms = _monotonic_elapsed_ms(
                started_at=setup_started_at,
                completed_at=time.monotonic(),
            )
            if is_delivery_disqualifying_validator_pair_error(exc.error_code):
                _log_artifact_execution_finished(
                    batch_id=batch_id,
                    artifact=artifact,
                    artifact_index=artifact_index,
                    artifact_count=artifact_count,
                    planned_task_count=len(remaining_tasks),
                    success_count=0,
                    failure_count=0,
                    unresolved_count=len(remaining_tasks),
                    setup_ms=setup_ms,
                    evaluation_ms=0.0,
                    teardown_ms=0.0,
                    total_ms=_monotonic_elapsed_ms(
                        started_at=artifact_started_at,
                        completed_at=time.monotonic(),
                    ),
                    outcome="validator_batch_failure",
                    error_code=str(exc.error_code),
                )
                if stop_dequeuing is not None:
                    stop_dequeuing()
                if self._activity is not None:
                    self._activity.mark_artifact_finished(batch_id, artifact.artifact_id)
                return _CompletedArtifactResult(
                    artifact_id=artifact.artifact_id,
                    submissions=earlier_submissions,
                    validator_model_llm_baseline=completed_artifact_baseline(),
                    timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                    retry_round=retry_round,
                    validator_batch_failure=self._conclusive_batch_failure_from_artifact_error(
                        artifact=artifact,
                        tasks=tasks,
                        completed_submissions=earlier_submissions,
                        failure=exc,
                        recorded_pairs=recorded_pairs,
                    ),
                )
            try:
                artifact_submissions = (
                    *earlier_submissions,
                    *(await self._record_artifact_failure(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure=exc,
                    )),
                )
                unresolved_count = 0
            except UnexpectedArtifactExecutionError as backfill_exc:
                artifact_submissions = (*earlier_submissions, *backfill_exc.completed_submissions)
                unresolved_count = len(backfill_exc.remaining_tasks)
                success_count, failure_count = _count_submission_outcomes(artifact_submissions)
                _log_artifact_execution_finished(
                    batch_id=batch_id,
                    artifact=artifact,
                    artifact_index=artifact_index,
                    artifact_count=artifact_count,
                    planned_task_count=len(remaining_tasks),
                    success_count=success_count,
                    failure_count=failure_count,
                    unresolved_count=unresolved_count,
                    setup_ms=setup_ms,
                    evaluation_ms=0.0,
                    teardown_ms=0.0,
                    total_ms=_monotonic_elapsed_ms(
                        started_at=artifact_started_at,
                        completed_at=time.monotonic(),
                    ),
                    outcome="setup_failed",
                    error_code=str(exc.error_code),
                )
                if self._activity is not None:
                    self._activity.mark_artifact_finished(batch_id, artifact.artifact_id)
                raise backfill_exc.cause from backfill_exc
            success_count, failure_count = _count_submission_outcomes(artifact_submissions)
            _log_artifact_execution_finished(
                batch_id=batch_id,
                artifact=artifact,
                artifact_index=artifact_index,
                artifact_count=artifact_count,
                planned_task_count=len(remaining_tasks),
                success_count=success_count,
                failure_count=failure_count,
                unresolved_count=unresolved_count,
                setup_ms=setup_ms,
                evaluation_ms=0.0,
                teardown_ms=0.0,
                total_ms=_monotonic_elapsed_ms(
                    started_at=artifact_started_at,
                    completed_at=time.monotonic(),
                ),
                outcome="setup_failed",
                error_code=str(exc.error_code),
            )
            if self._activity is not None:
                self._activity.mark_artifact_finished(batch_id, artifact.artifact_id)
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=artifact_submissions,
                validator_model_llm_baseline=completed_artifact_baseline(),
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                retry_round=retry_round,
            )

        artifact_result: ArtifactEvaluationOutcome | None = None
        primary_failure_raised = False
        evaluation_started_at: float | None = None
        try:
            orchestrator = self._make_orchestrator(deployment.client)
            if self._activity is not None:
                self._activity.mark_artifact_stage(batch_id, "artifact_evaluation_started")
            evaluation_started_at = time.monotonic()
            artifact_result = await self._evaluate_artifact_with_timeout_state(
                batch_id=batch_id,
                artifact=artifact,
                tasks=remaining_tasks,
                orchestrator=orchestrator,
                validator_model_llm_baseline=completed_artifact_baseline(),
                completed_artifact_baseline=completed_artifact_baseline,
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                earlier_submissions=earlier_submissions,
                task_session_limiter=task_session_limiter,
            )
            evaluation_ms = _monotonic_elapsed_ms(
                started_at=evaluation_started_at,
                completed_at=time.monotonic(),
            )
            artifact_submissions = tuple(artifact_result.submissions)
            unresolved_count = len(artifact_result.unresolved_tasks)
        except ValidatorBatchFailedError as exc:
            primary_failure_raised = True
            if stop_dequeuing is not None:
                stop_dequeuing()
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            if exc.completed_submissions is not None:
                artifact_submissions = exc.completed_submissions
            if exc.remaining_tasks is not None:
                unresolved_count = len(exc.remaining_tasks)
            outcome = "validator_batch_failure"
            error_code = str(exc.error_code)
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=artifact_submissions,
                validator_model_llm_baseline=completed_artifact_baseline(),
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                retry_round=retry_round,
                validator_batch_failure=ValidatorBatchFailedError(
                    error_code=exc.error_code,
                    message=str(exc),
                    failure_detail=exc.failure_detail,
                    completed_submissions=artifact_submissions,
                    remaining_tasks=exc.remaining_tasks,
                ),
            )
        except UnexpectedArtifactExecutionError as exc:
            primary_failure_raised = True
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            artifact_submissions = exc.completed_submissions
            unresolved_count = len(exc.remaining_tasks)
            outcome = "unexpected_failure"
            raise exc.cause from exc
        except Exception:
            primary_failure_raised = True
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            if backfill_primary_outcome is None:
                outcome = "unexpected_failure"
            else:
                outcome, error_code = backfill_primary_outcome
            raise
        finally:
            if self._activity is not None:
                self._activity.mark_artifact_stage(batch_id, "sandbox_teardown_started")
            teardown_started_at = time.monotonic()
            teardown_exc: Exception | None = None
            try:
                await _run_blocking_call(blocking_executor, self._sandboxes.stop, deployment)
            except Exception as exc:
                teardown_exc = exc
                teardown_ms = _monotonic_elapsed_ms(
                    started_at=teardown_started_at,
                    completed_at=time.monotonic(),
                )
                if not _has_primary_artifact_outcome(
                    outcome=outcome,
                    primary_failure_raised=primary_failure_raised,
                ):
                    outcome = "teardown_failed"
                    error_code = str(MinerTaskErrorCode.SANDBOX_FAILED)
                else:
                    logger.warning(
                        "artifact teardown failed after primary failure",
                        extra={
                            "data": {
                                "batch_id": str(batch_id),
                                "uid": artifact.uid,
                                "artifact_id": str(artifact.artifact_id),
                                "primary_outcome": outcome,
                                "primary_error_code": error_code,
                            }
                        },
                    )
            else:
                teardown_ms = _monotonic_elapsed_ms(
                    started_at=teardown_started_at,
                    completed_at=time.monotonic(),
                )
            success_count, failure_count = _count_submission_outcomes(artifact_submissions)
            _log_artifact_execution_finished(
                batch_id=batch_id,
                artifact=artifact,
                artifact_index=artifact_index,
                artifact_count=artifact_count,
                planned_task_count=len(remaining_tasks),
                success_count=success_count,
                failure_count=failure_count,
                unresolved_count=unresolved_count,
                setup_ms=setup_ms,
                evaluation_ms=evaluation_ms,
                teardown_ms=teardown_ms,
                total_ms=_monotonic_elapsed_ms(
                    started_at=artifact_started_at,
                    completed_at=time.monotonic(),
                ),
                outcome=outcome,
                error_code=error_code,
            )
            if teardown_exc is not None and not _has_primary_artifact_outcome(
                outcome=outcome,
                primary_failure_raised=primary_failure_raised,
            ):
                if self._activity is not None:
                    self._activity.mark_artifact_finished(batch_id, artifact.artifact_id)
                raise teardown_exc
            if self._activity is not None:
                self._activity.mark_artifact_finished(batch_id, artifact.artifact_id)

        if artifact_result is None:
            raise RuntimeError("artifact evaluation completed without result")

        return _CompletedArtifactResult(
            artifact_id=artifact.artifact_id,
            submissions=artifact_submissions,
            validator_model_llm_baseline=artifact_result.validator_model_llm_baseline,
            timeout_retry_state_by_pair={
                pair_key: TimeoutRetryState(prior_observations=observations)
                for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
            },
            unresolved_tasks=artifact_result.unresolved_tasks,
            retry_round=retry_round,
        )

    def _stop_artifact_dequeue(self, dispatch: _BatchArtifactDispatchState) -> None:
        dispatch.stop_dequeuing = True

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

    def _artifact_execution_failure(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
        exception_type: str | None,
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
        )

    def _remaining_tasks_for_failed_artifact(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        recorded_pairs: frozenset[tuple[UUID, UUID]],
    ) -> tuple[MinerTask, ...]:
        return tuple(
            task
            for task in tasks
            if (artifact.artifact_id, task.task_id) not in recorded_pairs
        )

    def _conclusive_batch_failure_from_artifact_error(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        failure: ArtifactExecutionFailedError,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
    ) -> ValidatorBatchFailedError:
        return ValidatorBatchFailedError(
            error_code=failure.error_code,
            message=str(failure),
            failure_detail=failure.failure_detail,
            completed_submissions=completed_submissions,
            remaining_tasks=self._remaining_tasks_for_failed_artifact(
                artifact=artifact,
                tasks=tasks,
                recorded_pairs=recorded_pairs,
            ),
        )

    def _build_final_batch_failure(
        self,
        *,
        batch_id: UUID,
        dispatch: _BatchArtifactDispatchState,
        completed_runs_after_sequence: int,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
    ) -> ValidatorBatchFailedError:
        canonical_validator_failure_index = (
            min(dispatch.validator_batch_failures_by_artifact_index)
            if dispatch.validator_batch_failures_by_artifact_index
            else None
        )
        failure = dispatch.published_batch_failure
        if failure is None:
            raise RuntimeError("published batch failure missing when finalizing batch failure")
        if canonical_validator_failure_index is not None:
            failure = dispatch.validator_batch_failures_by_artifact_index[canonical_validator_failure_index]
        completed_submissions = self._completed_submissions_for_final_failure(
            batch_id=batch_id,
            after_sequence=completed_runs_after_sequence,
            recorded_pairs=recorded_pairs,
            failure=failure,
        )
        return ValidatorBatchFailedError(
            error_code=failure.error_code,
            message=str(failure),
            failure_detail=failure.failure_detail,
            completed_submissions=completed_submissions,
            remaining_tasks=failure.remaining_tasks,
        )

    def _completed_submissions_for_final_failure(
        self,
        *,
        batch_id: UUID,
        after_sequence: int,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
        failure: ValidatorBatchFailedError,
    ) -> tuple[MinerTaskRunSubmission, ...]:
        completed_by_pair: dict[tuple[UUID, UUID], MinerTaskRunSubmission] = {}
        while True:
            page = self._progress.completed_run_page(
                batch_id,
                after_sequence=after_sequence,
                limit=_BATCH_FAILURE_PROGRESS_PAGE_SIZE,
            )
            for item in page["items"]:
                submission = item["submission"]
                pair = _submission_pair(submission)
                if pair not in recorded_pairs:
                    completed_by_pair[pair] = submission
            if not page["has_more"]:
                break
            next_after_sequence = page["next_after_sequence"]
            if next_after_sequence <= after_sequence:
                raise RuntimeError("completed run progress page did not advance")
            after_sequence = next_after_sequence

        for submission in failure.completed_submissions or ():
            pair = _submission_pair(submission)
            if pair not in recorded_pairs:
                completed_by_pair.setdefault(pair, submission)
        return tuple(completed_by_pair.values())

async def _run_blocking_call(
    executor: Executor,
    func: Callable[..., _T],
    /,
    *args: object,
) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args))


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
