"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TypeVar, cast
from uuid import UUID, uuid4

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    MinerTask,
    MinerTaskErrorCode,
)
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.assigned_work import AssignedArtifactWork
from harnyx_validator.application.dto.evaluation import (
    DIAGNOSTIC_ID_MAX_LENGTH,
    DIAGNOSTIC_LOG_TAIL_MAX_LENGTH,
    DIAGNOSTIC_STATE_ERROR_MAX_LENGTH,
    DIAGNOSTIC_TEXT_MAX_LENGTH,
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
    SandboxFailureDiagnostics,
    ScriptArtifactSpec,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.platform_tool_proxy import PlatformToolProxyScopeRegistry
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_runner import (
    LOCAL_RETRY_ATTEMPTS,
    ArtifactExecutionFailedError,
    EvaluationRunner,
)
from harnyx_validator.application.status import BatchActivityTracker
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]
_T = TypeVar("_T")

logger = logging.getLogger("harnyx_validator.scheduler")
_NON_SENSITIVE_DIAGNOSTIC_ENV_KEYS = frozenset(
    {
        "AGENT_PATH",
        "EVALUATION_RUN_ID",
        "MINER_UID",
        "SANDBOX_HOST",
        "SANDBOX_PORT",
    }
)


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta
    artifact_parallelism: int = 4
    artifact_task_parallelism: int = 20


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
        platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry | None = None,
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
            platform_tool_proxy_scopes=platform_tool_proxy_scopes,
        )

    async def run_assigned_task(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        attempt_number: int,
        max_attempts: int,
        assignment_token: str,
    ) -> PlatformOwnedTaskResult:
        self._mark_artifact_activity_started_best_effort(batch_id=batch_id, artifact_id=artifact.artifact_id)
        deployment: SandboxDeployment | None = None
        attempt_started_at = self._clock()
        try:
            deployment = await self._start_artifact_with_retry(
                batch_id=batch_id,
                artifact=artifact,
                tasks=(task,),
                blocking_executor=self._blocking_executor,
            )
            orchestrator = self._make_orchestrator(deployment.client)
            return await self._runner.evaluate_assigned_task(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                assignment_token=assignment_token,
                orchestrator=orchestrator,
            )
        except ArtifactExecutionFailedError as exc:
            if exc.error_code is MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED:
                results = await self._runner.record_assigned_task_setup_failures(
                    batch_id=batch_id,
                    artifact=artifact,
                    assignments=(
                        MinerTaskWorkAssignment(
                            batch_id=batch_id,
                            artifact=artifact,
                            task=task,
                            attempt_number=attempt_number,
                            max_attempts=max_attempts,
                            assignment_token=assignment_token,
                        ),
                    ),
                    error_code=exc.error_code,
                    error_message=str(exc),
                )
                return results[0]
            return self._platform_result_from_artifact_setup_failure(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                started_at=attempt_started_at,
                failure=exc,
            )
        finally:
            if deployment is not None:
                await self._stop_deployment_best_effort(
                    deployment,
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                )
            self._mark_artifact_activity_finished_best_effort(
                batch_id=batch_id,
                artifact_id=artifact.artifact_id,
            )

    async def run_assigned_artifact_queue(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        initial_assignments: Sequence[MinerTaskWorkAssignment],
        assigned_work: AssignedArtifactWork,
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult],
    ) -> None:
        assignments = tuple(initial_assignments)
        if not assignments:
            raise ValueError("assigned artifact queue requires at least one assignment")
        for assignment in assignments:
            _require_assignment_for_artifact(
                assignment,
                batch_id=batch_id,
                artifact_id=artifact.artifact_id,
            )

        self._mark_artifact_activity_started_best_effort(batch_id=batch_id, artifact_id=artifact.artifact_id)
        deployment: SandboxDeployment | None = None
        attempt_started_at = self._clock()
        try:
            deployment = await self._start_artifact_with_retry(
                batch_id=batch_id,
                artifact=artifact,
                tasks=tuple(assignment.task for assignment in assignments),
                blocking_executor=self._blocking_executor,
            )
            orchestrator = self._make_orchestrator(deployment.client)
            assigned_work.mark_dispatch_ready()
            await self._runner.evaluate_assigned_task_queue(
                batch_id=batch_id,
                artifact=artifact,
                initial_assignments=assignments,
                assigned_work=assigned_work,
                close_requested=close_requested,
                result_queue=result_queue,
                orchestrator=orchestrator,
            )
        except ArtifactExecutionFailedError as exc:
            affected_assignments = (*assignments, *assigned_work.drain_for_setup_failure())
            if exc.error_code is MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED:
                results = await self._runner.record_assigned_task_setup_failures(
                    batch_id=batch_id,
                    artifact=artifact,
                    assignments=affected_assignments,
                    error_code=exc.error_code,
                    error_message=str(exc),
                )
                for result in results:
                    await result_queue.put(result)
                return

            representative = affected_assignments[0]
            await result_queue.put(
                self._platform_result_from_artifact_setup_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=representative.task,
                    attempt_number=representative.attempt_number,
                    max_attempts=representative.max_attempts,
                    started_at=attempt_started_at,
                    failure=exc,
                )
            )
        finally:
            if deployment is not None:
                await self._stop_deployment_best_effort(
                    deployment,
                    batch_id=batch_id,
                    artifact_id=artifact.artifact_id,
                )
            self._mark_artifact_activity_finished_best_effort(
                batch_id=batch_id,
                artifact_id=artifact.artifact_id,
            )

    async def _start_artifact_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        blocking_executor: Executor,
    ) -> SandboxDeployment:
        last_error_message = ""
        last_options: object | None = None
        for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
            options = await self._build_sandbox_options_or_raise_artifact_failure(
                batch_id=batch_id,
                artifact=artifact,
                tasks=tasks,
                blocking_executor=blocking_executor,
            )
            last_options = options
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
            sandbox_diagnostics=_sandbox_failure_diagnostics_from_options(last_options),
        )

    async def _build_sandbox_options_or_raise_artifact_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        blocking_executor: Executor,
    ) -> SandboxOptions:
        try:
            return await _run_blocking_call(blocking_executor, self._sandbox_options, artifact)
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

    def _platform_result_from_artifact_setup_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        attempt_number: int,
        max_attempts: int,
        started_at: datetime,
        failure: ArtifactExecutionFailedError,
    ) -> PlatformOwnedTaskResult:
        finished_at = self._clock()
        attempt = MinerTaskAttemptAuditRecord(
            validator_session_id=uuid4(),
            batch_id=batch_id,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            uid=artifact.uid,
            miner_hotkey_ss58=artifact.miner_hotkey_ss58 or "unknown-miner-hotkey",
            started_at=started_at,
            finished_at=finished_at,
            status=MinerTaskAttemptStatus.FAILED,
            error_code=str(failure.error_code),
            error_summary_code=str(failure.error_code),
            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
            terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
            max_attempts=max_attempts,
            execution_log=(),
            delivery_failure_detail=failure.failure_detail,
        )
        result = PlatformOwnedTaskResult(
            batch_id=batch_id,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            attempt_number=attempt_number,
            result=None,
            terminal_attempt=attempt,
        )
        self._record_terminated_attempt_best_effort(result)
        return result

    async def _stop_deployment_best_effort(
        self,
        deployment: SandboxDeployment,
        *,
        batch_id: UUID,
        artifact_id: UUID,
    ) -> None:
        try:
            await _run_blocking_call(self._blocking_executor, self._sandboxes.stop, deployment)
        except Exception:
            logger.exception(
                "assigned artifact sandbox teardown failed after result construction",
                extra={"batch_id": str(batch_id), "artifact_id": str(artifact_id)},
            )

    def _mark_artifact_activity_started_best_effort(self, *, batch_id: UUID, artifact_id: UUID) -> None:
        if self._activity is None:
            return
        try:
            self._activity.mark_batch_started(batch_id)
            self._activity.mark_artifact_started(batch_id, artifact_id)
        except Exception:
            logger.exception(
                "assigned artifact activity start marker failed",
                extra={"batch_id": str(batch_id), "artifact_id": str(artifact_id)},
            )

    def _mark_artifact_activity_finished_best_effort(self, *, batch_id: UUID, artifact_id: UUID) -> None:
        if self._activity is None:
            return
        try:
            self._activity.mark_artifact_finished(batch_id, artifact_id)
            self._activity.mark_batch_finished(batch_id)
        except Exception:
            logger.exception(
                "assigned artifact activity finalizer failed after result construction",
                extra={"batch_id": str(batch_id), "artifact_id": str(artifact_id)},
            )

    def _record_terminated_attempt_best_effort(self, result: PlatformOwnedTaskResult) -> None:
        try:
            self._progress.record_terminated_attempt(result.terminal_attempt)
        except Exception:
            logger.exception(
                "assigned artifact setup-failure progress write failed after result construction",
                extra={
                    "batch_id": str(result.batch_id),
                    "artifact_id": str(result.artifact_id),
                    "task_id": str(result.task_id),
                    "attempt_number": result.attempt_number,
                },
            )

    def _artifact_execution_failure(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
        exception_type: str | None,
        sandbox_diagnostics: SandboxFailureDiagnostics | None = None,
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
                sandbox_diagnostics=sandbox_diagnostics,
            ),
            completed_submissions=(),
            remaining_tasks=tuple(tasks),
        )

def _require_assignment_for_artifact(
    assignment: MinerTaskWorkAssignment,
    *,
    batch_id: UUID,
    artifact_id: UUID,
) -> None:
    if assignment.batch_id != batch_id:
        raise ValueError("assigned task batch_id does not match artifact queue")
    if assignment.artifact.artifact_id != artifact_id:
        raise ValueError("assigned task artifact_id does not match artifact queue")


def _sandbox_failure_diagnostics_from_options(options: object | None) -> SandboxFailureDiagnostics | None:
    if not isinstance(options, SandboxOptions) or options.failure_diagnostics_dir is None:
        return None
    diagnostics_dir = Path(options.failure_diagnostics_dir)
    try:
        sandbox_options = _read_json_object(diagnostics_dir / "sandbox-options.json")
        docker_inspect = _docker_inspect_container(_read_json(diagnostics_dir / "docker-inspect.json"))
        docker_state = _object_field(docker_inspect, "State")
        docker_pull_result = _read_json_object(diagnostics_dir / "docker-pull-result.json")
        docker_run_result = _read_json_object(diagnostics_dir / "docker-run-result.json")
        return SandboxFailureDiagnostics(
            image=_bounded_identifier(options.image or _string_field(sandbox_options, "image")),
            pull_policy=_bounded_identifier(options.pull_policy or _string_field(sandbox_options, "pull_policy")),
            container_name=_bounded_identifier(
                options.container_name or _string_field(sandbox_options, "container_name")
            ),
            container_id=_bounded_identifier(_string_field(docker_inspect, "Id")),
            status=_bounded_identifier(_string_field(docker_state, "Status")),
            exit_code=_int_field(docker_state, "ExitCode"),
            oom_killed=_bool_field(docker_state, "OOMKilled"),
            state_error=_bounded_text(
                _redact_sensitive_text(_string_field(docker_state, "Error"), options),
                max_length=DIAGNOSTIC_STATE_ERROR_MAX_LENGTH,
            ),
            error_text=_bounded_text(
                _redact_sensitive_text(_read_text(diagnostics_dir / "error.txt"), options),
                max_length=DIAGNOSTIC_TEXT_MAX_LENGTH,
            ),
            docker_logs_tail=_bounded_log_tail(
                _redact_sensitive_text(_read_text(diagnostics_dir / "docker-logs.txt"), options)
            ),
            pull_returncode=_int_field(docker_pull_result, "returncode"),
            pull_stdout_tail=_bounded_text(
                _redact_sensitive_text(_string_field(docker_pull_result, "stdout"), options),
                max_length=DIAGNOSTIC_TEXT_MAX_LENGTH,
            ),
            pull_stderr_tail=_bounded_text(
                _redact_sensitive_text(_string_field(docker_pull_result, "stderr"), options),
                max_length=DIAGNOSTIC_TEXT_MAX_LENGTH,
            ),
            run_returncode=_int_field(docker_run_result, "returncode"),
            run_stdout_tail=_bounded_text(
                _redact_sensitive_text(_string_field(docker_run_result, "stdout"), options),
                max_length=DIAGNOSTIC_TEXT_MAX_LENGTH,
            ),
            run_stderr_tail=_bounded_text(
                _redact_sensitive_text(_string_field(docker_run_result, "stderr"), options),
                max_length=DIAGNOSTIC_TEXT_MAX_LENGTH,
            ),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path must not mask the failure
        logger.warning(
            "sandbox failure diagnostics could not be summarized",
            extra={"diagnostics_dir": str(diagnostics_dir)},
            exc_info=exc,
        )
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: Path) -> object | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _read_json_object(path: Path) -> dict[str, object] | None:
    value = _read_json(path)
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def _docker_inspect_container(value: object | None) -> dict[str, object] | None:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return cast("dict[str, object]", value[0])
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def _object_field(mapping: dict[str, object] | None, field: str) -> dict[str, object] | None:
    if mapping is None:
        return None
    value = mapping.get(field)
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def _string_field(mapping: dict[str, object] | None, field: str) -> str | None:
    if mapping is None:
        return None
    value = mapping.get(field)
    if isinstance(value, str):
        return value
    return None


def _int_field(mapping: dict[str, object] | None, field: str) -> int | None:
    if mapping is None:
        return None
    value = mapping.get(field)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _bool_field(mapping: dict[str, object] | None, field: str) -> bool | None:
    if mapping is None:
        return None
    value = mapping.get(field)
    if isinstance(value, bool):
        return value
    return None


def _bounded_identifier(value: str | None) -> str | None:
    return _bounded_text(value, max_length=DIAGNOSTIC_ID_MAX_LENGTH)


def _redact_sensitive_text(value: str | None, options: SandboxOptions) -> str | None:
    if value is None:
        return None
    redacted = value
    for key, env_value in options.env.items():
        if key in _NON_SENSITIVE_DIAGNOSTIC_ENV_KEYS or not env_value:
            continue
        redacted = redacted.replace(env_value, "<redacted>")
    return redacted


def _bounded_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    if len(value) <= max_length:
        return value
    return value[-max_length:]


def _bounded_log_tail(value: str | None) -> str | None:
    if value is None:
        return None
    lines = value.splitlines()
    if len(lines) > 200:
        value = "\n".join(lines[-200:])
    return _bounded_text(value, max_length=DIAGNOSTIC_LOG_TAIL_MAX_LENGTH)


async def _run_blocking_call(
    executor: Executor,
    func: Callable[..., _T],
    /,
    *args: object,
) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args))


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
