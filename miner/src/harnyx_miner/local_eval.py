from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from harnyx_commons.domain.miner_task import AnswerCitation, MinerTask, ReferenceAnswer, Response
from harnyx_commons.domain.tool_usage import LlmModelUsageCost
from harnyx_commons.miner_task_ranking import (
    ArtifactRankingRow,
    CascadeConfig,
    RankingCascade,
    aggregate_ranking_rows,
    ordered_challengers,
)
from harnyx_commons.miner_task_scoring import EvaluationScoringConfig, EvaluationScoringService
from harnyx_commons.observability.logging import ExtrasFormatter
from harnyx_commons.sandbox.agent_staging import stage_agent_source
from harnyx_commons.sandbox.diagnostic_files import (
    ensure_private_diagnostic_dir,
    write_private_bytes,
    write_private_json,
)
from harnyx_commons.sandbox.manager import SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import build_sandbox_options, create_sandbox_manager
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.invocation_clients import build_tool_invocation_clients
from harnyx_miner.agent_source import (
    agent_sha256,
    load_agent_bytes,
    require_existing_agent_path,
    validate_agent_bytes,
)
from harnyx_miner.platform_monitoring import (
    PlatformMonitoringClient,
    RecordedBatchResultsSnapshot,
    SelectedBatchContext,
    platform_base_url_from_env,
)
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission, ScriptArtifactSpec
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker
from harnyx_validator.application.ports.progress import ProgressRecorder, ProviderFailureEvidence
from harnyx_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
    WeightSubmissionCadence,
    WeightSubmissionCadenceStatus,
)
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    ArtifactFailure,
    EvaluationRunner,
)
from harnyx_validator.infrastructure.http.local_tool_host import LocalToolHostHandle, start_local_tool_host
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.runtime.bootstrap import (
    _build_state,
    _build_tooling,
    _create_scoring_service,
    _resolve_scoring_judge_route,
)
from harnyx_validator.runtime.settings import Settings
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

_CLI_LOGGER_ROOTS = (
    "harnyx_commons",
    "harnyx_miner",
    "harnyx_validator",
)
_LOCAL_SESSION_TTL = timedelta(minutes=30)
_LOCAL_VALIDATOR_UID = 0
_LOCAL_SELECTION_VALIDATOR_ID = UUID(int=0)
_DEFAULT_OUTPUT_PREFIX = "local-eval-report"
_DEFAULT_LOCAL_ARTIFACT_TASK_PARALLELISM = SchedulerConfig.artifact_task_parallelism
_LOCAL_SANDBOX_HOST_PROBE_ADDRESS = "127.0.0.1"
_LOCAL_EVAL_FAILURE_ROOT = Path(tempfile.gettempdir()) / "harnyx-local-eval-failures"
_REPO_FRESHNESS_GIT_TIMEOUT_SECONDS = 5.0


def _emit_progress(message: str) -> None:
    print(f"[local-eval] {message}", file=sys.stderr, flush=True)


@dataclass(slots=True)
class _CliProgressReporter(ProgressRecorder):
    _artifact_labels: dict[UUID, str]
    _artifact_totals: dict[UUID, int]
    _artifact_completed: dict[UUID, int]
    _batch_by_session: dict[UUID, UUID]
    _provider_counters_by_batch: dict[UUID, dict[tuple[str, str], ProviderFailureEvidence]]
    _failed_provider_keys_by_session: dict[UUID, set[tuple[str, str]]]

    def __init__(self) -> None:
        self._artifact_labels = {}
        self._artifact_totals = {}
        self._artifact_completed = {}
        self._batch_by_session = {}
        self._provider_counters_by_batch = {}
        self._failed_provider_keys_by_session = {}

    def log(self, message: str) -> None:
        _emit_progress(message)

    def begin_artifact(
        self,
        *,
        label: str,
        artifact: ScriptArtifactSpec,
        task_count: int,
    ) -> None:
        self._artifact_labels[artifact.artifact_id] = label
        self._artifact_totals[artifact.artifact_id] = task_count
        self._artifact_completed[artifact.artifact_id] = 0
        self.log(
            f"starting {label} evaluation: artifact_id={artifact.artifact_id} tasks={task_count}"
        )

    def finish_artifact(
        self,
        *,
        label: str,
        artifact: ScriptArtifactSpec,
        submissions: Sequence[MinerTaskRunSubmission],
    ) -> None:
        total_score = round(sum(submission.score for submission in submissions), 6)
        error_count = sum(1 for submission in submissions if submission.run.details.error is not None)
        self.log(
            f"finished {label} evaluation: artifact_id={artifact.artifact_id} "
            f"tasks={len(submissions)} total_score={total_score} errors={error_count}"
        )

    def register(self, batch: MinerTaskBatchSpec) -> None:
        del batch
        return None

    def record(self, result: MinerTaskRunSubmission) -> None:
        artifact_id = result.run.artifact_id
        completed = self._artifact_completed.get(artifact_id, 0) + 1
        self._artifact_completed[artifact_id] = completed
        total = self._artifact_totals.get(artifact_id, completed)
        label = self._artifact_labels.get(artifact_id, "artifact")
        error = result.run.details.error
        if error is None:
            outcome = f"score={result.score:.3f}"
        else:
            outcome = f"error={error.code}"
        elapsed_ms = result.run.details.elapsed_ms
        elapsed_text = f" elapsed_ms={round(elapsed_ms, 1)}" if elapsed_ms is not None else ""
        self.log(
            f"{label} task {completed}/{total} complete: task_id={result.run.task_id} {outcome}{elapsed_text}"
        )

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        del batch, submissions, provider_evidence
        return None

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        del batch_id
        return frozenset()

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        self._batch_by_session[session_id] = batch_id

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        evidence = self._provider_evidence(session_id=session_id, provider=provider, model=model)
        if evidence is None:
            return
        evidence["total_calls"] += 1

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        evidence = self._provider_evidence(session_id=session_id, provider=provider, model=model)
        if evidence is None:
            return
        evidence["failed_calls"] += 1
        failure_reason = reason.strip()
        if failure_reason:
            evidence["failure_reason"] = failure_reason
        self._failed_provider_keys_by_session.setdefault(session_id, set()).add((provider, model))

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        batch_id = self._batch_by_session.get(session_id)
        if batch_id is None:
            return ()
        keys = self._failed_provider_keys_by_session.pop(session_id, None)
        if not keys:
            return ()
        counters = self._provider_counters_by_batch.get(batch_id, {})
        evidence: list[ProviderFailureEvidence] = []
        for key in sorted(keys):
            snapshot = counters.get(key)
            if snapshot is None:
                continue
            evidence.append(snapshot.copy())
        return tuple(evidence)

    def clear_task_session(self, session_id: UUID) -> None:
        self._batch_by_session.pop(session_id, None)
        self._failed_provider_keys_by_session.pop(session_id, None)

    def _provider_evidence(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> ProviderFailureEvidence | None:
        batch_id = self._batch_by_session.get(session_id)
        if batch_id is None:
            return None
        counters = self._provider_counters_by_batch.setdefault(batch_id, {})
        return counters.setdefault(
            (provider, model),
            ProviderFailureEvidence(
                provider=provider,
                model=model,
                total_calls=0,
                failed_calls=0,
            ),
        )


@dataclass(slots=True)
class _LocalProgressRecorder(ProgressRecorder):
    _display: _CliProgressReporter | None
    _storage: FileBackedRunProgress

    def register(self, batch: MinerTaskBatchSpec) -> None:
        self._storage.register(batch)
        if self._display is not None:
            self._display.register(batch)

    def record(self, result: MinerTaskRunSubmission) -> None:
        self._storage.record(result)
        if self._display is not None:
            self._display.record(result)

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        self._storage.restore_completed_runs(
            batch,
            submissions,
            provider_evidence=provider_evidence,
        )
        if self._display is not None:
            self._display.restore_completed_runs(
                batch,
                submissions,
                provider_evidence=provider_evidence,
            )

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        return self._storage.recorded_pairs(batch_id)

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        self._storage.register_task_session(batch_id=batch_id, session_id=session_id)

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        self._storage.record_provider_call(
            session_id=session_id,
            provider=provider,
            model=model,
        )

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        self._storage.record_provider_failure(
            session_id=session_id,
            provider=provider,
            model=model,
            reason=reason,
        )

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        return self._storage.consume_provider_failures(session_id)

    def clear_task_session(self, session_id: UUID) -> None:
        self._storage.clear_task_session(session_id)


class _LocalEvalFailureCategory(StrEnum):
    SANDBOX_STARTUP = "sandbox_startup"
    EVALUATED_APPLICATION = "evaluated_application"
    LOCAL_EVAL_RUNTIME = "local_eval_runtime"


@dataclass(slots=True)
class LocalEvaluationRuntime:
    settings: Settings
    tool_executor: ToolExecutor
    scoring_service: EvaluationScoringService
    scoring_config: EvaluationScoringConfig
    _runner: EvaluationRunner
    _state: Any
    _search_client: Any
    _llm_provider_registry: Any
    _tool_llm_provider: Any
    _scoring_llm_provider: Any | None
    _sandbox_manager: SandboxManager
    _tool_host: LocalToolHostHandle | None
    _tool_host_lock: asyncio.Lock
    _run_id: str
    _progress_reporter: _CliProgressReporter | None

    @classmethod
    def create(
        cls,
        *,
        run_progress_root: Path,
        progress_reporter: _CliProgressReporter | None = None,
    ) -> LocalEvaluationRuntime:
        settings = Settings.load()
        state = _build_state(settings, progress_storage_root=run_progress_root)
        invocation_clients = build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )
        scoring_route = _resolve_scoring_judge_route(settings)
        scoring_llm_provider = invocation_clients.llm_provider_registry.resolve(scoring_route.provider)
        scoring_service = _create_scoring_service(
            settings,
            scoring_llm_provider,
            scoring_route=scoring_route,
        )
        scoring_config = cast(EvaluationScoringConfig, scoring_service._config)
        return cls._from_components(
            settings=settings,
            state=state,
            search_client=invocation_clients.search_client,
            llm_provider_registry=invocation_clients.llm_provider_registry,
            tool_llm_provider=invocation_clients.tool_llm_provider,
            scoring_llm_provider=scoring_llm_provider,
            scoring_service=scoring_service,
            scoring_config=scoring_config,
            progress_reporter=progress_reporter,
        )

    @classmethod
    def create_invocation_only(
        cls,
        *,
        scoring_service: EvaluationScoringService,
        scoring_config: EvaluationScoringConfig,
        run_progress_root: Path,
        progress_reporter: _CliProgressReporter | None = None,
    ) -> LocalEvaluationRuntime:
        settings = Settings.load()
        state = _build_state(settings, progress_storage_root=run_progress_root)
        invocation_clients = build_tool_invocation_clients(
            llm_settings=settings.llm,
            bedrock_settings=settings.bedrock,
            vertex_settings=settings.vertex,
        )
        return cls._from_components(
            settings=settings,
            state=state,
            search_client=invocation_clients.search_client,
            llm_provider_registry=invocation_clients.llm_provider_registry,
            tool_llm_provider=invocation_clients.tool_llm_provider,
            scoring_llm_provider=None,
            scoring_service=scoring_service,
            scoring_config=scoring_config,
            progress_reporter=progress_reporter,
        )

    @classmethod
    def _from_components(
        cls,
        *,
        settings: Settings,
        state: Any,
        search_client: Any,
        llm_provider_registry: Any,
        tool_llm_provider: Any,
        scoring_llm_provider: Any | None,
        scoring_service: EvaluationScoringService,
        scoring_config: EvaluationScoringConfig,
        progress_reporter: _CliProgressReporter | None,
    ) -> LocalEvaluationRuntime:
        _, tool_executor = _build_tooling(
            state=state,
            resolved=settings,
            search_client=search_client,
            tool_llm_provider=tool_llm_provider,
        )
        progress = _LocalProgressRecorder(
            _display=progress_reporter,
            _storage=state.progress_tracker,
        )
        runner = EvaluationRunner(
            subtensor_client=_LocalSubtensorClient(),
            session_manager=state.session_manager,
            evaluation_records=state.evaluation_records,
            receipt_log=state.receipt_log,
            config=SchedulerConfig(
                token_secret_bytes=32,
                session_ttl=_LOCAL_SESSION_TTL,
                artifact_task_parallelism=_DEFAULT_LOCAL_ARTIFACT_TASK_PARALLELISM,
            ),
            clock=_utcnow,
            progress=progress,
        )
        return cls(
            settings=settings,
            tool_executor=tool_executor,
            scoring_service=scoring_service,
            scoring_config=scoring_config,
            _runner=runner,
            _state=state,
            _search_client=search_client,
            _llm_provider_registry=llm_provider_registry,
            _tool_llm_provider=tool_llm_provider,
            _scoring_llm_provider=scoring_llm_provider,
            _sandbox_manager=create_sandbox_manager(
                logger_name="harnyx_miner.local_eval.sandbox",
                host=_LOCAL_SANDBOX_HOST_PROBE_ADDRESS,
                published_port_bind_host=_LOCAL_SANDBOX_HOST_PROBE_ADDRESS,
            ),
            _tool_host=None,
            _tool_host_lock=asyncio.Lock(),
            _run_id=uuid4().hex,
            _progress_reporter=progress_reporter,
        )

    async def evaluate_artifact(
        self,
        *,
        artifact_label: str,
        agent_source: bytes,
        artifact: ScriptArtifactSpec,
        batch_id: UUID,
        tasks: Sequence[MinerTask],
        scoring_service: EvaluationScoringService | None = None,
    ) -> ArtifactEvaluationOutcome:
        if self._progress_reporter is not None:
            self._progress_reporter.begin_artifact(
                label=artifact_label,
                artifact=artifact,
                task_count=len(tasks),
            )
        tool_host = await self._ensure_tool_host()
        state_dir_handle = tempfile.TemporaryDirectory(prefix=f"harnyx-local-eval-{artifact.artifact_id}-")
        deployment = None
        failure_bundle_recorded = False
        failure_dir = self._failure_bundle_dir(artifact_label=artifact_label, artifact=artifact)
        try:
            state_dir = Path(state_dir_handle.name)
            staged_agent = stage_agent_source(
                state_dir=state_dir,
                container_root=DEFAULT_STATE_DIR,
                namespace="local_eval_agents",
                key=str(artifact.artifact_id),
                data=agent_source,
            )
            _ensure_private_failure_bundle_dir(failure_dir)
            options = build_sandbox_options(
                image=self.settings.sandbox.sandbox_image,
                network=None,
                pull_policy=self.settings.sandbox.sandbox_pull_policy,
                rpc_port=tool_host.port,
                container_name=f"harnyx-local-eval-{artifact.artifact_id.hex[:12]}-{uuid4().hex[:8]}",
                volumes=((str(state_dir), DEFAULT_STATE_DIR, "ro"),),
                extra_env={"AGENT_PATH": staged_agent.container_path},
                host_container_url=tool_host.host_container_url,
            )
            options = replace(options, failure_diagnostics_dir=str(failure_dir))
            try:
                deployment = await self._start_sandbox_deployment(options)
            except Exception as exc:
                self._record_failure_bundle(
                    failure_dir=failure_dir,
                    failure_category=_LocalEvalFailureCategory.SANDBOX_STARTUP,
                    artifact_label=artifact_label,
                    artifact=artifact,
                    agent_source=agent_source,
                    error=exc,
                )
                failure_bundle_recorded = True
                raise
            orchestrator = TaskRunOrchestrator(
                entrypoint_invoker=EntrypointInvoker(
                    session_registry=self._state.session_registry,
                    sandbox_client=deployment.client,
                    token_registry=self._state.token_registry,
                    receipt_log=self._state.receipt_log,
                ),
                receipt_log=self._state.receipt_log,
                scoring_service=scoring_service or self.scoring_service,
                session_registry=self._state.session_registry,
                clock=_utcnow,
            )
            evaluation_outcome = await self._runner.evaluate_artifact(
                batch_id=batch_id,
                artifact=artifact,
                tasks=tasks,
                orchestrator=orchestrator,
            )
            if evaluation_outcome.artifact_failure is not None:
                self._record_failure_bundle(
                    failure_dir=failure_dir,
                    failure_category=_LocalEvalFailureCategory.EVALUATED_APPLICATION,
                    artifact_label=artifact_label,
                    artifact=artifact,
                    agent_source=agent_source,
                    error=evaluation_outcome.artifact_failure,
                )
                failure_bundle_recorded = True
            if self._progress_reporter is not None and evaluation_outcome.artifact_failure is None:
                self._progress_reporter.finish_artifact(
                    label=artifact_label,
                    artifact=artifact,
                    submissions=evaluation_outcome.submissions,
                )
            return _compact_artifact_outcome_for_local_report(evaluation_outcome)
        except Exception as exc:
            if failure_bundle_recorded:
                raise
            self._record_failure_bundle(
                failure_dir=failure_dir,
                failure_category=_LocalEvalFailureCategory.LOCAL_EVAL_RUNTIME,
                artifact_label=artifact_label,
                artifact=artifact,
                agent_source=agent_source,
                error=exc,
            )
            raise
        finally:
            if deployment is not None:
                await asyncio.to_thread(self._sandbox_manager.stop, deployment)
            state_dir_handle.cleanup()

    def _failure_bundle_dir(
        self,
        *,
        artifact_label: str,
        artifact: ScriptArtifactSpec,
    ) -> Path:
        return _LOCAL_EVAL_FAILURE_ROOT / self._run_id / f"{artifact_label}-{artifact.artifact_id.hex[:12]}"

    def _record_failure_bundle(
        self,
        *,
        failure_dir: Path,
        failure_category: _LocalEvalFailureCategory,
        artifact_label: str,
        artifact: ScriptArtifactSpec,
        agent_source: bytes,
        error: BaseException | ArtifactFailure,
    ) -> None:
        if not self._try_write_local_failure_bundle(
            failure_dir=failure_dir,
            failure_category=failure_category,
            artifact_label=artifact_label,
            artifact=artifact,
            agent_source=agent_source,
            error=error,
        ):
            return
        if self._progress_reporter is not None:
            self._progress_reporter.log(f"failure category: {failure_category.value}")
            self._progress_reporter.log(f"failure bundle: {failure_dir}")

    def _try_write_local_failure_bundle(
        self,
        *,
        failure_dir: Path,
        failure_category: _LocalEvalFailureCategory,
        artifact_label: str,
        artifact: ScriptArtifactSpec,
        agent_source: bytes,
        error: BaseException | ArtifactFailure,
    ) -> bool:
        try:
            _write_local_failure_bundle(
                failure_dir=failure_dir,
                failure_category=failure_category,
                artifact_label=artifact_label,
                artifact=artifact,
                run_id=self._run_id,
                agent_source=agent_source,
                error=error,
            )
        except Exception as exc:
            if self._progress_reporter is not None:
                self._progress_reporter.log(
                    f"failure bundle write failed: path={failure_dir} error={exc}"
                )
            return False
        return True

    async def _start_sandbox_deployment(self, options: SandboxOptions) -> Any:
        start_task = asyncio.create_task(
            asyncio.to_thread(self._sandbox_manager.start, options),
            name=f"harnyx-local-eval-sandbox-start-{options.container_name}",
        )
        try:
            return await asyncio.shield(start_task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                deployment = await asyncio.shield(start_task)
                await asyncio.to_thread(self._sandbox_manager.stop, deployment)
            raise

    async def _ensure_tool_host(self) -> LocalToolHostHandle:
        if self._tool_host is not None:
            return self._tool_host
        async with self._tool_host_lock:
            if self._tool_host is None:
                if self._progress_reporter is not None:
                    self._progress_reporter.log("starting ephemeral local tool host")
                self._tool_host = await start_local_tool_host(
                    tool_executor=self.tool_executor,
                    tool_concurrency_limiter=self._state.tool_concurrency_limiter,
                )
                if self._progress_reporter is not None:
                    self._progress_reporter.log(
                        f"tool host ready: callback_url={self._tool_host.host_container_url}"
                    )
        if self._tool_host is None:  # pragma: no cover - defensive
            raise RuntimeError("local tool host did not initialize")
        return self._tool_host

    async def aclose(self) -> None:
        errors: list[Exception] = []
        if self._tool_host is not None:
            try:
                await self._tool_host.aclose()
            except Exception as exc:
                exc.add_note("local eval tool host close failed")
                errors.append(exc)
            finally:
                self._tool_host = None
        await _close_async_resources(
            self._search_client,
            self._llm_provider_registry,
            errors=errors,
            owner="local eval",
        )


def _write_local_failure_bundle(
    *,
    failure_dir: Path,
    failure_category: _LocalEvalFailureCategory,
    artifact_label: str,
    artifact: ScriptArtifactSpec,
    run_id: str,
    agent_source: bytes,
    error: BaseException | ArtifactFailure,
) -> None:
    _ensure_private_failure_bundle_dir(failure_dir)
    write_private_bytes(failure_dir / "agent.py", agent_source)
    write_private_json(
        failure_dir / "local-eval-context.json",
        _local_failure_context(
            failure_category=failure_category,
            artifact_label=artifact_label,
            artifact=artifact,
            run_id=run_id,
            error=error,
        ),
    )


def _ensure_private_failure_bundle_dir(failure_dir: Path) -> None:
    ensure_private_diagnostic_dir(_LOCAL_EVAL_FAILURE_ROOT)
    ensure_private_diagnostic_dir(failure_dir.parent)
    ensure_private_diagnostic_dir(failure_dir)


def _local_failure_context(
    *,
    failure_category: _LocalEvalFailureCategory,
    artifact_label: str,
    artifact: ScriptArtifactSpec,
    run_id: str,
    error: BaseException | ArtifactFailure,
) -> dict[str, object]:
    context = {
        "run_id": run_id,
        "artifact_label": artifact_label,
        "artifact_id": str(artifact.artifact_id),
        "artifact_uid": artifact.uid,
        "artifact_content_hash": artifact.content_hash,
        "artifact_size_bytes": artifact.size_bytes,
        "failure_category": failure_category.value,
    }
    context.update(_local_failure_error_context(error))
    return context


def _local_failure_error_context(error: BaseException | ArtifactFailure) -> dict[str, object]:
    if isinstance(error, ArtifactFailure):
        return {
            "error_type": "ArtifactFailure",
            "error_code": str(error.error_code),
            "error_message": error.message,
            "artifact_breaker_tripped": error.artifact_breaker_tripped,
        }
    return {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }


class _LocalSubtensorClient(SubtensorClientPort):
    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def fetch_metagraph(self) -> MetagraphSnapshot:
        return MetagraphSnapshot(uids=(), hotkeys=())

    def fetch_commitment(self, uid: int) -> CommitmentRecord | None:
        del uid
        return None

    def publish_commitment(self, data: str, *, blocks_until_reveal: int = 1) -> CommitmentRecord:
        del data, blocks_until_reveal
        raise RuntimeError("local evaluation does not publish commitments")

    def current_block(self) -> int:
        return 0

    def last_update_block(self, uid: int) -> int | None:
        del uid
        return None

    def weight_submission_cadence(self, netuid: int) -> WeightSubmissionCadence:
        del netuid
        return WeightSubmissionCadence(
            status=WeightSubmissionCadenceStatus.OPEN,
            validator_uid=_LOCAL_VALIDATOR_UID,
            commit_reveal_enabled=False,
            current_block=0,
            last_update_block=None,
            blocks_since_last_update=None,
            weights_rate_limit=None,
        )

    def validator_info(self) -> ValidatorNodeInfo:
        return ValidatorNodeInfo(uid=_LOCAL_VALIDATOR_UID, version_key=None)

    def submit_weights(self, weights: Mapping[int, float]) -> str:
        del weights
        raise RuntimeError("local evaluation does not submit weights")

    def fetch_weight(self, uid: int) -> float:
        del uid
        return 0.0

    def tempo(self, netuid: int) -> int:
        del netuid
        return 0

    def get_next_epoch_start_block(
        self,
        netuid: int,
        *,
        reference_block: int | None = None,
    ) -> int:
        del netuid, reference_block
        return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local miner artifact against a completed public miner-task batch.",
    )
    parser.add_argument("--agent-path", required=True, help="Path to the local miner agent file.")
    parser.add_argument(
        "--batch-id",
        help="Specific public batch to use. Defaults to the latest completed public miner-task batch.",
    )
    parser.add_argument(
        "--mode",
        choices=("vs-champion", "target-only"),
        default="vs-champion",
        help="Evaluation mode. Default: vs-champion.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd()),
        help="Directory for the JSON and Markdown reports. Default: current working directory.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


async def _amain(argv: Sequence[str] | None) -> None:
    args = _parse_args(argv)
    target_path = require_existing_agent_path(args.agent_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = _CliProgressReporter()
    _warn_if_repo_not_latest(progress=progress, repo_path=Path.cwd())

    monitoring = PlatformMonitoringClient.from_env()
    runtime: LocalEvaluationRuntime | None = None
    try:
        progress.log("resolving batch context")
        requested_batch_id = UUID(args.batch_id) if args.batch_id else None
        batch_context = monitoring.resolve_batch_context(requested_batch_id)
        _log_recorded_results_status(progress=progress, batch_context=batch_context)
        tasks = _load_batch_tasks(batch_context.detail)
        progress.log(
            f"selected batch: batch_id={batch_context.batch_id} source={batch_context.source} tasks={len(tasks)}"
        )
        target_bytes = load_agent_bytes(target_path)
        target_sha256 = agent_sha256(target_bytes)
        progress.log(
            "loaded target artifact: "
            f"path={target_path} size_bytes={len(target_bytes)} sha256={target_sha256}"
        )
        target_artifact = _build_target_artifact_spec(
            batch_context=batch_context,
            target_bytes=target_bytes,
        )
        champion_artifact: ScriptArtifactSpec | None = None
        champion_bytes: bytes | None = None
        champion_submissions: tuple[MinerTaskRunSubmission, ...] | None = None
        if args.mode == "vs-champion":
            champion_artifact_id = _require_champion_artifact_id(batch_context.detail)
            progress.log(f"fetching champion artifact: artifact_id={champion_artifact_id}")
            champion_script = monitoring.get_script(champion_artifact_id)
            champion_bytes = _decode_script_content(
                artifact_id=champion_artifact_id,
                script_payload=champion_script,
            )
            champion_artifact = ScriptArtifactSpec(
                uid=_require_int(champion_script.get("uid"), label="champion script uid"),
                artifact_id=champion_artifact_id,
                content_hash=_require_str(champion_script.get("content_hash"), label="champion script content hash"),
                size_bytes=_require_int(champion_script.get("size_bytes"), label="champion script size bytes"),
            )
            progress.log(
                f"champion artifact ready: artifact_id={champion_artifact.artifact_id} uid={champion_artifact.uid}"
            )
        progress.log("starting local evaluation runtime")
        run_progress_root = output_dir / ".harnyx-local-run-progress" / str(batch_context.batch_id)
        runtime = LocalEvaluationRuntime.create(
            run_progress_root=run_progress_root,
            progress_reporter=progress,
        )
        progress.log(
            f"running local evaluations: artifact_task_parallelism={_DEFAULT_LOCAL_ARTIFACT_TASK_PARALLELISM}"
        )
        if champion_artifact is not None and champion_bytes is not None:
            progress.log("running target and champion evaluations concurrently")
            target_outcome, champion_outcome = await asyncio.gather(
                runtime.evaluate_artifact(
                    artifact_label="target",
                    agent_source=target_bytes,
                    artifact=target_artifact,
                    batch_id=batch_context.batch_id,
                    tasks=tasks,
                ),
                runtime.evaluate_artifact(
                    artifact_label="champion",
                    agent_source=champion_bytes,
                    artifact=champion_artifact,
                    batch_id=batch_context.batch_id,
                    tasks=tasks,
                ),
            )
            target_submissions = _require_completed_local_eval_outcome(
                artifact_label="target",
                artifact=target_artifact,
                outcome=target_outcome,
            )
            champion_submissions = _require_completed_local_eval_outcome(
                artifact_label="champion",
                artifact=champion_artifact,
                outcome=champion_outcome,
            )
        else:
            target_outcome = await runtime.evaluate_artifact(
                artifact_label="target",
                agent_source=target_bytes,
                artifact=target_artifact,
                batch_id=batch_context.batch_id,
                tasks=tasks,
            )
            target_submissions = _require_completed_local_eval_outcome(
                artifact_label="target",
                artifact=target_artifact,
                outcome=target_outcome,
            )

        progress.log("writing reports")
        report = _build_report(
            batch_context=batch_context,
            target_path=target_path,
            target_bytes=target_bytes,
            target_artifact=target_artifact,
            target_submissions=target_submissions,
            champion_artifact=champion_artifact,
            champion_submissions=champion_submissions,
            mode=args.mode,
            output_dir=output_dir,
            platform_base_url=platform_base_url_from_env(),
            sandbox_image=runtime.settings.sandbox.sandbox_image,
            sandbox_pull_policy=runtime.settings.sandbox.sandbox_pull_policy,
            scoring_config=runtime.scoring_config,
            validator_version=VALIDATOR_RELEASE_VERSION,
        )
        json_path, markdown_path = _write_reports(
            report=report,
            output_dir=output_dir,
            batch_id=batch_context.batch_id,
            mode=args.mode,
        )
        progress.log(f"reports written: json={json_path} markdown={markdown_path}")
    finally:
        if runtime is not None:
            await runtime.aclose()
        monitoring.close()

    print(
        json.dumps(
            {
                "batch_id": str(batch_context.batch_id),
                "mode": args.mode,
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
            }
        )
    )


def _warn_if_repo_not_latest(*, progress: _CliProgressReporter, repo_path: Path) -> None:
    warning = _repo_freshness_warning(repo_path)
    if warning is None:
        return
    progress.log(warning)


def _repo_freshness_warning(repo_path: Path) -> str | None:
    local_head = _git_stdout(repo_path, "rev-parse", "HEAD")
    remote_head = _git_remote_head(repo_path)
    if local_head is None or remote_head is None:
        return None
    if local_head == remote_head:
        return None
    return (
        "repository is not at latest origin/HEAD; local eval will continue, "
        "but update before comparing final results"
    )


def _git_remote_head(repo_path: Path) -> str | None:
    output = _git_stdout(repo_path, "ls-remote", "origin", "HEAD")
    if output is None:
        return None
    parts = output.split(maxsplit=1)
    if not parts:
        return None
    return parts[0]


def _git_stdout(repo_path: Path, *args: str) -> str | None:
    command = ["git", "-C", str(repo_path), *args]
    try:
        completed = subprocess.run(  # noqa: S603 - internal read-only Git query
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_REPO_FRESHNESS_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    if not output:
        return None
    return output


def _require_completed_local_eval_outcome(
    *,
    artifact_label: str,
    artifact: ScriptArtifactSpec,
    outcome: ArtifactEvaluationOutcome,
) -> tuple[MinerTaskRunSubmission, ...]:
    if outcome.artifact_failure is None:
        return outcome.submissions
    failure = outcome.artifact_failure
    raise RuntimeError(
        f"{artifact_label} local evaluation failed for artifact {artifact.artifact_id}: "
        f"{failure.error_code} ({failure.message})"
    )


def _compact_artifact_outcome_for_local_report(
    outcome: ArtifactEvaluationOutcome,
) -> ArtifactEvaluationOutcome:
    return replace(
        outcome,
        submissions=tuple(
            _compact_submission_for_local_report(submission) for submission in outcome.submissions
        ),
    )


def _compact_submission_for_local_report(
    submission: MinerTaskRunSubmission,
) -> MinerTaskRunSubmission:
    return submission.model_copy(update={"execution_log": ()})


def _load_batch_tasks(detail: Mapping[str, object]) -> tuple[MinerTask, ...]:
    batch = _require_mapping(detail.get("batch"), label="batch detail batch")
    raw_tasks = _require_sequence(batch.get("tasks"), label="batch tasks")
    parsed: list[MinerTask] = []
    for raw_task in raw_tasks:
        if isinstance(raw_task, MinerTask):
            parsed.append(raw_task)
            continue
        parsed.append(MinerTask.model_validate_json(json.dumps(raw_task)))
    return tuple(parsed)


def _build_target_artifact_spec(
    *,
    batch_context: SelectedBatchContext,
    target_bytes: bytes,
) -> ScriptArtifactSpec:
    detail_batch = _require_mapping(batch_context.detail.get("batch"), label="batch detail batch")
    raw_artifacts = _require_sequence(detail_batch.get("artifacts"), label="batch artifacts")
    recorded_uids = [
        _require_int(_require_mapping(raw_artifact, label="batch artifact").get("uid"), label="batch artifact uid")
        for raw_artifact in raw_artifacts
    ]
    content_hash = agent_sha256(target_bytes)
    return ScriptArtifactSpec(
        uid=max(recorded_uids, default=0) + 1,
        artifact_id=uuid5(NAMESPACE_URL, f"harnyx-local-eval:{batch_context.batch_id}:{content_hash}"),
        content_hash=content_hash,
        size_bytes=len(target_bytes),
    )


def _require_champion_artifact_id(detail: Mapping[str, object]) -> UUID:
    summary = _require_mapping(detail.get("summary"), label="batch summary")
    champion_artifact_id = summary.get("champion_artifact_id")
    if champion_artifact_id in (None, ""):
        raise RuntimeError(
            "selected batch does not expose a champion artifact; rerun with --mode target-only"
        )
    return UUID(str(champion_artifact_id))


def _decode_script_content(
    *,
    artifact_id: UUID,
    script_payload: Mapping[str, object],
) -> bytes:
    content_b64 = script_payload.get("content_b64")
    if not isinstance(content_b64, str) or not content_b64:
        raise RuntimeError(f"artifact {artifact_id} is missing content_b64")
    try:
        content = base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"artifact {artifact_id} content_b64 is invalid") from exc
    try:
        return validate_agent_bytes(content, label=f"artifact {artifact_id}")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _build_report(
    *,
    batch_context: SelectedBatchContext,
    target_path: Path,
    target_bytes: bytes,
    target_artifact: ScriptArtifactSpec,
    target_submissions: Sequence[MinerTaskRunSubmission],
    champion_artifact: ScriptArtifactSpec | None,
    champion_submissions: Sequence[MinerTaskRunSubmission] | None,
    mode: str,
    output_dir: Path,
    platform_base_url: str,
    sandbox_image: str,
    sandbox_pull_policy: str,
    scoring_config: EvaluationScoringConfig,
    validator_version: str,
) -> dict[str, object]:
    tasks = _load_batch_tasks(batch_context.detail)
    target_by_task = {submission.run.task_id: submission for submission in target_submissions}
    champion_by_task = (
        {submission.run.task_id: submission for submission in champion_submissions}
        if champion_submissions is not None
        else {}
    )
    recorded_results = batch_context.recorded_results
    recorded_rows = _group_recorded_rows(recorded_results.rows or ())
    leaderboard_entries = [
        _artifact_summary_entry(
            label="target",
            source="local",
            artifact=target_artifact,
            submissions=target_submissions,
        )
    ]
    if champion_artifact is not None and champion_submissions is not None:
        leaderboard_entries.append(
            _artifact_summary_entry(
                label="champion",
                source="local",
                artifact=champion_artifact,
                submissions=champion_submissions,
            )
        )
    leaderboard = _sort_leaderboard(leaderboard_entries)
    return {
        "batch_metadata": {
            "batch_id": str(batch_context.batch_id),
            "selection_source": batch_context.source,
            "summary": batch_context.detail.get("summary"),
            "batch": batch_context.detail.get("batch"),
        },
        "mode": mode,
        "identifiers": {
            "batch_id": str(batch_context.batch_id),
            "target_artifact_id": str(target_artifact.artifact_id),
            "target_uid": target_artifact.uid,
            "champion_artifact_id": str(champion_artifact.artifact_id) if champion_artifact is not None else None,
            "champion_uid": champion_artifact.uid if champion_artifact is not None else None,
        },
        "evaluation_config": {
            "platform_base_url": platform_base_url,
            "agent_path": str(target_path),
            "output_dir": str(output_dir),
            "validator_version": validator_version,
            "artifact_task_parallelism": _DEFAULT_LOCAL_ARTIFACT_TASK_PARALLELISM,
            "artifact_evaluation_parallelism": 2 if champion_artifact is not None else 1,
            "execution_boundary": "docker-sandbox",
            "tool_host_mode": "ephemeral-local-http",
            "sandbox_image": sandbox_image,
            "sandbox_pull_policy": sandbox_pull_policy,
            "session_ttl_seconds": int(_LOCAL_SESSION_TTL.total_seconds()),
            "local_validator_uid": _LOCAL_VALIDATOR_UID,
        },
        "scoring_context": {
            "provider": scoring_config.provider,
            "model": scoring_config.model,
            "temperature": scoring_config.temperature,
            "max_output_tokens": scoring_config.max_output_tokens,
            "reasoning_effort": scoring_config.reasoning_effort,
            "timeout_seconds": scoring_config.timeout_seconds,
            "scoring_version": scoring_config.scoring_version,
            "weights": {
                "comparison_score": 1.0,
            },
            "methods": {
                "comparison_score": "pairwise judge vs reference answer with swapped order",
            },
        },
        "artifacts": {
            "target": {
                "artifact_id": str(target_artifact.artifact_id),
                "uid": target_artifact.uid,
                "content_hash": target_artifact.content_hash,
                "size_bytes": target_artifact.size_bytes,
                "path": str(target_path),
                "sha256": agent_sha256(target_bytes),
            },
            "champion": (
                {
                    "artifact_id": str(champion_artifact.artifact_id),
                    "uid": champion_artifact.uid,
                    "content_hash": champion_artifact.content_hash,
                    "size_bytes": champion_artifact.size_bytes,
                }
                if champion_artifact is not None
                else None
            ),
        },
        "local_result_summary": {
            "leaderboard": leaderboard,
            "local_champion_selection": _local_champion_selection_summary(
                target_artifact=target_artifact,
                target_submissions=target_submissions,
                champion_artifact=champion_artifact,
                champion_submissions=champion_submissions,
            ),
            "head_to_head": _head_to_head_summary(
                target_submissions=target_submissions,
                champion_submissions=champion_submissions or (),
            )
            if champion_submissions is not None
            else None,
        },
        "recorded_platform_context": {
            "batch_detail": batch_context.detail,
            "results": recorded_results.rows,
            "results_status": _serialize_recorded_results_status(recorded_results),
            "results_scope": _serialize_recorded_results_scope(recorded_results),
        },
        "tasks": [
            _task_report(
                task=task,
                target_submission=target_by_task.get(task.task_id),
                champion_submission=champion_by_task.get(task.task_id),
                recorded_rows=(
                    recorded_rows.get(task.task_id, ())
                    if recorded_results.rows is not None
                    else None
                ),
            )
            for task in tasks
        ],
    }


def _task_report(
    *,
    task: MinerTask,
    target_submission: MinerTaskRunSubmission | None,
    champion_submission: MinerTaskRunSubmission | None,
    recorded_rows: Sequence[dict[str, object]] | None,
) -> dict[str, object]:
    return {
        "task_id": str(task.task_id),
        "question": task.query.text,
        "reference_answer": _serialize_answer(task.reference_answer),
        "reference_context": {
            "reference_answer": _serialize_answer(task.reference_answer),
            "budget_usd": task.budget_usd,
        },
        "target": _submission_detail(target_submission),
        "opponent": _submission_detail(champion_submission) if champion_submission is not None else None,
        "recorded_platform_rows": list(recorded_rows) if recorded_rows is not None else None,
    }


def _serialize_recorded_results_status(batch_results: RecordedBatchResultsSnapshot) -> dict[str, object]:
    if batch_results.error is None:
        return {
            "state": "available",
            "error": None,
        }
    error: dict[str, object] = {"detail": batch_results.error.detail}
    if batch_results.error.path is not None:
        error["path"] = batch_results.error.path
    if batch_results.error.status_code is not None:
        error["status_code"] = batch_results.error.status_code
    return {
        "state": "unavailable",
        "error": error,
    }


def _serialize_recorded_results_scope(batch_results: RecordedBatchResultsSnapshot) -> dict[str, object] | None:
    if batch_results.scope is None:
        return None
    return {
        "kind": batch_results.scope.kind,
        "batch_id": str(batch_results.scope.batch_id),
        "artifact_id": str(batch_results.scope.artifact_id),
    }


def _log_recorded_results_status(
    *,
    progress: _CliProgressReporter,
    batch_context: SelectedBatchContext,
) -> None:
    error = batch_context.recorded_results.error
    if error is None:
        return
    path = error.path if error.path is not None else "local-eval-recorded-context"
    status_code = str(error.status_code) if error.status_code is not None else "n/a"
    progress.log(
        "recorded platform results unavailable: "
        f"path={path} status_code={status_code} detail={error.detail}"
    )


def _submission_detail(submission: MinerTaskRunSubmission | None) -> dict[str, object] | None:
    if submission is None:
        return None
    run = submission.run
    return {
        "artifact_id": str(run.artifact_id),
        "uid": run.uid,
        "answer": _serialize_answer(run.response) if run.response is not None else None,
        "score": submission.score,
        "score_details": run.details.model_dump(mode="json"),
        "cost_and_usage": {
            "cost_totals": _cost_totals_from_submission(submission),
            "token_usage": submission.usage.model_dump(mode="json"),
            "provider_model_usage": _serialize_provider_model_usage(run.details.total_tool_usage.llm.providers),
        },
        "elapsed_ms": run.details.elapsed_ms,
        "attempt_count": submission.session.active_attempt,
        "session_status": submission.session.status.value,
        "error": run.details.error.model_dump(mode="json") if run.details.error is not None else None,
    }


def _serialize_answer(answer: object) -> dict[str, object]:
    model_dump = getattr(answer, "model_dump", None)
    if not callable(model_dump):
        raise RuntimeError("answer payload must support model_dump()")
    return model_dump(mode="json", exclude_none=True)


def _artifact_summary_entry(
    *,
    label: str,
    source: str,
    artifact: ScriptArtifactSpec,
    submissions: Sequence[MinerTaskRunSubmission],
) -> dict[str, object]:
    total_score = round(sum(submission.score for submission in submissions), 6)
    cost_totals = _aggregate_cost_totals(submissions)
    return {
        "label": label,
        "source": source,
        "artifact_id": str(artifact.artifact_id),
        "uid": artifact.uid,
        "task_count": len(submissions),
        "total_score": total_score,
        "avg_score": round(total_score / len(submissions), 6) if submissions else 0.0,
        "error_count": sum(1 for submission in submissions if submission.run.details.error is not None),
        "cost_totals": cost_totals,
    }


def _sort_leaderboard(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        entries,
        key=lambda entry: (
            -_require_float(entry.get("total_score"), label="leaderboard total score"),
            _require_float(
                _require_mapping(entry.get("cost_totals"), label="leaderboard cost totals").get("total_cost_usd"),
                label="leaderboard total cost usd",
            ),
            str(entry["label"]),
        ),
    )


def _local_champion_selection_summary(
    *,
    target_artifact: ScriptArtifactSpec,
    target_submissions: Sequence[MinerTaskRunSubmission],
    champion_artifact: ScriptArtifactSpec | None,
    champion_submissions: Sequence[MinerTaskRunSubmission] | None,
) -> dict[str, object]:
    candidate_artifacts = [target_artifact]
    candidate_submissions = list(target_submissions)
    incumbent_artifact_id: UUID | None = None
    artifact_labels = {target_artifact.artifact_id: "target"}
    artifact_uids = {target_artifact.artifact_id: target_artifact.uid}
    if champion_artifact is not None and champion_submissions is not None:
        candidate_artifacts.insert(0, champion_artifact)
        candidate_submissions.extend(champion_submissions)
        incumbent_artifact_id = champion_artifact.artifact_id
        artifact_labels[champion_artifact.artifact_id] = "champion"
        artifact_uids[champion_artifact.artifact_id] = champion_artifact.uid

    aggregates = aggregate_ranking_rows(
        tuple(_ranking_row_from_submission(submission) for submission in candidate_submissions)
    )
    candidate_artifact_ids = [artifact.artifact_id for artifact in candidate_artifacts]
    selected_artifact_id = RankingCascade(CascadeConfig()).decide(
        initial=incumbent_artifact_id,
        challengers_ordered=ordered_challengers(
            initial=incumbent_artifact_id,
            candidate_artifact_ids=candidate_artifact_ids,
        ),
        aggregates=aggregates,
    )
    return {
        "logic": "platform-ranking-cascade",
        "cohort": {
            "type": "local-eval",
            "validator_count": 1,
            "artifact_count": len(candidate_artifact_ids),
        },
        "initial_incumbent_artifact_id": str(incumbent_artifact_id) if incumbent_artifact_id is not None else None,
        "candidate_artifact_ids": [str(artifact_id) for artifact_id in candidate_artifact_ids],
        "selected_artifact_id": str(selected_artifact_id) if selected_artifact_id is not None else None,
        "selected_label": artifact_labels.get(selected_artifact_id),
        "aggregates_by_label": {
            label: {
                "artifact_id": str(artifact_id),
                "uid": artifact_uids[artifact_id],
                "total_score": float(aggregates.totals.get(artifact_id, 0.0)),
                "median_cost_usd": float(aggregates.costs.get(artifact_id, 0.0)),
                "median_elapsed_ms": aggregates.median_elapsed_ms.get(artifact_id),
            }
            for artifact_id, label in artifact_labels.items()
        },
    }


def _head_to_head_summary(
    *,
    target_submissions: Sequence[MinerTaskRunSubmission],
    champion_submissions: Sequence[MinerTaskRunSubmission],
) -> dict[str, object]:
    champion_by_task = {submission.run.task_id: submission for submission in champion_submissions}
    wins = 0
    losses = 0
    ties = 0
    for target_submission in target_submissions:
        opponent = champion_by_task.get(target_submission.run.task_id)
        if opponent is None:
            continue
        if target_submission.score > opponent.score:
            wins += 1
        elif target_submission.score < opponent.score:
            losses += 1
        else:
            ties += 1
    raw_target_total = sum(submission.score for submission in target_submissions)
    raw_champion_total = sum(submission.score for submission in champion_submissions)
    if raw_target_total > raw_champion_total:
        winner_by_total_score = "target"
    elif raw_target_total < raw_champion_total:
        winner_by_total_score = "champion"
    else:
        winner_by_total_score = "tie"
    target_total = round(raw_target_total, 6)
    champion_total = round(raw_champion_total, 6)
    return {
        "winner_by_total_score": winner_by_total_score,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "target_total_score": target_total,
        "champion_total_score": champion_total,
    }


def _ranking_row_from_submission(submission: MinerTaskRunSubmission) -> ArtifactRankingRow:
    details = submission.run.details
    return ArtifactRankingRow(
        validator_id=_LOCAL_SELECTION_VALIDATOR_ID,
        artifact_id=submission.run.artifact_id,
        task_id=submission.run.task_id,
        score=float(submission.score),
        total_cost_usd=float(details.total_tool_usage.llm_cost + details.total_tool_usage.search_tool_cost),
        elapsed_ms=details.elapsed_ms,
    )


def _group_recorded_rows(results: Sequence[dict[str, object]]) -> dict[UUID, tuple[dict[str, object], ...]]:
    grouped: dict[UUID, list[dict[str, object]]] = {}
    for row in results:
        task_id = UUID(str(row["task_id"]))
        grouped.setdefault(task_id, []).append(row)
    return {task_id: tuple(rows) for task_id, rows in grouped.items()}


def _aggregate_cost_totals(submissions: Sequence[MinerTaskRunSubmission]) -> dict[str, object]:
    total_llm_cost = 0.0
    total_search_cost = 0.0
    total_llm_tokens = 0
    total_llm_calls = 0
    total_search_calls = 0
    for submission in submissions:
        details = submission.run.details.total_tool_usage
        total_llm_cost += details.llm_cost
        total_search_cost += details.search_tool_cost
        total_llm_tokens += submission.usage.total_tokens
        total_llm_calls += submission.usage.call_count
        total_search_calls += details.search_tool.call_count
    return {
        "llm_cost_usd": round(total_llm_cost, 6),
        "search_tool_cost_usd": round(total_search_cost, 6),
        "total_cost_usd": round(total_llm_cost + total_search_cost, 6),
        "llm_total_tokens": total_llm_tokens,
        "llm_call_count": total_llm_calls,
        "search_tool_call_count": total_search_calls,
    }


def _cost_totals_from_submission(submission: MinerTaskRunSubmission) -> dict[str, object]:
    return _aggregate_cost_totals((submission,))


def _write_reports(
    *,
    report: Mapping[str, object],
    output_dir: Path,
    batch_id: UUID,
    mode: str,
) -> tuple[Path, Path]:
    json_path = output_dir / f"{_DEFAULT_OUTPUT_PREFIX}-{batch_id}-{mode}.json"
    markdown_path = output_dir / f"{_DEFAULT_OUTPUT_PREFIX}-{batch_id}-{mode}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def _serialize_provider_model_usage(
    providers: Mapping[str, Mapping[str, LlmModelUsageCost]],
) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for provider_name, models in providers.items():
        serialized[provider_name] = {
            model_name: _serialize_model_usage_cost(model_usage)
            for model_name, model_usage in models.items()
        }
    return serialized


def _serialize_model_usage_cost(model_usage: LlmModelUsageCost) -> dict[str, object]:
    usage = model_usage.usage
    return {
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "call_count": usage.call_count,
        },
        "cost": model_usage.cost,
    }


def _render_markdown_report(report: Mapping[str, object]) -> str:
    batch_metadata = _require_mapping(report.get("batch_metadata"), label="batch metadata")
    identifiers = _require_mapping(report.get("identifiers"), label="identifiers")
    evaluation_config = _require_mapping(report.get("evaluation_config"), label="evaluation config")
    scoring_context = _require_mapping(report.get("scoring_context"), label="scoring context")
    local_result_summary = _require_mapping(report.get("local_result_summary"), label="local result summary")
    leaderboard = _require_sequence(local_result_summary.get("leaderboard"), label="leaderboard")
    local_champion_selection = _require_mapping(
        local_result_summary.get("local_champion_selection"),
        label="local champion selection",
    )
    head_to_head = local_result_summary.get("head_to_head")
    tasks = _require_sequence(report.get("tasks"), label="task reports")

    lines = [
        "# Local Evaluation Report",
        "",
        "## Summary",
        f"- Batch: `{identifiers['batch_id']}` ({batch_metadata['selection_source']})",
        f"- Mode: `{report['mode']}`",
        f"- Target artifact: `{identifiers['target_artifact_id']}`",
    ]
    if identifiers.get("champion_artifact_id") is not None:
        lines.append(f"- Champion artifact: `{identifiers['champion_artifact_id']}`")
    lines.extend(
        [
            f"- Agent path: `{evaluation_config['agent_path']}`",
            "",
            "## Scoring Context",
            f"- Provider/model: `{scoring_context['provider']}` / `{scoring_context['model']}`",
            f"- Scoring version: `{scoring_context['scoring_version']}`",
            "- Weights: comparison `1.0`",
            "",
            "## Local Leaderboard",
            "",
            "| Rank | Label | Total score | Avg score | Errors | Total cost USD |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, raw_entry in enumerate(leaderboard, start=1):
        entry = _require_mapping(raw_entry, label="leaderboard entry")
        costs = _require_mapping(entry.get("cost_totals"), label="leaderboard entry costs")
        lines.append(
            f"| {index} | {entry['label']} | {entry['total_score']} | {entry['avg_score']} | "
            f"{entry['error_count']} | {costs['total_cost_usd']} |"
        )
    lines.extend(
        [
            "",
            "## Local Champion Selection",
            f"- Logic: `{local_champion_selection['logic']}`",
            f"- Cohort: `{_require_mapping(local_champion_selection['cohort'], label='selection cohort')['type']}`",
            f"- Selected artifact label: `{local_champion_selection['selected_label']}`",
        ]
    )
    if head_to_head is not None:
        head_to_head_map = _require_mapping(head_to_head, label="head to head summary")
        lines.extend(
            [
                "",
                "## Head-to-Head",
                f"- Winner by total score: `{head_to_head_map['winner_by_total_score']}`",
                "- Wins / losses / ties: "
                f"{head_to_head_map['wins']} / {head_to_head_map['losses']} / {head_to_head_map['ties']}",
                f"- Target total score: {head_to_head_map['target_total_score']}",
                f"- Champion total score: {head_to_head_map['champion_total_score']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Recorded Platform Context",
            _render_recorded_platform_context_markdown(report),
            "",
            "## Per-Task Details",
        ]
    )
    for raw_task in tasks:
        task = _require_mapping(raw_task, label="task report")
        lines.extend(
            [
                "",
                f"### Task `{task['task_id']}`",
                f"- Question: {task['question']}",
            ]
        )
        lines.extend(
            _render_answer_markdown(
                "Reference",
                task.get("reference_answer"),
                model_type=ReferenceAnswer,
            )
        )
        target = _require_mapping(task.get("target"), label="task target")
        lines.extend(_render_submission_markdown("Target", target))
        opponent = task.get("opponent")
        if opponent is not None:
            lines.extend(_render_submission_markdown("Opponent", _require_mapping(opponent, label="task opponent")))
        lines.append(_render_recorded_rows_markdown(task))
    lines.append("")
    return "\n".join(lines)


def _render_recorded_platform_context_markdown(report: Mapping[str, object]) -> str:
    recorded_context = _require_mapping(
        report.get("recorded_platform_context"),
        label="recorded platform context",
    )
    results_status = _require_mapping(
        recorded_context.get("results_status"),
        label="recorded results status",
    )
    status = _require_str(results_status.get("state"), label="recorded results status state")
    if status == "available":
        return (
            "- The JSON report contains the full batch detail and champion-artifact recorded monitoring rows "
            "for automated analysis."
        )
    error = _require_mapping(results_status.get("error"), label="recorded results error")
    return (
        "- Recorded monitoring rows were unavailable for this run: "
        f"{_require_str(error.get('detail'), label='recorded results error detail')}"
    )


def _render_recorded_rows_markdown(task: Mapping[str, object]) -> str:
    raw_recorded_rows = task.get("recorded_platform_rows")
    if raw_recorded_rows is None:
        return "- Recorded platform rows: unavailable"
    recorded_rows = _require_sequence(raw_recorded_rows, label="recorded rows")
    return f"- Recorded platform rows: {len(recorded_rows)}"


def _render_submission_markdown(label: str, submission: Mapping[str, object]) -> list[str]:
    cost_and_usage = _require_mapping(submission.get("cost_and_usage"), label=f"{label} cost and usage")
    cost_totals = _require_mapping(cost_and_usage.get("cost_totals"), label=f"{label} cost totals")
    lines = [
        f"- {label} score: {submission['score']}",
        f"- {label} attempts: {submission['attempt_count']}",
        f"- {label} elapsed ms: {submission['elapsed_ms']}",
        f"- {label} total cost USD: {cost_totals['total_cost_usd']}",
    ]
    lines.extend(_render_answer_markdown(label, submission.get("answer"), model_type=Response))
    return lines


def _render_answer_markdown(
    label: str,
    raw_answer: object,
    *,
    model_type: type[ReferenceAnswer] | type[Response],
) -> list[str]:
    if raw_answer is None:
        return [f"- {label} answer: (none)"]
    answer = model_type.model_validate(raw_answer, strict=True)
    lines = [f"- {label} answer: {answer.text}"]
    lines.extend(_render_citations_markdown(label, answer.citations))
    return lines


def _render_citations_markdown(label: str, citations: tuple[AnswerCitation, ...] | None) -> list[str]:
    if citations is None:
        return []
    if not citations:
        return []
    lines = [f"- {label} citations:"]
    for citation in citations:
        lines.append(f"  - {_citation_markdown_line(citation)}")
    return lines


def _citation_markdown_line(citation: AnswerCitation) -> str:
    parts = [citation.title or citation.url]
    if citation.title:
        parts.append(citation.url)
    if citation.note:
        parts.append(citation.note)
    return " - ".join(parts)


def _answer_text(raw_answer: object, *, label: str) -> str:
    answer = _require_mapping(raw_answer, label=label)
    text = answer.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"{label} text must be a string")
    return text


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(f"{label} must be an integer") from exc
    raise RuntimeError(f"{label} must be an integer")


def _require_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise RuntimeError(f"{label} must be a number") from exc
    raise RuntimeError(f"{label} must be a number")


def _require_str(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _unique_async_resources(*resources: object) -> tuple[Any, ...]:
    unique: list[Any] = []
    seen: set[int] = set()
    for resource in resources:
        if resource is None or not hasattr(resource, "aclose"):
            continue
        resource_id = id(resource)
        if resource_id in seen:
            continue
        seen.add(resource_id)
        unique.append(resource)
    return tuple(unique)


async def _close_async_resources(
    *resources: object,
    errors: list[Exception],
    owner: str,
) -> None:
    for resource in _unique_async_resources(*resources):
        try:
            await resource.aclose()
        except Exception as exc:
            exc.add_note(f"{owner} async resource close failed")
            errors.append(exc)
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    raise ExceptionGroup(f"{owner} cleanup failed", errors)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _configure_cli_logging() -> None:
    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "WARNING").upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ExtrasFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.handlers = [handler]
    _reset_cli_package_loggers(root.level)


def _reset_cli_package_loggers(root_level: int) -> None:
    for root_name in _CLI_LOGGER_ROOTS:
        logger = logging.getLogger(root_name)
        logger.setLevel(root_level)
        logger.disabled = False
        logger.propagate = True

    for name, entry in logging.Logger.manager.loggerDict.items():
        if not isinstance(entry, logging.Logger):
            continue
        for root_name in _CLI_LOGGER_ROOTS:
            if name.startswith(f"{root_name}."):
                entry.setLevel(logging.NOTSET)
                entry.disabled = False
                entry.propagate = True
                break


def main(argv: Sequence[str] | None = None) -> None:
    try:
        _configure_cli_logging()
        asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


__all__ = [
    "LocalEvaluationRuntime",
    "main",
]
