"""Helpers for preparing miner-task batch execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR, resolve_state_mount_source
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.scheduler import EvaluationScheduler, SchedulerConfig


@dataclass(frozen=True, slots=True)
class EvaluationBatchConfig:
    """Configuration for evaluation batch processing."""

    state_dir: str = DEFAULT_STATE_DIR
    token_secret_bytes: int = 16


SandboxOptionsFactory = Callable[[], SandboxOptions]
OrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]


class AgentArtifact(Protocol):
    @property
    def container_path(self) -> str: ...


AgentResolver = Callable[[UUID, MinerTaskBatchSpec, Path, str], Mapping[UUID, AgentArtifact]]


@dataclass(frozen=True, slots=True)
class RunContext:
    batch_id: UUID
    tasks: tuple[MinerTask, ...]
    config: EvaluationBatchConfig
    base_options: SandboxOptions
    base_env: dict[str, str]
    base_volumes: tuple[tuple[str, str, str | None], ...]
    state_dir: Path


class BatchExecutionPlanner:
    """Builds run context, resolves agents, and constructs the scheduler."""

    def __init__(
        self,
        *,
        subtensor_client: SubtensorClientPort,
        sandbox_manager: DockerSandboxManager,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        receipt_log: ReceiptLogPort,
        orchestrator_factory: OrchestratorFactory,
        sandbox_options_factory: SandboxOptionsFactory,
        agent_resolver: AgentResolver,
        progress: ProgressRecorder | None,
        config: EvaluationBatchConfig,
    ) -> None:
        self._subtensor = subtensor_client
        self._sandbox_manager = sandbox_manager
        self._session_manager = session_manager
        self._evaluation_records = evaluation_records
        self._receipt_log = receipt_log
        self._orchestrator_factory = orchestrator_factory
        self._sandbox_options_factory = sandbox_options_factory
        self._agent_resolver = agent_resolver
        self._progress = progress
        self._config = config

    def build_run_context(self, batch: MinerTaskBatchSpec) -> RunContext:
        base_options = self._sandbox_options_factory()
        state_dir = Path(self._config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        return RunContext(
            batch_id=batch.batch_id,
            tasks=batch.tasks,
            config=self._config,
            base_options=base_options,
            base_env=dict(base_options.env),
            base_volumes=tuple(base_options.volumes),
            state_dir=state_dir,
        )

    def prepare_execution(
        self,
        run_ctx: RunContext,
        batch: MinerTaskBatchSpec,
    ) -> tuple[tuple[ScriptArtifactSpec, ...], EvaluationScheduler]:
        agent_artifacts, volumes, selected_artifacts = self._resolve_agents(run_ctx, batch)
        scheduler = self._build_scheduler(run_ctx, agent_artifacts, volumes)
        return selected_artifacts, scheduler

    def _resolve_agents(
        self,
        run_ctx: RunContext,
        batch: MinerTaskBatchSpec,
    ) -> tuple[
        Mapping[UUID, AgentArtifact],
        tuple[tuple[str, str, str | None], ...],
        tuple[ScriptArtifactSpec, ...],
    ]:
        agent_artifacts = self._agent_resolver(
            run_ctx.batch_id,
            batch,
            run_ctx.state_dir,
            run_ctx.config.state_dir,
        )
        volumes = run_ctx.base_volumes + ((resolve_state_mount_source(), run_ctx.config.state_dir, "ro"),)
        return agent_artifacts, volumes, batch.artifacts

    def _build_scheduler(
        self,
        run_ctx: RunContext,
        agent_artifacts: Mapping[UUID, AgentArtifact],
        volumes: tuple[tuple[str, str, str | None], ...],
    ) -> EvaluationScheduler:
        options_factory = self._build_sandbox_options_factory(run_ctx, agent_artifacts, volumes)
        return EvaluationScheduler(
            tasks=run_ctx.tasks,
            subtensor_client=self._subtensor,
            sandbox_manager=self._sandbox_manager,
            session_manager=self._session_manager,
            evaluation_records=self._evaluation_records,
            receipt_log=self._receipt_log,
            orchestrator_factory=self._orchestrator_factory,
            sandbox_options_factory=options_factory,
            clock=lambda: datetime.now(UTC),
            config=SchedulerConfig(
                token_secret_bytes=run_ctx.config.token_secret_bytes,
                session_ttl=timedelta(minutes=5),
            ),
            progress=self._progress,
        )

    def _build_sandbox_options_factory(
        self,
        run_ctx: RunContext,
        agent_artifacts: Mapping[UUID, AgentArtifact],
        volumes: tuple[tuple[str, str, str | None], ...],
    ) -> Callable[[ScriptArtifactSpec], SandboxOptions]:
        def sandbox_options_factory(artifact: ScriptArtifactSpec) -> SandboxOptions:
            container_name = (
                f"harnyx-sandbox-{artifact.uid}-{artifact.artifact_id.hex[:8]}-{run_ctx.batch_id.hex[:8]}"
            )
            env = dict(run_ctx.base_env)
            env["MINER_UID"] = str(artifact.uid)
            env["EVALUATION_RUN_ID"] = str(run_ctx.batch_id)
            resolved_artifact = agent_artifacts.get(artifact.artifact_id)
            if resolved_artifact is not None:
                env["AGENT_PATH"] = resolved_artifact.container_path
            elif "AGENT_PATH" not in env:
                raise RuntimeError(f"agent path missing for artifact {artifact.uid}/{artifact.artifact_id}")
            return replace(
                run_ctx.base_options,
                container_name=container_name,
                env=env,
                volumes=volumes,
            )

        return sandbox_options_factory


__all__ = ["RunContext", "BatchExecutionPlanner", "EvaluationBatchConfig"]
