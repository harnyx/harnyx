"""Helpers for preparing miner-task batch execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID

from caster_commons.application.session_manager import SessionManager
from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.docker import DockerSandboxManager
from caster_commons.sandbox.options import SandboxOptions
from caster_commons.sandbox.state import DEFAULT_STATE_DIR, resolve_state_mount_source
from caster_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort
from caster_validator.application.ports.progress import ProgressRecorder
from caster_validator.application.ports.subtensor import SubtensorClientPort
from caster_validator.application.providers.claims import StaticClaimsProvider
from caster_validator.application.scheduler import EvaluationScheduler, SchedulerConfig


@dataclass(frozen=True, slots=True)
class EvaluationBatchConfig:
    """Configuration for evaluation batch processing."""

    state_dir: str = DEFAULT_STATE_DIR
    token_secret_bytes: int = 16

SandboxOptionsFactory = Callable[[], SandboxOptions]
OrchestratorFactory = Callable[[SandboxClient], EvaluationOrchestrator]


class AgentArtifact(Protocol):
    @property
    def container_path(self) -> str: ...


AgentResolver = Callable[[UUID, MinerTaskBatchSpec, Path, str], Mapping[UUID, AgentArtifact]]


@dataclass(frozen=True, slots=True)
class RunContext:
    batch_id: UUID
    entrypoint: str
    config: EvaluationBatchConfig
    claims_provider: StaticClaimsProvider
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
            entrypoint=batch.entrypoint,
            config=self._config,
            claims_provider=StaticClaimsProvider(batch.claims),
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
        agent_artifacts, volumes, selected_candidates = self._resolve_agents(run_ctx, batch)
        scheduler = self._build_scheduler(run_ctx, agent_artifacts, volumes)
        return selected_candidates, scheduler

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
        selected_candidates = batch.candidates
        return agent_artifacts, volumes, selected_candidates

    def _build_scheduler(
        self,
        run_ctx: RunContext,
        agent_artifacts: Mapping[UUID, AgentArtifact],
        volumes: tuple[tuple[str, str, str | None], ...],
    ) -> EvaluationScheduler:
        options_factory = self._build_sandbox_options_factory(run_ctx, agent_artifacts, volumes)
        return EvaluationScheduler(
            claims_provider=run_ctx.claims_provider,
            subtensor_client=self._subtensor,
            sandbox_manager=self._sandbox_manager,
            session_manager=self._session_manager,
            evaluation_records=self._evaluation_records,
            orchestrator_factory=self._orchestrator_factory,
            sandbox_options_factory=options_factory,
            clock=lambda: datetime.now(UTC),
            config=SchedulerConfig(
                entrypoint=run_ctx.entrypoint,
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
        def sandbox_options_factory(candidate: ScriptArtifactSpec) -> SandboxOptions:
            container_name = (
                f"caster-sandbox-{candidate.uid}-{candidate.artifact_id.hex[:8]}-{run_ctx.batch_id.hex[:8]}"
            )
            env = dict(run_ctx.base_env)
            env["CASTER_MINER_UID"] = str(candidate.uid)
            env["CASTER_EVALUATION_RUN_ID"] = str(run_ctx.batch_id)
            artifact = agent_artifacts.get(candidate.artifact_id)
            if artifact is not None:
                env["CASTER_AGENT_PATH"] = artifact.container_path
            elif "CASTER_AGENT_PATH" not in env:
                raise RuntimeError(f"agent path missing for candidate {candidate.uid}/{candidate.artifact_id}")
            return replace(
                run_ctx.base_options,
                container_name=container_name,
                env=env,
                volumes=volumes,
            )

        return sandbox_options_factory

__all__ = ["RunContext", "BatchExecutionPlanner"]
