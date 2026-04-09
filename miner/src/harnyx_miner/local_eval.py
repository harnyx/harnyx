from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from harnyx_commons.sandbox.agent_staging import stage_agent_source
from harnyx_commons.sandbox.manager import SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import build_sandbox_options, create_sandbox_manager
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_miner.agent_source import (
    agent_sha256,
    load_agent_bytes,
    require_existing_agent_path,
    validate_agent_bytes,
)
from harnyx_miner.platform_monitoring import PlatformMonitoringClient, SelectedBatchContext, platform_base_url_from_env
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission, ScriptArtifactSpec
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker
from harnyx_validator.application.ports.progress import ProgressRecorder, ProviderFailureEvidence
from harnyx_validator.application.ports.subtensor import (
    CommitmentRecord,
    MetagraphSnapshot,
    SubtensorClientPort,
    ValidatorNodeInfo,
)
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    ArtifactEvaluationOutcome,
    EvaluationRunner,
)
from harnyx_validator.application.services.evaluation_scoring import EvaluationScoringConfig, EvaluationScoringService
from harnyx_validator.infrastructure.http.local_tool_host import LocalToolHostHandle, start_local_tool_host
from harnyx_validator.runtime.bootstrap import (
    _build_local_eval_tooling_clients,
    _build_state,
    _build_tooling,
    _create_scoring_embedding_client,
    _create_scoring_service,
)
from harnyx_validator.runtime.settings import Settings
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

_LOCAL_SESSION_TTL = timedelta(minutes=30)
_LOCAL_VALIDATOR_UID = 0
_LOCAL_SELECTION_VALIDATOR_ID = UUID(int=0)
_DEFAULT_OUTPUT_PREFIX = "local-eval-report"
_COMPARISON_WEIGHT = 0.5
_SIMILARITY_WEIGHT = 0.5
_DEFAULT_PUBLISHED_SANDBOX_NETWORK = "bridge"
_DEFAULT_LOCAL_ARTIFACT_TASK_PARALLELISM = SchedulerConfig.artifact_task_parallelism
_LOCAL_SANDBOX_HOST_PROBE_ADDRESS = "127.0.0.1"


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
    ) -> None:
        evidence = self._provider_evidence(session_id=session_id, provider=provider, model=model)
        if evidence is None:
            return
        evidence["failed_calls"] += 1
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
class LocalEvaluationRuntime:
    settings: Settings
    tool_executor: ToolExecutor
    scoring_service: EvaluationScoringService
    scoring_config: EvaluationScoringConfig
    _runner: EvaluationRunner
    _state: Any
    _search_client: Any
    _tool_llm_provider: Any
    _scoring_llm_provider: Any
    _scoring_embedding_client: Any
    _sandbox_manager: SandboxManager
    _tool_host: LocalToolHostHandle | None
    _tool_host_lock: asyncio.Lock
    _progress_reporter: _CliProgressReporter | None

    @classmethod
    def create(cls, *, progress_reporter: _CliProgressReporter | None = None) -> LocalEvaluationRuntime:
        settings = Settings.load()
        state = _build_state()
        search_client, tool_llm_provider, scoring_llm_provider = _build_local_eval_tooling_clients(
            settings
        )
        _, tool_executor = _build_tooling(
            state=state,
            resolved=settings,
            search_client=search_client,
            tool_llm_provider=tool_llm_provider,
        )
        scoring_embedding_client = _create_scoring_embedding_client(settings)
        scoring_service = _create_scoring_service(
            settings,
            scoring_llm_provider,
            embedding_client=scoring_embedding_client,
        )
        scoring_config = cast(EvaluationScoringConfig, scoring_service._config)
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
            progress=progress_reporter,
        )
        return cls(
            settings=settings,
            tool_executor=tool_executor,
            scoring_service=scoring_service,
            scoring_config=scoring_config,
            _runner=runner,
            _state=state,
            _search_client=search_client,
            _tool_llm_provider=tool_llm_provider,
            _scoring_llm_provider=scoring_llm_provider,
            _scoring_embedding_client=scoring_embedding_client,
            _sandbox_manager=create_sandbox_manager(
                logger_name="harnyx_miner.local_eval.sandbox",
                host=_LOCAL_SANDBOX_HOST_PROBE_ADDRESS,
            ),
            _tool_host=None,
            _tool_host_lock=asyncio.Lock(),
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
        try:
            state_dir = Path(state_dir_handle.name)
            staged_agent = stage_agent_source(
                state_dir=state_dir,
                container_root=DEFAULT_STATE_DIR,
                namespace="local_eval_agents",
                key=str(artifact.artifact_id),
                data=agent_source,
            )
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
            deployment = await self._start_sandbox_deployment(options)
            orchestrator = TaskRunOrchestrator(
                entrypoint_invoker=EntrypointInvoker(
                    session_registry=self._state.session_registry,
                    sandbox_client=deployment.client,
                    token_registry=self._state.token_registry,
                    receipt_log=self._state.receipt_log,
                ),
                receipt_log=self._state.receipt_log,
                scoring_service=self.scoring_service,
                session_registry=self._state.session_registry,
                clock=_utcnow,
            )
            evaluation_outcome = await self._runner.evaluate_artifact(
                batch_id=batch_id,
                artifact=artifact,
                tasks=tasks,
                orchestrator=orchestrator,
            )
            if self._progress_reporter is not None and evaluation_outcome.artifact_failure is None:
                self._progress_reporter.finish_artifact(
                    label=artifact_label,
                    artifact=artifact,
                    submissions=evaluation_outcome.submissions,
                )
            return evaluation_outcome
        finally:
            if deployment is not None:
                await asyncio.to_thread(self._sandbox_manager.stop, deployment)
            state_dir_handle.cleanup()

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
                    token_semaphore=self._state.token_semaphore,
                )
                if self._progress_reporter is not None:
                    self._progress_reporter.log(
                        f"tool host ready: callback_url={self._tool_host.host_container_url}"
                    )
        if self._tool_host is None:  # pragma: no cover - defensive
            raise RuntimeError("local tool host did not initialize")
        return self._tool_host

    async def aclose(self) -> None:
        if self._tool_host is not None:
            await self._tool_host.aclose()
            self._tool_host = None
        for resource in _unique_async_resources(
            self._search_client,
            self._tool_llm_provider,
            self._scoring_llm_provider,
            self._scoring_embedding_client,
        ):
            await resource.aclose()


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

    monitoring = PlatformMonitoringClient.from_env()
    runtime: LocalEvaluationRuntime | None = None
    try:
        progress.log("resolving batch context")
        requested_batch_id = UUID(args.batch_id) if args.batch_id else None
        batch_context = monitoring.resolve_batch_context(requested_batch_id)
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
        runtime = LocalEvaluationRuntime.create(progress_reporter=progress)
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
    recorded_rows = _group_recorded_rows(batch_context.results)
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
                "comparison_score": _COMPARISON_WEIGHT,
                "similarity_score": _SIMILARITY_WEIGHT,
            },
            "methods": {
                "comparison_score": "pairwise judge vs reference answer with swapped order",
                "similarity_score": "embedding cosine similarity vs reference answer",
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
            "results": batch_context.results,
        },
        "tasks": [
            _task_report(
                task=task,
                target_submission=target_by_task.get(task.task_id),
                champion_submission=champion_by_task.get(task.task_id),
                recorded_rows=recorded_rows.get(task.task_id, ()),
            )
            for task in tasks
        ],
    }


def _task_report(
    *,
    task: MinerTask,
    target_submission: MinerTaskRunSubmission | None,
    champion_submission: MinerTaskRunSubmission | None,
    recorded_rows: Sequence[dict[str, object]],
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
        "recorded_platform_rows": list(recorded_rows),
    }


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
            f"- Weights: comparison `{_COMPARISON_WEIGHT}`, similarity `{_SIMILARITY_WEIGHT}`",
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
            "- The JSON report contains the full batch detail and recorded monitoring rows for automated analysis.",
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
        recorded_rows = _require_sequence(task.get("recorded_platform_rows"), label="recorded rows")
        lines.append(
            f"- Recorded platform rows: {len(recorded_rows)}"
        )
    lines.append("")
    return "\n".join(lines)


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


def _utcnow() -> datetime:
    return datetime.now(UTC)


def main(argv: Sequence[str] | None = None) -> None:
    try:
        asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


__all__ = [
    "LocalEvaluationRuntime",
    "main",
]
