"""Runtime wiring for the validator runtime."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast, get_args, runtime_checkable
from uuid import UUID

import bittensor as bt

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.clients import PLATFORM
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import (
    CachedLlmProviderRegistry,
    build_cached_llm_provider_registry,
    build_routed_llm_provider,
)
from harnyx_commons.llm.provider_types import BEDROCK_PROVIDER
from harnyx_commons.llm.routing import ResolvedLlmRoute, resolve_llm_route
from harnyx_commons.llm.tool_models import ALLOWED_TOOL_MODELS, parse_miner_selected_llm_provider_model
from harnyx_commons.miner_task_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import build_sandbox_options, create_sandbox_manager
from harnyx_commons.tools.dto import ToolInvocationRequest, tool_payload_for_invocation
from harnyx_commons.tools.embedding_models import parse_miner_selected_embedding_provider_model
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.invocation_clients import build_optional_tool_embedding_provider
from harnyx_commons.tools.ports import EmbeddingProviderPort, WebSearchProviderPort
from harnyx_commons.tools.runtime_invoker import (
    EmbeddingProviderResolver,
    LlmProviderResolver,
    RuntimeToolInvoker,
    SearchProviderResolver,
    build_miner_sandbox_tool_invoker,
)
from harnyx_commons.tools.search_models import SearchProviderName
from harnyx_commons.tools.token_semaphore import DEFAULT_TOOL_CONCURRENCY_LIMITS, ToolConcurrencyLimiter
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.assigned_work import AssignedArtifactWork
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchSpec,
    PlatformOwnedTaskExecution,
    PlatformOwnedTaskResult,
)
from harnyx_validator.application.dto.registration import ValidatorRegistrationMetadata
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, score_platform_execution
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker, SandboxClient
from harnyx_validator.application.platform_tool_proxy import (
    PlatformToolProxyProxyToolInvoker,
    PlatformToolProxyScopeRegistry,
)
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.platform import PlatformPort, PlatformToolProxyPlatformPort
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_batch_prep import (
    SANDBOX_CONTAINER_NAME_PREFIX,
    SANDBOX_LABELS,
    BatchExecutionPlanner,
    EvaluationBatchConfig,
)
from harnyx_validator.application.similarity_judge import SimilarityJudge, SimilarityJudgeConfig
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.application.submit_weights import WeightSubmissionService
from harnyx_validator.infrastructure.auth.sr25519 import BittensorSr25519InboundVerifier
from harnyx_validator.infrastructure.http.routes import (
    StatusSigner,
    ToolRouteDeps,
    ValidatorControlDeps,
)
from harnyx_validator.infrastructure.platform.registration_client import (
    PlatformRegistrationClient,
    register_with_retry,
)
from harnyx_validator.infrastructure.state.evaluation_record import CompactEvaluationRecordStore
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.infrastructure.subtensor.client import RuntimeSubtensorClient
from harnyx_validator.infrastructure.subtensor.hotkey import create_wallet
from harnyx_validator.infrastructure.tools.platform_client import (
    AsyncPlatformToolProxyPlatformClient,
    HttpPlatformClient,
)
from harnyx_validator.runtime.agent_artifact import create_platform_agent_resolver
from harnyx_validator.runtime.platform_work_worker import (
    PlatformWorkWorker,
    ScoringExecutor,
    ScoringSlotConfig,
    ScoringSlotConfigEntry,
)
from harnyx_validator.runtime.registration_metadata import resolve_validator_registration_metadata
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageProvider
from harnyx_validator.runtime.settings import Settings

logger = logging.getLogger("harnyx_validator.runtime")

_SANDBOX_CPUSET_MAX_CPUS = 4
_SANDBOX_CPUSET_LABEL = "harnyx.sandbox.cpuset_cpus"
_DIRECT_SCORING_LLM_MODEL = "google/gemma-4-31B-turbo-TEE"
_SCORING_LLM_REASONING_EFFORT = "high"
_DUPLICATION_DETECTION_LLM_MODEL = "google/gemma-4-31B-turbo-TEE"
_SCORING_FALLBACK_MODELS = ("zai-org/GLM-5-TEE", "moonshotai/Kimi-K2.5-TEE")
_SCORING_SLOT_CONFIG = ScoringSlotConfig(
    entries=(
        ScoringSlotConfigEntry(
            model="google/gemma-4-31B-turbo-TEE",
            slot_limit=10,
            fallback_models=_SCORING_FALLBACK_MODELS,
        ),
        ScoringSlotConfigEntry(
            model="Qwen/Qwen3.6-27B-TEE",
            slot_limit=10,
            fallback_models=_SCORING_FALLBACK_MODELS,
        ),
    )
)
_DUPLICATION_DETECTION_FALLBACK_MODELS = (
    "moonshotai/Kimi-K2.5-TEE",
    "zai-org/GLM-5-TEE",
)
_SEARCH_PROVIDER_TOOLS = frozenset(("search_web", "search_ai", "fetch_page"))
_MINER_SELECTED_SEARCH_PROVIDERS = frozenset(get_args(SearchProviderName))
_BATCH_BLOCKING_LANE_NAME = "validator-batch-blocking"


class _ProviderTrackingToolExecutor(ToolExecutor):
    def __init__(
        self,
        *,
        session_registry: InMemorySessionRegistry,
        receipt_log: InMemoryReceiptLog,
        usage_tracker: UsageTracker,
        tool_invoker: ToolInvoker,
        token_registry: InMemoryTokenRegistry,
        clock: Callable[[], datetime],
        progress: FileBackedRunProgress,
        search_provider_name: str | None,
        llm_route_resolver: Callable[[str], ResolvedLlmRoute],
    ) -> None:
        super().__init__(
            session_registry=session_registry,
            receipt_log=receipt_log,
            usage_tracker=usage_tracker,
            tool_invoker=tool_invoker,
            token_registry=token_registry,
            clock=clock,
        )
        self._progress = progress
        self._search_provider_name = search_provider_name
        self._llm_route_resolver = llm_route_resolver

    async def _invoke_tool_output_async(
        self,
        request: ToolInvocationRequest,
        *,
        context: ToolInvocationContext | None,
    ) -> ToolInvocationOutput:
        provider_key = _provider_key_from_request(
            request=request,
            search_provider_name=self._search_provider_name,
            llm_route_resolver=self._llm_route_resolver,
        )
        try:
            response = await super()._invoke_tool_output_async(request, context=context)
        except ToolProviderError as exc:
            self._record_provider_call(request=request, provider_key=provider_key)
            if provider_key is not None:
                provider, model = provider_key
                self._progress.record_provider_failure(
                    session_id=request.session_id,
                    provider=provider,
                    model=model,
                    reason=_provider_failure_reason(exc),
                )
            raise
        self._record_provider_call(request=request, provider_key=provider_key)
        return response

    def _record_provider_call(
        self,
        *,
        request: ToolInvocationRequest,
        provider_key: tuple[str, str] | None,
    ) -> None:
        if provider_key is None:
            return
        provider, model = provider_key
        self._progress.record_provider_call(
            session_id=request.session_id,
            provider=provider,
            model=model,
        )


def _provider_failure_reason(exc: ToolProviderError) -> str:
    source = exc.__cause__ or exc
    reason = " ".join(str(source).split())
    return reason or type(source).__name__


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Aggregated runtime components for the validator service."""

    settings: Settings
    platform_hotkey: bt.Keypair
    sandbox_manager: DockerSandboxManager
    batch_blocking_executor: Executor
    session_manager: SessionManager
    session_registry: InMemorySessionRegistry
    token_registry: InMemoryTokenRegistry
    receipt_log: InMemoryReceiptLog
    evaluation_records: EvaluationRecordPort
    progress_tracker: FileBackedRunProgress
    usage_tracker: UsageTracker
    search_client: WebSearchProviderPort | None
    llm_provider_registry: CachedLlmProviderRegistry
    tool_llm_provider: LlmProviderPort | None
    tool_embedding_provider: EmbeddingProviderPort | None
    scoring_llm_provider: LlmProviderPort | None
    similarity_llm_provider: LlmProviderPort | None
    tool_invoker: ToolInvoker
    tool_executor: ToolExecutor
    tool_concurrency_limiter: ToolConcurrencyLimiter
    subtensor_client: SubtensorClientPort
    scoring_service: EvaluationScoringService
    similarity_judge: SimilarityJudge
    weight_submission_service: WeightSubmissionService
    create_entrypoint_invoker: Callable[[SandboxClient], EntrypointInvoker]
    create_evaluation_orchestrator: Callable[[SandboxClient], TaskRunOrchestrator]
    build_sandbox_options: Callable[[], SandboxOptions]
    platform_client: PlatformPort | None
    platform_tool_proxy_platform_client: PlatformToolProxyPlatformPort | None
    platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry
    platform_work_worker: PlatformWorkWorker | None
    status_provider: StatusProvider
    registration_metadata: ValidatorRegistrationMetadata
    batch_activity: BatchActivityTracker
    tool_route_deps_provider: Callable[[], ToolRouteDeps]
    control_deps_provider: Callable[[], ValidatorControlDeps]
    inbound_auth_verifier: BittensorSr25519InboundVerifier

    def register_with_platform(self) -> None:
        _register_with_platform(
            self.settings,
            self.platform_hotkey,
            self.settings.platform_api.validator_public_base_url,
            metadata=self.registration_metadata,
            attempts=30,
            delay_seconds=2.0,
        )

    def refresh_platform_registration(self) -> None:
        _register_with_platform(
            self.settings,
            self.platform_hotkey,
            self.settings.platform_api.validator_public_base_url,
            metadata=self.registration_metadata,
            attempts=1,
            delay_seconds=0.0,
        )


@dataclass(frozen=True, slots=True)
class RuntimeLlmClients:
    search_client: WebSearchProviderPort | None
    llm_provider_registry: CachedLlmProviderRegistry
    tool_llm_provider: LlmProviderPort | None
    tool_embedding_provider: EmbeddingProviderPort | None
    scoring_llm_provider: LlmProviderPort | None
    similarity_llm_provider: LlmProviderPort | None
    scoring_routes: Mapping[str, ResolvedLlmRoute]
    similarity_route: ResolvedLlmRoute


@dataclass(frozen=True, slots=True)
class InMemoryState:
    session_registry: InMemorySessionRegistry
    token_registry: InMemoryTokenRegistry
    receipt_log: InMemoryReceiptLog
    evaluation_records: EvaluationRecordPort
    progress_tracker: FileBackedRunProgress
    batch_activity: BatchActivityTracker
    usage_tracker: UsageTracker
    tool_concurrency_limiter: ToolConcurrencyLimiter
    session_manager: SessionManager
    platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry


def build_runtime(settings: Settings | None = None) -> RuntimeContext:
    """Construct the runtime context shared across CLI commands."""
    resolved = settings or Settings.load()
    logger.info("loading validator runtime configuration", extra={"settings": resolved})

    state = _build_state(resolved)
    (
        platform_client,
        platform_tool_proxy_platform_client,
        platform_hotkey,
        subtensor_client,
    ) = _build_external_clients(resolved)

    llm_clients = _build_llm_clients(resolved)
    tool_invoker, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform_tool_proxy_platform_client,
    )

    scoring_services, similarity_judge, weight_submission_service = _build_services(
        resolved=resolved,
        scoring_llm_provider=llm_clients.scoring_llm_provider,
        similarity_llm_provider=llm_clients.similarity_llm_provider,
        scoring_routes=llm_clients.scoring_routes,
        similarity_route=llm_clients.similarity_route,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )
    scoring_service = _direct_scoring_service(scoring_services)

    sandbox_manager = create_sandbox_manager(logger_name="harnyx_validator.sandbox")
    _cleanup_stale_sandbox_containers(sandbox_manager)
    entrypoint_factory, orchestrator_factory, options_factory = _build_factories(
        resolved=resolved,
        state=state,
        scoring_service=scoring_service,
    )
    (
        tool_route_provider,
        control_provider,
        status_provider,
        inbound_auth_verifier,
    ) = _build_http_dependencies(
        resolved=resolved,
        state=state,
        tool_executor=tool_executor,
        similarity_judge=similarity_judge,
        validator_hotkey=platform_hotkey,
        platform_client=platform_client,
        platform_tool_proxy_platform_client=platform_tool_proxy_platform_client,
    )
    registration_metadata = resolve_validator_registration_metadata()

    batch_blocking_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=_BATCH_BLOCKING_LANE_NAME,
    )
    platform_work_worker = _build_platform_work_worker(
        resolved=resolved,
        state=state,
        sandbox_manager=sandbox_manager,
        batch_blocking_executor=batch_blocking_executor,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
        scoring_services=scoring_services,
        orchestrator_factory=orchestrator_factory,
        options_factory=options_factory,
    )

    return RuntimeContext(
        settings=resolved,
        platform_hotkey=platform_hotkey,
        sandbox_manager=sandbox_manager,
        batch_blocking_executor=batch_blocking_executor,
        session_manager=state.session_manager,
        session_registry=state.session_registry,
        token_registry=state.token_registry,
        receipt_log=state.receipt_log,
        evaluation_records=state.evaluation_records,
        progress_tracker=state.progress_tracker,
        usage_tracker=state.usage_tracker,
        search_client=llm_clients.search_client,
        llm_provider_registry=llm_clients.llm_provider_registry,
        tool_llm_provider=llm_clients.tool_llm_provider,
        tool_embedding_provider=llm_clients.tool_embedding_provider,
        scoring_llm_provider=llm_clients.scoring_llm_provider,
        similarity_llm_provider=llm_clients.similarity_llm_provider,
        tool_invoker=tool_invoker,
        tool_executor=tool_executor,
        tool_concurrency_limiter=state.tool_concurrency_limiter,
        subtensor_client=subtensor_client,
        scoring_service=scoring_service,
        similarity_judge=similarity_judge,
        weight_submission_service=weight_submission_service,
        create_entrypoint_invoker=entrypoint_factory,
        create_evaluation_orchestrator=orchestrator_factory,
        build_sandbox_options=options_factory,
        platform_client=platform_client,
        platform_tool_proxy_platform_client=platform_tool_proxy_platform_client,
        platform_tool_proxy_scopes=state.platform_tool_proxy_scopes,
        platform_work_worker=platform_work_worker,
        status_provider=status_provider,
        registration_metadata=registration_metadata,
        batch_activity=state.batch_activity,
        tool_route_deps_provider=tool_route_provider,
        control_deps_provider=control_provider,
        inbound_auth_verifier=inbound_auth_verifier,
    )


def _build_platform_work_worker(
    *,
    resolved: Settings,
    state: InMemoryState,
    sandbox_manager: DockerSandboxManager,
    batch_blocking_executor: Executor,
    subtensor_client: SubtensorClientPort,
    platform_client: PlatformPort | None,
    scoring_services: Mapping[str, EvaluationScoringService],
    orchestrator_factory: Callable[[SandboxClient], TaskRunOrchestrator],
    options_factory: Callable[[], SandboxOptions],
) -> PlatformWorkWorker | None:
    if platform_client is None:
        return None
    agent_resolver = create_platform_agent_resolver(platform_client)
    config = EvaluationBatchConfig()
    planner = BatchExecutionPlanner(
        subtensor_client=subtensor_client,
        sandbox_manager=sandbox_manager,
        session_manager=state.session_manager,
        evaluation_records=state.evaluation_records,
        receipt_log=state.receipt_log,
        blocking_executor=batch_blocking_executor,
        orchestrator_factory=orchestrator_factory,
        sandbox_options_factory=options_factory,
        agent_resolver=agent_resolver,
        progress=state.progress_tracker,
        activity=state.batch_activity,
        config=config,
        platform_tool_proxy_scopes=state.platform_tool_proxy_scopes,
    )

    async def execute_artifact_assignments(
        artifact_id: UUID,
        assigned_work: AssignedArtifactWork,
        close_requested: asyncio.Event,
        result_queue: asyncio.Queue[PlatformOwnedTaskResult | PlatformOwnedTaskExecution],
    ) -> None:
        first_assignment = await assigned_work.take_for_startup()
        if first_assignment.artifact.artifact_id != artifact_id:
            raise ValueError("platform work artifact queue received mismatched first assignment")
        initial_assignments = [first_assignment]
        while True:
            try:
                assignment = assigned_work.take_nowait_for_startup()
            except asyncio.QueueEmpty:
                break
            if assignment.artifact.artifact_id != artifact_id:
                raise ValueError("platform work artifact queue received mismatched assignment")
            initial_assignments.append(assignment)
        batch = MinerTaskBatchSpec(
            batch_id=first_assignment.batch_id,
            cutoff_at="platform-owned",
            created_at="platform-owned",
            tasks=tuple(assignment.task for assignment in initial_assignments),
            artifacts=(first_assignment.artifact,),
        )
        run_ctx = planner.build_run_context(batch)
        _, scheduler = planner.prepare_execution(run_ctx, batch)
        await scheduler.run_assigned_artifact_queue(
            batch_id=first_assignment.batch_id,
            artifact=first_assignment.artifact,
            initial_assignments=tuple(initial_assignments),
            assigned_work=assigned_work,
            close_requested=close_requested,
            result_queue=result_queue,
        )

    score_execution_by_model = {
        model: _score_platform_execution_with(scoring_service)
        for model, scoring_service in scoring_services.items()
    }
    return PlatformWorkWorker(
        platform=platform_client,
        execute_artifact_assignments=execute_artifact_assignments,
        score_execution_by_model=score_execution_by_model,
        scoring_slot_config=_SCORING_SLOT_CONFIG,
        target_concurrency=config.artifact_task_parallelism,
        max_active_artifacts=config.artifact_parallelism,
    )


def _score_platform_execution_with(scoring_service: EvaluationScoringService) -> ScoringExecutor:
    async def score(execution: PlatformOwnedTaskExecution) -> PlatformOwnedTaskResult:
        return await score_platform_execution(
            scoring_service,
            execution,
            convert_scoring_error=True,
        )

    return score


def _direct_scoring_service(
    scoring_services: Mapping[str, EvaluationScoringService],
) -> EvaluationScoringService:
    return scoring_services[_DIRECT_SCORING_LLM_MODEL]


def _cleanup_stale_sandbox_containers(sandbox_manager: DockerSandboxManager) -> None:
    sandbox_manager.cleanup_stale_sandbox_containers(
        labels=SANDBOX_LABELS,
        name_prefix=f"{SANDBOX_CONTAINER_NAME_PREFIX}-",
    )


def _build_state(
    settings: Settings,
    *,
    progress_storage_root: Path | None = None,
) -> InMemoryState:
    session_registry = InMemorySessionRegistry()
    token_registry = InMemoryTokenRegistry()
    receipt_log = InMemoryReceiptLog()
    evaluation_records = CompactEvaluationRecordStore()
    progress_tracker = FileBackedRunProgress(
        storage_root=progress_storage_root or settings.validator_state_dir / "run-progress",
    )
    progress_tracker.prune_stale_batch_dirs_older_than(
        datetime.now(UTC) - timedelta(seconds=settings.run_progress_retention_seconds)
    )
    batch_activity = BatchActivityTracker()
    tool_concurrency_limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    usage_tracker = UsageTracker()
    session_manager = SessionManager(session_registry, token_registry)
    platform_tool_proxy_scopes = PlatformToolProxyScopeRegistry()
    return InMemoryState(
        session_registry=session_registry,
        token_registry=token_registry,
        receipt_log=receipt_log,
        evaluation_records=evaluation_records,
        progress_tracker=progress_tracker,
        batch_activity=batch_activity,
        usage_tracker=usage_tracker,
        tool_concurrency_limiter=tool_concurrency_limiter,
        session_manager=session_manager,
        platform_tool_proxy_scopes=platform_tool_proxy_scopes,
    )


def _build_external_clients(
    settings: Settings,
) -> tuple[PlatformPort, PlatformToolProxyPlatformPort, bt.Keypair, SubtensorClientPort]:
    platform_client, platform_tool_proxy_client, platform_hotkey = _create_platform_client(settings)
    subtensor_client = _build_subtensor_client(settings)
    return platform_client, platform_tool_proxy_client, platform_hotkey, subtensor_client


def _build_llm_clients(settings: Settings) -> RuntimeLlmClients:
    llm_provider_registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    scoring_routes = {
        entry.model: _resolve_scoring_judge_route(settings, model=entry.model)
        for entry in _SCORING_SLOT_CONFIG.entries
    }
    similarity_route = _resolve_similarity_judge_route(settings)
    scoring_provider = build_routed_llm_provider(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=llm_provider_registry,
    )
    similarity_provider = build_routed_llm_provider(
        surface="duplication_detection",
        default_provider=settings.llm.similarity_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=llm_provider_registry,
    )
    return RuntimeLlmClients(
        search_client=None,
        llm_provider_registry=llm_provider_registry,
        tool_llm_provider=None,
        tool_embedding_provider=build_optional_tool_embedding_provider(settings.llm),
        scoring_llm_provider=scoring_provider,
        similarity_llm_provider=similarity_provider,
        scoring_routes=scoring_routes,
        similarity_route=similarity_route,
    )


def _resolve_scoring_judge_route(settings: Settings, *, model: str) -> ResolvedLlmRoute:
    if settings.llm.scoring_llm_provider == BEDROCK_PROVIDER:
        raise ValueError("SCORING_LLM_PROVIDER='bedrock' is not supported")
    return resolve_llm_route(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        model=model,
        overrides=settings.llm.llm_model_provider_overrides,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )


def _resolve_similarity_judge_route(settings: Settings) -> ResolvedLlmRoute:
    return resolve_llm_route(
        surface="duplication_detection",
        default_provider=settings.llm.similarity_llm_provider,
        model=_effective_similarity_llm_model(settings),
        overrides=settings.llm.llm_model_provider_overrides,
        allowed_providers={"bedrock", "chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )


def _effective_similarity_llm_model(settings: Settings) -> str:
    override = settings.llm.similarity_llm_model_override_value
    if override is not None:
        return override
    return _DUPLICATION_DETECTION_LLM_MODEL


def _similarity_judge_fallback_models(settings: Settings) -> tuple[str, ...]:
    return _fallback_tail_after_primary(
        primary_model=_effective_similarity_llm_model(settings),
        ordered_models=(_DUPLICATION_DETECTION_LLM_MODEL, *_DUPLICATION_DETECTION_FALLBACK_MODELS),
        fallback_tail=_DUPLICATION_DETECTION_FALLBACK_MODELS,
    )


def _fallback_tail_after_primary(
    *,
    primary_model: str,
    ordered_models: tuple[str, ...],
    fallback_tail: tuple[str, ...],
) -> tuple[str, ...]:
    try:
        primary_index = ordered_models.index(primary_model)
    except ValueError:
        return fallback_tail
    return ordered_models[primary_index + 1 :]


def _build_proxy_tooling(
    *,
    state: InMemoryState,
    platform_tool_proxy_platform_client: PlatformToolProxyPlatformPort,
) -> tuple[ToolInvoker, ToolExecutor]:
    local_invoker = build_miner_sandbox_tool_invoker(
        state.receipt_log,
        web_search_client=None,
        web_search_provider_name=None,
        llm_provider=None,
        llm_provider_name=None,
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    tool_invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=local_invoker,
        platform_tool_proxy_platform=platform_tool_proxy_platform_client,
        scopes=state.platform_tool_proxy_scopes,
    )
    return tool_invoker, ToolExecutor(
        session_registry=state.session_registry,
        receipt_log=state.receipt_log,
        usage_tracker=state.usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=state.token_registry,
        clock=_clock,
    )


def _build_local_provider_tooling(
    *,
    state: InMemoryState,
    resolved: Settings,
    search_client: WebSearchProviderPort | None,
    tool_llm_provider: LlmProviderPort | None,
    tool_embedding_provider: EmbeddingProviderPort | None = None,
    search_provider_resolver: SearchProviderResolver | None = None,
    llm_provider_resolver: LlmProviderResolver | None = None,
    embedding_provider_resolver: EmbeddingProviderResolver | None = None,
) -> tuple[ToolInvoker, ToolExecutor]:
    local_invoker = build_miner_sandbox_tool_invoker(
        state.receipt_log,
        web_search_client=search_client,
        web_search_provider_name=resolved.llm.search_provider,
        web_search_provider_resolver=search_provider_resolver,
        llm_provider=tool_llm_provider,
        llm_provider_name=resolved.llm.tool_llm_provider,
        llm_provider_resolver=llm_provider_resolver,
        embedding_provider=tool_embedding_provider,
        embedding_provider_name=resolved.llm.tool_embedding_provider if tool_embedding_provider is not None else None,
        embedding_provider_resolver=embedding_provider_resolver,
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    return local_invoker, _ProviderTrackingToolExecutor(
        session_registry=state.session_registry,
        receipt_log=state.receipt_log,
        usage_tracker=state.usage_tracker,
        tool_invoker=local_invoker,
        token_registry=state.token_registry,
        clock=_clock,
        progress=state.progress_tracker,
        search_provider_name=resolved.llm.search_provider,
        llm_route_resolver=_build_tool_route_resolver(resolved),
    )


def _build_services(
    *,
    resolved: Settings,
    scoring_llm_provider: LlmProviderPort | None,
    similarity_llm_provider: LlmProviderPort | None,
    scoring_routes: Mapping[str, ResolvedLlmRoute],
    similarity_route: ResolvedLlmRoute,
    subtensor_client: SubtensorClientPort,
    platform_client: PlatformPort,
) -> tuple[dict[str, EvaluationScoringService], SimilarityJudge, WeightSubmissionService]:
    scoring_services = {
        entry.model: _create_scoring_service(
            resolved,
            scoring_llm_provider,
            scoring_route=scoring_routes[entry.model],
            fallback_models=entry.fallback_models,
        )
        for entry in _SCORING_SLOT_CONFIG.entries
    }
    similarity_judge = _create_similarity_judge(
        resolved,
        similarity_llm_provider,
        similarity_route=similarity_route,
    )
    weight_submission_service = _build_weight_service(
        resolved,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )
    return scoring_services, similarity_judge, weight_submission_service


def _build_factories(
    *,
    resolved: Settings,
    state: InMemoryState,
    scoring_service: EvaluationScoringService,
) -> tuple[
    Callable[[SandboxClient], EntrypointInvoker],
    Callable[[SandboxClient], TaskRunOrchestrator],
    Callable[[], SandboxOptions],
]:
    entrypoint_factory = _make_entrypoint_factory(state.session_registry, state.token_registry, state.receipt_log)
    orchestrator_factory = _make_orchestrator_factory(
        state.receipt_log,
        state.session_registry,
        scoring_service,
        entrypoint_factory,
    )
    options_factory = _make_options_factory(resolved)
    return entrypoint_factory, orchestrator_factory, options_factory


def _build_http_dependencies(
    *,
    resolved: Settings,
    state: InMemoryState,
    tool_executor: ToolExecutor,
    similarity_judge: SimilarityJudge,
    validator_hotkey: bt.Keypair,
    platform_client: PlatformPort,
    platform_tool_proxy_platform_client: PlatformToolProxyPlatformPort,
) -> tuple[
    Callable[[], ToolRouteDeps],
    Callable[[], ValidatorControlDeps],
    StatusProvider,
    BittensorSr25519InboundVerifier,
]:
    status_provider = StatusProvider()
    resource_usage_provider = ValidatorResourceUsageProvider()
    inbound_auth = _build_inbound_auth(resolved, status_provider=status_provider)
    tool_route_provider = _make_dependency_provider(
        tool_executor,
        state.tool_concurrency_limiter,
    )
    control_provider = _make_control_provider(
        resolved,
        status_provider,
        inbound_auth,
        validator_hotkey,
        resource_usage_provider,
        state.batch_activity,
        similarity_judge,
        platform_tool_proxy_platform_client,
        state.platform_tool_proxy_scopes,
    )
    return tool_route_provider, control_provider, status_provider, inbound_auth


def _create_platform_client(settings: Settings) -> tuple[PlatformPort, PlatformToolProxyPlatformPort, bt.Keypair]:
    base_url = settings.platform_api.platform_base_url
    if not base_url:
        raise RuntimeError("PLATFORM_BASE_URL must be configured")
    base_url_str = str(base_url)
    wallet = create_wallet(settings.subtensor)
    hotkey = wallet.hotkey
    if hotkey is None:
        raise RuntimeError("wallet hotkey is unavailable for platform signing")
    normalized_base = base_url_str.rstrip("/") or base_url_str
    client = HttpPlatformClient(
        base_url=normalized_base,
        hotkey=hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    platform_tool_proxy_client = AsyncPlatformToolProxyPlatformClient(
        base_url=normalized_base,
        hotkey=hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    return client, platform_tool_proxy_client, hotkey


def _register_with_platform(
    settings: Settings,
    hotkey: bt.Keypair,
    public_url: str | None,
    *,
    metadata: ValidatorRegistrationMetadata,
    attempts: int,
    delay_seconds: float,
) -> None:
    if not public_url:
        raise RuntimeError("VALIDATOR_PUBLIC_BASE_URL must be configured")
    base = settings.platform_api.platform_base_url
    if not base:
        raise RuntimeError("PLATFORM_BASE_URL must be configured for registration")
    logger.info(
        "registering validator with platform",
        extra={
            "data": {
                "platform_base_url": base.rstrip("/"),
                "validator_public_base_url": public_url.rstrip("/"),
                "validator_hotkey_ss58": hotkey.ss58_address,
            }
        },
    )
    client = PlatformRegistrationClient(
        platform_base_url=base.rstrip("/"),
        hotkey=hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    register_with_retry(
        client,
        public_url.rstrip("/"),
        metadata=metadata,
        attempts=attempts,
        delay_seconds=delay_seconds,
    )


def _build_subtensor_client(resolved: Settings) -> SubtensorClientPort:
    client = RuntimeSubtensorClient(resolved.subtensor)
    try:
        client.connect()
    except Exception as exc:
        logger.warning("subtensor client initialization failed", exc_info=exc)
    return client


def _provider_key_from_request(
    *,
    request: ToolInvocationRequest,
    search_provider_name: str | None,
    llm_route_resolver: Callable[[str], ResolvedLlmRoute],
) -> tuple[str, str] | None:
    payload = _payload_for_evidence(request)
    has_explicit_provider = "provider" in payload
    if request.tool in _SEARCH_PROVIDER_TOOLS:
        selected_provider = _explicit_search_provider(payload)
        if selected_provider is not None:
            return selected_provider, request.tool
        if has_explicit_provider:
            return None
        if search_provider_name is None:
            return None
        return search_provider_name, request.tool
    if request.tool == "embed_text":
        return _explicit_embedding_provider_model(payload)
    if request.tool != "llm_chat":
        return None
    selected_llm = _explicit_llm_provider_model(payload)
    if selected_llm is not None:
        return selected_llm
    if has_explicit_provider:
        return None
    model = _model_name_from_payload(payload)
    if model is None:
        return None
    route = llm_route_resolver(model)
    return route.provider, route.model


def _build_tool_route_resolver(settings: Settings) -> Callable[[str], ResolvedLlmRoute]:
    def resolve(model: str) -> ResolvedLlmRoute:
        return resolve_llm_route(
            surface="tool",
            default_provider=settings.llm.tool_llm_provider,
            model=model,
            overrides=settings.llm.llm_model_provider_overrides,
            allowed_providers={"chutes", "vertex"},
            allow_custom_openai_compatible=True,
        )

    return resolve


def _payload_for_evidence(request: ToolInvocationRequest) -> Mapping[str, object]:
    try:
        return tool_payload_for_invocation(request)
    except (TypeError, ValueError):
        return {}


def _explicit_search_provider(payload: Mapping[str, object]) -> SearchProviderName | None:
    raw_provider = payload.get("provider")
    if not isinstance(raw_provider, str):
        return None
    if raw_provider not in _MINER_SELECTED_SEARCH_PROVIDERS:
        return None
    return cast(SearchProviderName, raw_provider)


def _explicit_llm_provider_model(payload: Mapping[str, object]) -> tuple[str, str] | None:
    raw_provider = payload.get("provider")
    if not isinstance(raw_provider, str):
        return None
    raw_model = payload.get("model")
    model = raw_model if isinstance(raw_model, str) else None
    try:
        selected = parse_miner_selected_llm_provider_model(provider=raw_provider, model=model)
    except ValueError:
        return None
    return selected.provider, selected.model


def _explicit_embedding_provider_model(payload: Mapping[str, object]) -> tuple[str, str] | None:
    raw_provider = payload.get("provider")
    if not isinstance(raw_provider, str):
        return None
    raw_model = payload.get("model")
    model = raw_model if isinstance(raw_model, str) else None
    try:
        selected = parse_miner_selected_embedding_provider_model(provider=raw_provider, model=model)
    except ValueError:
        return None
    return selected.provider, selected.model


def _model_name_from_payload(payload: Mapping[str, object]) -> str | None:
    model_raw = payload.get("model")
    if not isinstance(model_raw, str):
        return None
    model = model_raw.strip()
    if not model:
        return None
    return model


def _make_dependency_provider(
    tool_executor: ToolExecutor,
    tool_concurrency_limiter: ToolConcurrencyLimiter,
) -> Callable[[], ToolRouteDeps]:
    def provider() -> ToolRouteDeps:
        return ToolRouteDeps(
            tool_executor=tool_executor,
            tool_concurrency_limiter=tool_concurrency_limiter,
        )

    return provider


def _make_control_provider(
    settings: Settings,
    status_provider: StatusProvider,
    inbound_auth: BittensorSr25519InboundVerifier,
    validator_hotkey: bt.Keypair,
    resource_usage_provider: ValidatorResourceUsageProvider | None = None,
    batch_activity: BatchActivityTracker | None = None,
    similarity_judge: SimilarityJudge | None = None,
    platform_tool_proxy_platform_client: PlatformToolProxyPlatformPort | None = None,
    platform_tool_proxy_scopes: PlatformToolProxyScopeRegistry | None = None,
) -> Callable[[], ValidatorControlDeps]:
    effective_resource_usage_provider = resource_usage_provider or ValidatorResourceUsageProvider()
    is_chutes_configured = bool(settings.chutes_api_key_value.strip())
    is_openrouter_configured = bool(settings.openrouter_api_key_value.strip())

    async def auth(
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        return await asyncio.to_thread(
            _verify_request,
            inbound_auth,
            method=method,
            path_qs=path_qs,
            body=body,
            authorization_header=authorization_header,
        )

    def provider() -> ValidatorControlDeps:
        return ValidatorControlDeps(
            status_provider=status_provider,
            auth=auth,
            validator_hotkey=cast(StatusSigner, validator_hotkey),
            resource_usage_provider=effective_resource_usage_provider,
            batch_activity=batch_activity or BatchActivityTracker(),
            is_chutes_configured=is_chutes_configured,
            is_openrouter_configured=is_openrouter_configured,
            platform_tool_proxy_platform=platform_tool_proxy_platform_client,
            platform_tool_proxy_scopes=platform_tool_proxy_scopes,
            similarity_judge=similarity_judge,
        )

    return provider


def _make_entrypoint_factory(
    session_registry: InMemorySessionRegistry,
    token_registry: InMemoryTokenRegistry,
    receipt_log: InMemoryReceiptLog,
) -> Callable[[SandboxClient], EntrypointInvoker]:
    def factory(client: SandboxClient) -> EntrypointInvoker:
        return EntrypointInvoker(
            session_registry=session_registry,
            sandbox_client=client,
            token_registry=token_registry,
            receipt_log=receipt_log,
        )

    return factory


def _make_orchestrator_factory(
    receipt_log: InMemoryReceiptLog,
    session_registry: InMemorySessionRegistry,
    scoring_service: EvaluationScoringService,
    entrypoint_factory: Callable[[SandboxClient], EntrypointInvoker],
) -> Callable[[SandboxClient], TaskRunOrchestrator]:
    def factory(client: SandboxClient) -> TaskRunOrchestrator:
        invoker = entrypoint_factory(client)
        return TaskRunOrchestrator(
            entrypoint_invoker=invoker,
            receipt_log=receipt_log,
            scoring_service=scoring_service,
            session_registry=session_registry,
            clock=_clock,
        )

    return factory


def _make_options_factory(resolved: Settings) -> Callable[[], SandboxOptions]:
    allowed_cpu_ids = sorted(os.sched_getaffinity(0))
    selected_cpu_ids = allowed_cpu_ids[:_SANDBOX_CPUSET_MAX_CPUS]
    cpuset_cpus = ",".join(str(cpu_id) for cpu_id in selected_cpu_ids)
    logger.info(
        "configured shared sandbox CPU set",
        extra={
            "data": {
                "sandbox_cpuset_cpus": cpuset_cpus,
                "sandbox_cpuset_max_cpus": _SANDBOX_CPUSET_MAX_CPUS,
            },
        },
    )

    def factory() -> SandboxOptions:
        base_options = build_sandbox_options(
            image=resolved.sandbox.sandbox_image,
            network=resolved.sandbox.sandbox_network,
            pull_policy=resolved.sandbox.sandbox_pull_policy,
            rpc_port=resolved.rpc_port,
            container_name="harnyx-sandbox-smoke",
            labels={_SANDBOX_CPUSET_LABEL: cpuset_cpus},
        )
        return replace(
            base_options,
            extra_args=base_options.extra_args + ("--cpuset-cpus", cpuset_cpus),
        )

    return factory


def _build_inbound_auth(
    resolved: Settings,
    *,
    status_provider: StatusProvider | None = None,
) -> BittensorSr25519InboundVerifier:
    subtensor_settings = resolved.subtensor
    endpoint = subtensor_settings.endpoint.strip()
    network_or_endpoint = endpoint or subtensor_settings.network
    subtensor = bt.Subtensor(network=network_or_endpoint)
    try:
        subnet_info = subtensor.get_subnet_info(subtensor_settings.netuid)
    finally:
        try:
            subtensor.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug("subtensor close failed during inbound auth setup", exc_info=exc)

    if subnet_info is None:
        raise RuntimeError(f"unable to resolve subnet info (netuid={subtensor_settings.netuid})")
    owner_coldkey = subnet_info.owner_ss58
    if not owner_coldkey:
        raise RuntimeError(f"unable to resolve subnet owner coldkey (netuid={subtensor_settings.netuid})")
    owner_coldkey_ss58 = str(owner_coldkey)
    logger.info(
        "configured inbound platform request verifier",
        extra={
            "data": {
                "netuid": subtensor_settings.netuid,
                "allowed_platform_owner_coldkey_ss58": owner_coldkey_ss58,
            }
        },
    )
    return BittensorSr25519InboundVerifier(
        netuid=subtensor_settings.netuid,
        network=network_or_endpoint,
        owner_coldkey_ss58=owner_coldkey_ss58,
        on_refresh_succeeded=status_provider.mark_auth_ready if status_provider is not None else None,
        on_refresh_failed=status_provider.mark_auth_unavailable if status_provider is not None else None,
    )


def _create_scoring_service(
    settings: Settings,
    provider: LlmProviderPort | None,
    *,
    scoring_route: ResolvedLlmRoute,
    fallback_models: tuple[str, ...] = (),
) -> EvaluationScoringService:
    if provider is None:
        raise ValueError("scoring_llm_provider must be configured")
    config = EvaluationScoringConfig(
        provider=settings.llm.scoring_llm_provider,
        model=scoring_route.model,
        fallback_models=fallback_models,
        temperature=settings.llm.scoring_llm_temperature,
        max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
        reasoning_effort=_SCORING_LLM_REASONING_EFFORT,
        timeout_seconds=settings.llm.scoring_llm_timeout_seconds,
        retry_policy=settings.llm.scoring_llm_retry_policy,
    )
    return EvaluationScoringService(
        llm_provider=provider,
        config=config,
    )


def _create_similarity_judge(
    settings: Settings,
    provider: LlmProviderPort | None,
    *,
    similarity_route: ResolvedLlmRoute,
) -> SimilarityJudge:
    if provider is None:
        raise ValueError("similarity_llm_provider must be configured")
    config = SimilarityJudgeConfig(
        provider=settings.llm.similarity_llm_provider,
        model=similarity_route.model,
        fallback_models=_similarity_judge_fallback_models(settings),
        temperature=settings.llm.similarity_llm_temperature,
        max_output_tokens=settings.llm.similarity_llm_max_output_tokens,
        reasoning_effort=_SCORING_LLM_REASONING_EFFORT,
        timeout_seconds=settings.llm.similarity_llm_timeout_seconds,
        retry_policy=settings.llm.similarity_llm_retry_policy,
    )
    return SimilarityJudge(
        llm_provider=provider,
        config=config,
    )


def _build_weight_service(
    settings: Settings,
    subtensor_client: SubtensorClientPort,
    platform_client: PlatformPort,
) -> WeightSubmissionService:
    return WeightSubmissionService(
        subtensor=subtensor_client,
        netuid=settings.subtensor.netuid,
        clock=_clock,
        platform=platform_client,
    )


async def close_runtime_resources(runtime: RuntimeContext) -> None:
    """Best-effort shutdown of shared async clients/providers."""

    async def _aclose(obj: _SupportsAclose | None) -> None:
        if obj is None:
            return
        await obj.aclose()

    runtime.batch_blocking_executor.shutdown(wait=False, cancel_futures=True)

    for owned in _unique_aclose_targets(
        runtime.search_client,
        runtime.llm_provider_registry,
        _aclose_target(runtime.platform_tool_proxy_platform_client),
    ):
        await _aclose(owned)


def _unique_aclose_targets(*objects: _SupportsAclose | None) -> tuple[_SupportsAclose, ...]:
    seen: set[int] = set()
    unique: list[_SupportsAclose] = []
    for obj in objects:
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        unique.append(obj)
    return tuple(unique)


def _aclose_target(obj: object | None) -> _SupportsAclose | None:
    if isinstance(obj, _SupportsAclose):
        return obj
    return None


def _verify_request(
    verifier: BittensorSr25519InboundVerifier,
    *,
    method: str,
    path_qs: str,
    body: bytes,
    authorization_header: str | None,
) -> str:
    return verifier.verify(
        method=method,
        path_qs=path_qs,
        body=body or b"",
        authorization_header=authorization_header,
    )


def _clock() -> datetime:
    return datetime.now(UTC)


__all__ = ["RuntimeContext", "RuntimeToolInvoker", "build_runtime", "close_runtime_resources"]


@runtime_checkable
class _SupportsAclose(Protocol):
    async def aclose(self) -> None: ...
