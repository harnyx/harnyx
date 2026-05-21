"""Runtime wiring for the validator runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

import bittensor as bt

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.clients import PLATFORM
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_factory import CachedLlmProviderRegistry
from harnyx_commons.llm.provider_types import BEDROCK_PROVIDER, parse_builtin_provider_name
from harnyx_commons.llm.routing import ResolvedLlmRoute, resolve_llm_route
from harnyx_commons.llm.tool_models import ALLOWED_TOOL_MODELS
from harnyx_commons.miner_task_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
)
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import build_sandbox_options, create_sandbox_manager
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationOutput
from harnyx_commons.tools.invocation_clients import build_tool_invocation_clients
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.runtime_invoker import (
    RuntimeToolInvoker,
    build_miner_sandbox_tool_invoker,
)
from harnyx_commons.tools.token_semaphore import DEFAULT_TOOL_CONCURRENCY_LIMITS, ToolConcurrencyLimiter
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker, SandboxClient
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.platform import PlatformPort
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.restore_batch import RestoreEvaluationBatch
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
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.evaluation_record import CompactEvaluationRecordStore
from harnyx_validator.infrastructure.state.restore_inbox import InMemoryRestoreInbox
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.infrastructure.subtensor.client import RuntimeSubtensorClient
from harnyx_validator.infrastructure.subtensor.hotkey import create_wallet
from harnyx_validator.infrastructure.tools.platform_client import HttpPlatformClient
from harnyx_validator.runtime.registration_metadata import resolve_validator_registration_metadata
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageProvider
from harnyx_validator.runtime.restore_worker import RestoreWorker
from harnyx_validator.runtime.settings import Settings

logger = logging.getLogger("harnyx_validator.runtime")

_SCORING_LLM_MODEL = "moonshotai/Kimi-K2.5-TEE"
_SCORING_LLM_REASONING_EFFORT = "high"
_SEARCH_PROVIDER_TOOLS = frozenset(("search_web", "search_ai", "fetch_page"))
_BATCH_BLOCKING_LANE_NAME = "validator-batch-blocking"


class _ProviderTrackingToolExecutor(ToolExecutor):
    def __init__(
        self,
        *,
        session_registry: InMemorySessionRegistry,
        receipt_log: InMemoryReceiptLog,
        usage_tracker: UsageTracker,
        tool_invoker: RuntimeToolInvoker,
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

    async def _invoke_tool_output_async(self, request: ToolInvocationRequest) -> ToolInvocationOutput:
        provider_key = _provider_key_from_request(
            request=request,
            search_provider_name=self._search_provider_name,
            llm_route_resolver=self._llm_route_resolver,
        )
        try:
            response = await super()._invoke_tool_output_async(request)
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
    scoring_llm_provider: LlmProviderPort | None
    tool_invoker: RuntimeToolInvoker
    tool_executor: ToolExecutor
    tool_concurrency_limiter: ToolConcurrencyLimiter
    subtensor_client: SubtensorClientPort
    scoring_service: EvaluationScoringService
    weight_submission_service: WeightSubmissionService
    create_entrypoint_invoker: Callable[[SandboxClient], EntrypointInvoker]
    create_evaluation_orchestrator: Callable[[SandboxClient], TaskRunOrchestrator]
    build_sandbox_options: Callable[[], SandboxOptions]
    platform_client: PlatformPort | None
    batch_inbox: InMemoryBatchInbox
    restore_worker: RestoreWorker
    status_provider: StatusProvider
    batch_activity: BatchActivityTracker
    tool_route_deps_provider: Callable[[], ToolRouteDeps]
    control_deps_provider: Callable[[], ValidatorControlDeps]
    inbound_auth_verifier: BittensorSr25519InboundVerifier

    def register_with_platform(self) -> None:
        _register_with_platform(
            self.settings,
            self.platform_hotkey,
            self.settings.platform_api.validator_public_base_url,
        )


@dataclass(frozen=True, slots=True)
class InMemoryState:
    session_registry: InMemorySessionRegistry
    token_registry: InMemoryTokenRegistry
    receipt_log: InMemoryReceiptLog
    evaluation_records: EvaluationRecordPort
    progress_tracker: FileBackedRunProgress
    batch_inbox: InMemoryBatchInbox
    restore_inbox: InMemoryRestoreInbox
    batch_activity: BatchActivityTracker
    usage_tracker: UsageTracker
    tool_concurrency_limiter: ToolConcurrencyLimiter
    session_manager: SessionManager


def build_runtime(settings: Settings | None = None) -> RuntimeContext:
    """Construct the runtime context shared across CLI commands."""
    resolved = settings or Settings.load()
    logger.info("loading validator runtime configuration", extra={"settings": resolved})

    state = _build_state(resolved)
    platform_client, platform_hotkey, subtensor_client = _build_external_clients(resolved)

    search_client, llm_provider_registry, tool_llm_provider, scoring_llm_provider, scoring_route = _build_llm_clients(
        resolved
    )
    tool_invoker, tool_executor = _build_tooling(
        state=state,
        resolved=resolved,
        search_client=search_client,
        tool_llm_provider=tool_llm_provider,
    )

    scoring_service, weight_submission_service = _build_services(
        resolved=resolved,
        scoring_llm_provider=scoring_llm_provider,
        scoring_route=scoring_route,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )

    sandbox_manager = create_sandbox_manager(logger_name="harnyx_validator.sandbox")
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
        restore_worker,
    ) = _build_http_dependencies(
        resolved=resolved,
        state=state,
        tool_executor=tool_executor,
        validator_hotkey=platform_hotkey,
        platform_client=platform_client,
    )

    batch_blocking_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=_BATCH_BLOCKING_LANE_NAME,
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
        search_client=search_client,
        llm_provider_registry=llm_provider_registry,
        tool_llm_provider=tool_llm_provider,
        scoring_llm_provider=scoring_llm_provider,
        tool_invoker=tool_invoker,
        tool_executor=tool_executor,
        tool_concurrency_limiter=state.tool_concurrency_limiter,
        subtensor_client=subtensor_client,
        scoring_service=scoring_service,
        weight_submission_service=weight_submission_service,
        create_entrypoint_invoker=entrypoint_factory,
        create_evaluation_orchestrator=orchestrator_factory,
        build_sandbox_options=options_factory,
        platform_client=platform_client,
        batch_inbox=state.batch_inbox,
        restore_worker=restore_worker,
        status_provider=status_provider,
        batch_activity=state.batch_activity,
        tool_route_deps_provider=tool_route_provider,
        control_deps_provider=control_provider,
        inbound_auth_verifier=inbound_auth_verifier,
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
    batch_inbox = InMemoryBatchInbox()
    restore_inbox = InMemoryRestoreInbox()
    batch_activity = BatchActivityTracker()
    tool_concurrency_limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    usage_tracker = UsageTracker()
    session_manager = SessionManager(session_registry, token_registry)
    return InMemoryState(
        session_registry=session_registry,
        token_registry=token_registry,
        receipt_log=receipt_log,
        evaluation_records=evaluation_records,
        progress_tracker=progress_tracker,
        batch_inbox=batch_inbox,
        restore_inbox=restore_inbox,
        batch_activity=batch_activity,
        usage_tracker=usage_tracker,
        tool_concurrency_limiter=tool_concurrency_limiter,
        session_manager=session_manager,
    )


def _build_external_clients(settings: Settings) -> tuple[PlatformPort, bt.Keypair, SubtensorClientPort]:
    platform_client, platform_hotkey = _create_platform_client(settings)
    subtensor_client = _build_subtensor_client(settings)
    return platform_client, platform_hotkey, subtensor_client


def _build_llm_clients(
    settings: Settings,
) -> tuple[
    WebSearchProviderPort | None,
    CachedLlmProviderRegistry,
    LlmProviderPort | None,
    LlmProviderPort | None,
    ResolvedLlmRoute,
]:
    invocation_clients = build_tool_invocation_clients(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
        lazy_search=False,
        require_search=True,
    )
    scoring_route = _resolve_scoring_judge_route(settings)
    scoring_llm_provider = invocation_clients.llm_provider_registry.resolve(scoring_route.provider)
    return (
        invocation_clients.search_client,
        invocation_clients.llm_provider_registry,
        invocation_clients.tool_llm_provider,
        scoring_llm_provider,
        scoring_route,
    )


def _resolve_scoring_judge_route(settings: Settings) -> ResolvedLlmRoute:
    if settings.llm.scoring_llm_provider == BEDROCK_PROVIDER:
        raise ValueError("SCORING_LLM_PROVIDER='bedrock' is not supported")
    return resolve_llm_route(
        surface="scoring",
        default_provider=settings.llm.scoring_llm_provider,
        model=_effective_scoring_llm_model(settings),
        overrides=settings.llm.llm_model_provider_overrides,
        allowed_providers={"bedrock", "chutes", "vertex"},
    )


def _effective_scoring_llm_model(settings: Settings) -> str:
    override = settings.llm.scoring_llm_model_override_value
    if override is not None:
        return override
    return _SCORING_LLM_MODEL


def _build_tooling(
    *,
    state: InMemoryState,
    resolved: Settings,
    search_client: WebSearchProviderPort | None,
    tool_llm_provider: LlmProviderPort | None,
) -> tuple[RuntimeToolInvoker, ToolExecutor]:
    tool_invoker = build_miner_sandbox_tool_invoker(
        state.receipt_log,
        web_search_client=search_client,
        web_search_provider_name=resolved.llm.search_provider,
        llm_provider=tool_llm_provider,
        llm_provider_name=resolved.llm.tool_llm_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    tool_executor = _ProviderTrackingToolExecutor(
        session_registry=state.session_registry,
        receipt_log=state.receipt_log,
        usage_tracker=state.usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=state.token_registry,
        clock=_clock,
        progress=state.progress_tracker,
        search_provider_name=resolved.llm.search_provider,
        llm_route_resolver=_build_tool_route_resolver(resolved),
    )
    return tool_invoker, tool_executor


def _build_services(
    *,
    resolved: Settings,
    scoring_llm_provider: LlmProviderPort | None,
    scoring_route: ResolvedLlmRoute,
    subtensor_client: SubtensorClientPort,
    platform_client: PlatformPort,
) -> tuple[EvaluationScoringService, WeightSubmissionService]:
    scoring_service = _create_scoring_service(
        resolved,
        scoring_llm_provider,
        scoring_route=scoring_route,
    )
    weight_submission_service = _build_weight_service(
        resolved,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )
    return scoring_service, weight_submission_service


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
    validator_hotkey: bt.Keypair,
    platform_client: PlatformPort,
) -> tuple[
    Callable[[], ToolRouteDeps],
    Callable[[], ValidatorControlDeps],
    StatusProvider,
    BittensorSr25519InboundVerifier,
    RestoreWorker,
]:
    status_provider = StatusProvider()
    resource_usage_provider = ValidatorResourceUsageProvider()
    inbound_auth = _build_inbound_auth(resolved, status_provider=status_provider)
    tool_route_provider = _make_dependency_provider(
        tool_executor,
        state.tool_concurrency_limiter,
    )
    accept_batch = AcceptEvaluationBatch(state.batch_inbox, status_provider, state.progress_tracker)
    restore_batch = RestoreEvaluationBatch(
        accepted_batches=accept_batch,
        platform=platform_client,
        batch_activity=state.batch_activity,
    )
    restore_worker = RestoreWorker(
        restore_service=restore_batch,
        restore_inbox=state.restore_inbox,
    )
    control_provider = _make_control_provider(
        accept_batch,
        restore_batch,
        restore_worker,
        status_provider,
        inbound_auth,
        state.progress_tracker,
        validator_hotkey,
        resource_usage_provider,
        state.batch_activity,
    )
    return tool_route_provider, control_provider, status_provider, inbound_auth, restore_worker


def _create_platform_client(settings: Settings) -> tuple[PlatformPort, bt.Keypair]:
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
    return client, hotkey


def _register_with_platform(settings: Settings, hotkey: bt.Keypair, public_url: str | None) -> None:
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
    metadata = resolve_validator_registration_metadata()
    client = PlatformRegistrationClient(
        platform_base_url=base.rstrip("/"),
        hotkey=hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    register_with_retry(client, public_url.rstrip("/"), metadata=metadata, attempts=30)


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
    if request.tool in _SEARCH_PROVIDER_TOOLS:
        if search_provider_name is None:
            return None
        return search_provider_name, request.tool
    if request.tool != "llm_chat":
        return None
    model = _model_name_from_request(request)
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


def _model_name_from_request(request: ToolInvocationRequest) -> str | None:
    payload = dict(request.kwargs)
    if not payload and request.args:
        first_arg = request.args[0]
        if isinstance(first_arg, dict):
            payload = {key: value for key, value in first_arg.items() if isinstance(key, str)}
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
    accept_batch: AcceptEvaluationBatch,
    restore_batch: RestoreEvaluationBatch,
    restore_worker: RestoreWorker,
    status_provider: StatusProvider,
    inbound_auth: BittensorSr25519InboundVerifier,
    progress_tracker: FileBackedRunProgress,
    validator_hotkey: bt.Keypair,
    resource_usage_provider: ValidatorResourceUsageProvider | None = None,
    batch_activity: BatchActivityTracker | None = None,
) -> Callable[[], ValidatorControlDeps]:
    effective_resource_usage_provider = resource_usage_provider or ValidatorResourceUsageProvider()

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
            accept_batch=accept_batch,
            restore_batch=restore_batch,
            restore_worker=restore_worker,
            status_provider=status_provider,
            auth=auth,
            progress_tracker=progress_tracker,
            validator_hotkey=cast(StatusSigner, validator_hotkey),
            resource_usage_provider=effective_resource_usage_provider,
            batch_activity=batch_activity or BatchActivityTracker(),
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
    def factory() -> SandboxOptions:
        return build_sandbox_options(
            image=resolved.sandbox.sandbox_image,
            network=resolved.sandbox.sandbox_network,
            pull_policy=resolved.sandbox.sandbox_pull_policy,
            rpc_port=resolved.rpc_port,
            container_name="harnyx-sandbox-smoke",
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
) -> EvaluationScoringService:
    if provider is None:
        raise ValueError("scoring_llm_provider must be configured")
    scoring_provider = parse_builtin_provider_name(scoring_route.provider, component="scoring")
    config = EvaluationScoringConfig(
        provider=scoring_provider,
        model=scoring_route.model,
        temperature=settings.llm.scoring_llm_temperature,
        max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
        reasoning_effort=_SCORING_LLM_REASONING_EFFORT,
        timeout_seconds=settings.llm.scoring_llm_timeout_seconds,
    )
    return EvaluationScoringService(
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


__all__ = ["RuntimeContext", "build_runtime", "close_runtime_resources"]


class _SupportsAclose(Protocol):
    async def aclose(self) -> None: ...
