"""Runtime wiring for the validator runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import bittensor as bt
from fastapi import Request

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.clients import CHUTES, DESEARCH, PLATFORM
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import build_sandbox_options, create_sandbox_manager
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.runtime_invoker import (
    ALLOWED_TOOL_MODELS,
    RuntimeToolInvoker,
    build_miner_sandbox_tool_invoker,
)
from harnyx_commons.tools.token_semaphore import TokenSemaphore
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker, SandboxClient
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.platform import PlatformPort
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_scoring import (
    EvaluationScoringConfig,
    EvaluationScoringService,
    TextEmbeddingPort,
)
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.application.submit_weights import WeightSubmissionService
from harnyx_validator.infrastructure.auth.sr25519 import BittensorSr25519InboundVerifier
from harnyx_validator.infrastructure.http.routes import ToolRouteDeps, ValidatorControlDeps
from harnyx_validator.infrastructure.platform.registration_client import (
    PlatformRegistrationClient,
    register_with_retry,
)
from harnyx_validator.infrastructure.scoring.factory import create_scoring_embedding_client
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.evaluation_record import InMemoryEvaluationRecordStore
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress
from harnyx_validator.infrastructure.subtensor.client import RuntimeSubtensorClient
from harnyx_validator.infrastructure.subtensor.hotkey import create_wallet
from harnyx_validator.infrastructure.tools.feed_search_provider import HttpFeedSearchToolProvider
from harnyx_validator.infrastructure.tools.platform_client import HttpPlatformClient
from harnyx_validator.runtime.llm_factory import create_llm_provider_factory
from harnyx_validator.runtime.settings import Settings

logger = logging.getLogger("harnyx_validator.runtime")

_SCORING_EMBEDDING_MODEL = "gemini-embedding-001"
_SCORING_CHUTES_EMBEDDING_MODEL = "text-embedding-3-small"


class _ScoringEmbeddingClient(TextEmbeddingPort, Protocol):
    async def aclose(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """Aggregated runtime components for the validator service."""

    settings: Settings
    platform_hotkey: bt.Keypair
    sandbox_manager: DockerSandboxManager
    session_manager: SessionManager
    session_registry: InMemorySessionRegistry
    token_registry: InMemoryTokenRegistry
    receipt_log: InMemoryReceiptLog
    evaluation_records: EvaluationRecordPort
    progress_tracker: InMemoryRunProgress
    usage_tracker: UsageTracker
    search_client: DeSearchClient | None
    tool_llm_provider: LlmProviderPort | None
    scoring_llm_provider: LlmProviderPort | None
    tool_invoker: RuntimeToolInvoker
    tool_executor: ToolExecutor
    token_semaphore: TokenSemaphore
    subtensor_client: SubtensorClientPort
    scoring_embedding_client: _ScoringEmbeddingClient | None
    scoring_service: EvaluationScoringService
    weight_submission_service: WeightSubmissionService
    create_entrypoint_invoker: Callable[[SandboxClient], EntrypointInvoker]
    create_evaluation_orchestrator: Callable[[SandboxClient], TaskRunOrchestrator]
    build_sandbox_options: Callable[[], SandboxOptions]
    platform_client: PlatformPort | None
    batch_inbox: InMemoryBatchInbox
    status_provider: StatusProvider
    tool_route_deps_provider: Callable[[], ToolRouteDeps]
    control_deps_provider: Callable[[], ValidatorControlDeps]

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
    progress_tracker: InMemoryRunProgress
    batch_inbox: InMemoryBatchInbox
    usage_tracker: UsageTracker
    token_semaphore: TokenSemaphore
    session_manager: SessionManager


def build_runtime(settings: Settings | None = None) -> RuntimeContext:
    """Construct the runtime context shared across CLI commands."""
    resolved = settings or Settings.load()
    logger.info("loading validator runtime configuration", extra={"settings": resolved})

    state = _build_state()
    platform_client, platform_hotkey, subtensor_client = _build_external_clients(resolved)

    search_client, tool_llm_provider, scoring_llm_provider = _build_llm_clients(resolved)
    feed_search_provider = HttpFeedSearchToolProvider(
        base_url=str(resolved.platform_api.platform_base_url),
        hotkey=platform_hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    tool_invoker, tool_executor = _build_tooling(
        state=state,
        resolved=resolved,
        search_client=search_client,
        tool_llm_provider=tool_llm_provider,
        feed_search_provider=feed_search_provider,
    )

    scoring_service, weight_submission_service, scoring_embedding_client = _build_services(
        resolved=resolved,
        scoring_llm_provider=scoring_llm_provider,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )

    sandbox_manager = create_sandbox_manager(logger_name="harnyx_validator.sandbox")
    entrypoint_factory, orchestrator_factory, options_factory = _build_factories(
        resolved=resolved,
        state=state,
        scoring_service=scoring_service,
    )
    tool_route_provider, control_provider, status_provider = _build_http_dependencies(
        resolved=resolved,
        state=state,
        tool_executor=tool_executor,
    )

    return RuntimeContext(
        settings=resolved,
        platform_hotkey=platform_hotkey,
        sandbox_manager=sandbox_manager,
        session_manager=state.session_manager,
        session_registry=state.session_registry,
        token_registry=state.token_registry,
        receipt_log=state.receipt_log,
        evaluation_records=state.evaluation_records,
        progress_tracker=state.progress_tracker,
        usage_tracker=state.usage_tracker,
        search_client=search_client,
        tool_llm_provider=tool_llm_provider,
        scoring_llm_provider=scoring_llm_provider,
        tool_invoker=tool_invoker,
        tool_executor=tool_executor,
        token_semaphore=state.token_semaphore,
        subtensor_client=subtensor_client,
        scoring_embedding_client=scoring_embedding_client,
        scoring_service=scoring_service,
        weight_submission_service=weight_submission_service,
        create_entrypoint_invoker=entrypoint_factory,
        create_evaluation_orchestrator=orchestrator_factory,
        build_sandbox_options=options_factory,
        platform_client=platform_client,
        batch_inbox=state.batch_inbox,
        status_provider=status_provider,
        tool_route_deps_provider=tool_route_provider,
        control_deps_provider=control_provider,
    )


def _build_state() -> InMemoryState:
    session_registry = InMemorySessionRegistry()
    token_registry = InMemoryTokenRegistry()
    receipt_log = InMemoryReceiptLog()
    evaluation_records = InMemoryEvaluationRecordStore()
    progress_tracker = InMemoryRunProgress()
    batch_inbox = InMemoryBatchInbox()
    token_semaphore = TokenSemaphore()
    usage_tracker = UsageTracker()
    session_manager = SessionManager(session_registry, token_registry)
    return InMemoryState(
        session_registry=session_registry,
        token_registry=token_registry,
        receipt_log=receipt_log,
        evaluation_records=evaluation_records,
        progress_tracker=progress_tracker,
        batch_inbox=batch_inbox,
        usage_tracker=usage_tracker,
        token_semaphore=token_semaphore,
        session_manager=session_manager,
    )


def _build_external_clients(settings: Settings) -> tuple[PlatformPort, bt.Keypair, SubtensorClientPort]:
    platform_client, platform_hotkey = _create_platform_client(settings)
    subtensor_client = _build_subtensor_client(settings)
    return platform_client, platform_hotkey, subtensor_client


def _build_llm_clients(
    settings: Settings,
) -> tuple[DeSearchClient | None, LlmProviderPort | None, LlmProviderPort | None]:
    search_client = _create_search_client(settings)
    tool_llm_provider = _create_tool_llm_provider(settings)
    scoring_llm_provider = _create_scoring_llm_provider(settings)
    return search_client, tool_llm_provider, scoring_llm_provider


def _build_tooling(
    *,
    state: InMemoryState,
    resolved: Settings,
    search_client: DeSearchClient | None,
    tool_llm_provider: LlmProviderPort | None,
    feed_search_provider: HttpFeedSearchToolProvider | None,
) -> tuple[RuntimeToolInvoker, ToolExecutor]:
    tool_invoker = build_miner_sandbox_tool_invoker(
        state.receipt_log,
        search_client=search_client,
        llm_provider=tool_llm_provider,
        llm_provider_name=resolved.llm.tool_llm_provider,
        feed_search_provider=feed_search_provider,
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    tool_executor = ToolExecutor(
        session_registry=state.session_registry,
        receipt_log=state.receipt_log,
        usage_tracker=state.usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=state.token_registry,
        clock=_clock,
    )
    return tool_invoker, tool_executor


def _build_services(
    *,
    resolved: Settings,
    scoring_llm_provider: LlmProviderPort | None,
    subtensor_client: SubtensorClientPort,
    platform_client: PlatformPort,
) -> tuple[EvaluationScoringService, WeightSubmissionService, _ScoringEmbeddingClient]:
    scoring_embedding_client = _create_scoring_embedding_client(resolved)
    scoring_service = _create_scoring_service(
        resolved,
        scoring_llm_provider,
        embedding_client=scoring_embedding_client,
    )
    weight_submission_service = _build_weight_service(
        resolved,
        subtensor_client=subtensor_client,
        platform_client=platform_client,
    )
    return scoring_service, weight_submission_service, scoring_embedding_client


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
) -> tuple[Callable[[], ToolRouteDeps], Callable[[], ValidatorControlDeps], StatusProvider]:
    status_provider = StatusProvider()
    inbound_auth = _build_inbound_auth(resolved)
    tool_route_provider = _make_dependency_provider(tool_executor, state.token_semaphore)
    accept_batch = AcceptEvaluationBatch(state.batch_inbox, status_provider, state.progress_tracker)
    control_provider = _make_control_provider(
        accept_batch,
        status_provider,
        inbound_auth,
        state.progress_tracker,
    )
    return tool_route_provider, control_provider, status_provider


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
    client = PlatformRegistrationClient(
        platform_base_url=base.rstrip("/"),
        hotkey=hotkey,
        timeout_seconds=PLATFORM.timeout_seconds,
    )
    register_with_retry(client, public_url.rstrip("/"), attempts=30)

def _build_subtensor_client(resolved: Settings) -> SubtensorClientPort:
    client = RuntimeSubtensorClient(resolved.subtensor)
    try:
        client.connect()
    except Exception as exc:
        logger.warning("subtensor client initialization failed", exc_info=exc)
    return client


def _make_dependency_provider(
    tool_executor: ToolExecutor,
    token_semaphore: TokenSemaphore,
) -> Callable[[], ToolRouteDeps]:
    def provider() -> ToolRouteDeps:
        return ToolRouteDeps(
            tool_executor=tool_executor,
            token_semaphore=token_semaphore,
        )

    return provider


def _make_control_provider(
    accept_batch: AcceptEvaluationBatch,
    status_provider: StatusProvider,
    inbound_auth: BittensorSr25519InboundVerifier,
    progress_tracker: InMemoryRunProgress,
) -> Callable[[], ValidatorControlDeps]:
    def provider() -> ValidatorControlDeps:
        return ValidatorControlDeps(
            accept_batch=accept_batch,
            status_provider=status_provider,
            auth=lambda request, body: _verify_request(inbound_auth, request, body),
            progress_tracker=progress_tracker,
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


def _build_inbound_auth(resolved: Settings) -> BittensorSr25519InboundVerifier:
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
    )


def _create_search_client(settings: Settings) -> DeSearchClient:
    return DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.llm.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
        max_concurrent=settings.llm.desearch_max_concurrent,
    )


def _create_tool_llm_provider(settings: Settings) -> LlmProviderPort | None:
    resolve_provider = create_llm_provider_factory(
        chutes_api_key=settings.llm.chutes_api_key_value,
        chutes_base_url=CHUTES.base_url,
        chutes_timeout=CHUTES.timeout_seconds,
        gcp_project_id=settings.vertex.gcp_project_id,
        gcp_location=settings.vertex.gcp_location,
        vertex_maas_gcp_location=settings.vertex.vertex_maas_gcp_location,
        vertex_timeout=settings.vertex.vertex_timeout_seconds,
        gcp_service_account_b64=settings.vertex.gcp_sa_credential_b64_value,
    )
    return resolve_provider(settings.llm.tool_llm_provider)


def _create_scoring_llm_provider(settings: Settings) -> LlmProviderPort | None:
    resolve_provider = create_llm_provider_factory(
        chutes_api_key=settings.llm.chutes_api_key_value,
        chutes_base_url=CHUTES.base_url,
        chutes_timeout=CHUTES.timeout_seconds,
        gcp_project_id=settings.vertex.gcp_project_id,
        gcp_location=settings.vertex.gcp_location,
        vertex_maas_gcp_location=settings.vertex.vertex_maas_gcp_location,
        vertex_timeout=settings.vertex.vertex_timeout_seconds,
        gcp_service_account_b64=settings.vertex.gcp_sa_credential_b64_value,
    )
    return resolve_provider(settings.llm.scoring_llm_provider)


def _create_scoring_service(
    settings: Settings,
    provider: LlmProviderPort | None,
    *,
    embedding_client: TextEmbeddingPort | None = None,
) -> EvaluationScoringService:
    if provider is None:
        raise ValueError("scoring_llm_provider must be configured")
    resolved_embedding_client = embedding_client or _create_scoring_embedding_client(settings)
    config = EvaluationScoringConfig(
        provider=settings.llm.scoring_llm_provider,
        model=settings.llm.scoring_llm_model,
        temperature=settings.llm.scoring_llm_temperature,
        max_output_tokens=settings.llm.scoring_llm_max_output_tokens,
        reasoning_effort=settings.llm.scoring_llm_reasoning_effort,
        timeout_seconds=settings.llm.scoring_llm_timeout_seconds,
    )
    return EvaluationScoringService(
        llm_provider=provider,
        embedding_client=resolved_embedding_client,
        config=config,
    )


def _create_scoring_embedding_client(settings: Settings) -> _ScoringEmbeddingClient:
    return create_scoring_embedding_client(
        provider_name=settings.llm.scoring_llm_provider,
        vertex_model=_SCORING_EMBEDDING_MODEL,
        chutes_model=_SCORING_CHUTES_EMBEDDING_MODEL,
        chutes_api_key=settings.llm.chutes_api_key_value,
        scoring_timeout_seconds=settings.llm.scoring_llm_timeout_seconds,
        vertex_project=settings.vertex.gcp_project_id,
        vertex_location=settings.vertex.gcp_location,
        vertex_maas_location=settings.vertex.vertex_maas_gcp_location,
        vertex_service_account_b64=settings.vertex.gcp_sa_credential_b64_value,
        vertex_timeout_seconds=settings.vertex.vertex_timeout_seconds,
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

    await _aclose(runtime.search_client)
    await _aclose(runtime.tool_llm_provider)
    await _aclose(runtime.scoring_llm_provider)
    await _aclose(runtime.scoring_embedding_client)


def _verify_request(
    verifier: BittensorSr25519InboundVerifier,
    request: Request,
    body: bytes,
) -> str:
    path = request.url.path or "/"
    query = request.url.query
    path_qs = f"{path}?{query}" if query else path
    authorization_header = request.headers.get("authorization")
    return verifier.verify(
        method=request.method,
        path_qs=path_qs,
        body=body or b"",
        authorization_header=authorization_header,
    )


def _clock() -> datetime:
    return datetime.now(UTC)


__all__ = ["RuntimeContext", "build_runtime", "close_runtime_resources"]


class _SupportsAclose(Protocol):
    async def aclose(self) -> None:
        ...
