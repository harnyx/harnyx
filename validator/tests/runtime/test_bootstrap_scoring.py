from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.sandbox import SandboxSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCallOutcome
from harnyx_commons.errors import BudgetExceededError, ConcurrencyLimitError, ToolProviderError
from harnyx_commons.llm.routing import ResolvedLlmRoute, RoutedLlmProvider
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.types import ToolName
from harnyx_validator.application.ports.platform import PlatformToolProxyGrant
from harnyx_validator.infrastructure.tools.platform_client import (
    PlatformToolProxyBudgetExceededError,
    PlatformToolProxyInvocationError,
    PlatformToolProxyProviderError,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import (
    _build_llm_clients,
    _build_proxy_tooling,
    _create_scoring_service,
    _create_similarity_judge,
    close_runtime_resources,
)
from harnyx_validator.runtime.settings import Settings

DEFAULT_LIMIT_LLM_MODEL = "openai/gpt-oss-20b"
TEST_SESSION_TOKEN = "validator-session-token"  # noqa: S105
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token


def _routed_surface(provider: object) -> str:
    assert isinstance(provider, RoutedLlmProvider)
    return provider._surface


class _ProviderFailingPlatformToolProxyClient:
    def __init__(self) -> None:
        self.grants = 0
        self.calls: list[dict[str, object]] = []

    async def create_platform_tool_proxy_grant(
        self,
        *,
        batch_id,
        artifact_id,
        task_id,
        validator_session_id,
        attempt_number,
        assignment_token,
    ):  # type: ignore[no-untyped-def]
        _ = batch_id, artifact_id, task_id, validator_session_id, attempt_number, assignment_token
        self.grants += 1
        return PlatformToolProxyGrant(
            token="platform-tool-proxy-token",  # noqa: S106 - fixed test-only proxy token
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    async def execute_platform_tool_proxy_tool(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)
        raise PlatformToolProxyProviderError(status_code=400, message="provider rejected")


class _ProxyPolicyFailingPlatformToolProxyClient(_ProviderFailingPlatformToolProxyClient):
    async def execute_platform_tool_proxy_tool(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)
        raise PlatformToolProxyInvocationError(
            status_code=400,
            error_code="miner_credential_missing",
            message="miner provider credential was not found",
        )


class _BudgetFailingPlatformToolProxyClient(_ProviderFailingPlatformToolProxyClient):
    async def execute_platform_tool_proxy_tool(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)
        raise PlatformToolProxyBudgetExceededError(status_code=400, message="platform tool proxy budget exhausted")


def _settings_for_tooling(search_provider: str = "desearch") -> Settings:
    return Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=search_provider,
            tool_llm_provider="chutes",
        )
    )


def _register_proxy_session(state: bootstrap.InMemoryState, *, batch_id, session_id, artifact_id, task_id) -> None:
    issued_at = datetime.now(UTC)
    session = Session(
        session_id=session_id,
        uid=7,
        task_id=task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
        budget_usd=1.0,
        usage=SessionUsage(),
        status=SessionStatus.ACTIVE,
    )
    state.session_registry.create(session)
    state.token_registry.register(session_id, TEST_SESSION_TOKEN)
    state.progress_tracker.register_task_session(batch_id=batch_id, session_id=session_id)
    state.platform_tool_proxy_scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )


def test_llm_settings_default_scoring_timeout_is_300_seconds() -> None:
    assert LlmSettings(_env_file=None).scoring_llm_timeout_seconds == pytest.approx(300.0)


def test_local_provider_tooling_forwards_separate_search_role_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    captured: dict[str, object] = {}
    web_client = object()
    ai_client = object()
    web_resolver = object()
    ai_resolver = object()

    def build_invoker(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bootstrap, "build_miner_sandbox_tool_invoker", build_invoker)

    bootstrap._build_local_provider_tooling(
        state=state,
        resolved=_settings_for_tooling(),
        search_client=web_client,  # type: ignore[arg-type]
        ai_search_client=ai_client,  # type: ignore[arg-type]
        tool_llm_provider=None,
        web_search_provider_resolver=web_resolver,  # type: ignore[arg-type]
        ai_search_provider_resolver=ai_resolver,  # type: ignore[arg-type]
    )

    assert captured["web_search_client"] is web_client
    assert captured["ai_search_client"] is ai_client
    assert captured["web_search_provider_resolver"] is web_resolver
    assert captured["ai_search_provider_resolver"] is ai_resolver


def test_build_llm_clients_uses_shared_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
            tool_llm_provider="bedrock",
            scoring_llm_provider="vertex",
            similarity_llm_provider="chutes",
            llm_model_provider_overrides_json=json.dumps({"tool": {"unused-tool-model": "bedrock"}}),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )
    calls: list[str] = []

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            calls.append(name)
            return f"provider:{name}"

    def fake_build_cached_llm_provider_registry(*, llm_settings, bedrock_settings, vertex_settings):
        assert llm_settings is settings.llm
        assert bedrock_settings is settings.bedrock
        assert vertex_settings is settings.vertex
        return _FakeRegistry()

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", fake_build_cached_llm_provider_registry)

    clients = _build_llm_clients(settings)

    assert clients.search_client is None
    assert clients.tool_llm_provider is None
    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert _routed_surface(clients.similarity_llm_provider) == "duplication_detection"
    assert type(clients.llm_provider_registry).__name__ == "_FakeRegistry"
    assert clients.scoring_routes == {
        entry.model: ResolvedLlmRoute(surface="scoring", provider="vertex", model=entry.model)
        for entry in bootstrap._SCORING_SLOT_CONFIG.entries
    }
    assert clients.similarity_route == ResolvedLlmRoute(
        surface="duplication_detection",
        provider="chutes",
        model=bootstrap._DUPLICATION_DETECTION_LLM_MODEL,
    )
    assert clients.similarity_request_extra_by_model == {}
    assert calls == []


def test_validator_runtime_llm_clients_do_not_build_local_tool_invocation_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
            tool_llm_provider="chutes",
            scoring_llm_provider="chutes",
            chutes_api_key=SecretStr("test-key"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    def fail_if_called(**_: object) -> object:
        raise AssertionError("validator runtime must not build local tool invocation clients")

    monkeypatch.setattr(bootstrap, "build_tool_invocation_clients", fail_if_called, raising=False)

    clients = _build_llm_clients(settings)

    assert clients.search_client is None
    assert clients.tool_llm_provider is None


def test_build_llm_clients_resolves_route_for_each_scoring_slot_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring_routes = {
        "google/gemma-4-31B-turbo-TEE": "custom-openai-compatible:gemma4-cloud-run-turbo",
        "Qwen/Qwen3.6-27B-TEE": "custom-openai-compatible:qwen36-cloud-run",
    }
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            llm_model_provider_overrides_json=json.dumps({"scoring": scoring_routes}),
            openai_compatible_endpoints_json=json.dumps(
                [
                    {
                        "id": "gemma4-cloud-run-turbo",
                        "base_url": "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app/v1",
                        "auth": {"type": "none"},
                    },
                    {
                        "id": "qwen36-cloud-run",
                        "base_url": "https://qwen3-6-27b-obbrpx3ppa-uc.a.run.app/v1",
                        "auth": {"type": "none"},
                    },
                ]
            ),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert clients.scoring_routes == {
        model: ResolvedLlmRoute(surface="scoring", provider=provider, model=model)
        for model, provider in scoring_routes.items()
    }


def test_build_state_uses_single_tool_concurrency_cap(tmp_path: Path) -> None:
    state = bootstrap._build_state(Settings(), progress_storage_root=tmp_path / "run-progress")
    session_id = uuid4()
    token = "token"  # noqa: S105
    tools: tuple[ToolName, ...] = ("search_web", "fetch_page", "tooling_info", "test_tool", "llm_chat")
    held = [
        ToolInvocationRequest(
            session_id=session_id,
            token=token,
            tool=tools[index % len(tools)],
            kwargs={"model": f"model-{index}"} if tools[index % len(tools)] == "llm_chat" else {},
        )
        for index in range(20)
    ]

    for invocation in held:
        state.tool_concurrency_limiter.acquire(invocation)
    with pytest.raises(ConcurrencyLimitError):
        state.tool_concurrency_limiter.acquire(
            ToolInvocationRequest(session_id=session_id, token=token, tool="search_web")
        )

    for invocation in held:
        state.tool_concurrency_limiter.release(invocation)

    next_invocation = ToolInvocationRequest(session_id=session_id, token=token, tool="search_web")
    state.tool_concurrency_limiter.acquire(next_invocation)
    state.tool_concurrency_limiter.release(next_invocation)


def test_build_proxy_tooling_uses_plain_executor_with_platform_tool_proxy_client(tmp_path: Path) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")

    tool_invoker, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=_ProviderFailingPlatformToolProxyClient(),
    )

    assert isinstance(tool_invoker, bootstrap.PlatformToolProxyProxyToolInvoker)
    assert isinstance(tool_executor, bootstrap.ToolExecutor)
    assert not isinstance(tool_executor, bootstrap._ProviderTrackingToolExecutor)


@pytest.mark.anyio("asyncio")
async def test_proxy_enabled_tool_executor_keeps_provider_failure_miner_owned_without_provider_evidence(
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _ProviderFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
    )

    with pytest.raises(ToolProviderError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.PROVIDER_ERROR
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "provider_failed"
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()


@pytest.mark.anyio("asyncio")
async def test_proxy_enabled_tool_executor_keeps_llm_provider_failure_miner_owned_without_provider_evidence(
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _ProviderFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="llm_chat",
        kwargs={
            "provider": "openrouter",
            "model": DEFAULT_LIMIT_LLM_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    with pytest.raises(ToolProviderError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.PROVIDER_ERROR
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "provider_failed"
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("invalid_provider", [123, "Parallel", " Parallel "])
async def test_proxy_enabled_tool_executor_does_not_default_attribute_invalid_explicit_provider(
    tmp_path: Path,
    invalid_provider: object,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _ProviderFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="search_web",
        kwargs={"provider": invalid_provider, "search_queries": ["harnyx"]},
    )

    with pytest.raises(ToolProviderError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "provider_failed"


@pytest.mark.anyio("asyncio")
async def test_proxy_enabled_tool_executor_does_not_record_provider_failure_for_proxy_policy_error(
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _ProxyPolicyFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
    )

    with pytest.raises(PlatformToolProxyInvocationError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "miner_credential_missing"


@pytest.mark.anyio("asyncio")
async def test_proxy_enabled_tool_executor_records_budget_exceeded_receipt_without_provider_evidence(
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling(), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _BudgetFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="search_web",
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
    )

    with pytest.raises(BudgetExceededError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.BUDGET_EXCEEDED
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "budget_exhausted"
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()


@pytest.mark.anyio("asyncio")
async def test_proxy_enabled_tool_executor_keeps_no_provider_payload_failure_miner_owned_without_provider_evidence(
    tmp_path: Path,
) -> None:
    state = bootstrap._build_state(_settings_for_tooling("desearch"), progress_storage_root=tmp_path / "run-progress")
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    _register_proxy_session(
        state,
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
    )
    platform = _ProviderFailingPlatformToolProxyClient()
    _, tool_executor = _build_proxy_tooling(
        state=state,
        platform_tool_proxy_platform_client=platform,
    )
    request = ToolInvocationRequest(
        session_id=session_id,
        token=TEST_SESSION_TOKEN,
        tool="search_web",
        kwargs={"search_queries": ["harnyx"]},
    )

    with pytest.raises(ToolProviderError):
        await tool_executor.execute(request)

    assert len(platform.calls) == 1
    receipts = tuple(state.receipt_log.for_session(session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.PROVIDER_ERROR
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["platform_tool_proxy_error_code"] == "provider_failed"
    assert state.progress_tracker.consume_provider_failures(session_id) == ()
    assert state.progress_tracker.provider_evidence(batch_id) == ()


def test_build_state_prunes_stale_run_progress_dirs(tmp_path: Path) -> None:
    run_progress_root = tmp_path / "run-progress"
    stale_batch_id = uuid4()
    stale_dir = run_progress_root / str(stale_batch_id)
    stale_dir.mkdir(parents=True)
    stale_blob = stale_dir / "runs-000001.blob"
    stale_blob.write_bytes(b"stale")
    os.utime(stale_dir, (0, 0))
    os.utime(stale_blob, (0, 0))
    settings = Settings.model_construct(run_progress_retention_seconds=3600)

    state = bootstrap._build_state(settings, progress_storage_root=run_progress_root)

    assert state.progress_tracker.storage_root == run_progress_root
    assert not stale_dir.exists()


def test_build_runtime_cleans_stale_sandbox_containers_on_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSandboxManager:
        def __init__(self) -> None:
            self.cleanup_calls: list[dict[str, object]] = []

        def cleanup_stale_sandbox_containers(self, *, labels: dict[str, str], name_prefix: str) -> None:
            self.cleanup_calls.append({"labels": dict(labels), "name_prefix": name_prefix})

    manager = FakeSandboxManager()

    monkeypatch.setattr(
        bootstrap, "_build_external_clients", lambda _settings: (object(), object(), object(), object())
    )
    monkeypatch.setattr(
        bootstrap,
        "_build_llm_clients",
        lambda _settings: bootstrap.RuntimeLlmClients(
            search_client=None,
            llm_provider_registry=object(),
            tool_llm_provider=None,
            tool_embedding_provider=None,
            scoring_llm_provider=None,
            similarity_llm_provider=None,
            scoring_routes=object(),
            similarity_route=object(),
            similarity_request_extra_by_model={},
        ),
    )
    monkeypatch.setattr(bootstrap, "_build_proxy_tooling", lambda **_kwargs: (object(), object()))
    monkeypatch.setattr(
        bootstrap,
        "_build_services",
        lambda **_kwargs: (
            {entry.model: object() for entry in bootstrap._SCORING_SLOT_CONFIG.entries},
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "_build_factories",
        lambda **_kwargs: (lambda _client: object(), lambda _client: object(), lambda: object()),
    )
    monkeypatch.setattr(
        bootstrap,
        "_build_http_dependencies",
        lambda **_kwargs: (lambda: object(), lambda: object(), object(), object()),
    )
    monkeypatch.setattr(bootstrap, "create_sandbox_manager", lambda **_kwargs: manager)
    settings = Settings.model_construct(
        validator_state_dir=tmp_path,
        run_progress_retention_seconds=3600,
    )

    runtime = bootstrap.build_runtime(settings)
    runtime.batch_blocking_executor.shutdown(wait=False, cancel_futures=True)

    assert manager.cleanup_calls == [
        {
            "labels": {"harnyx.sandbox.managed": "true", "harnyx.sandbox.owner": "validator"},
            "name_prefix": "harnyx-sandbox-",
        }
    ]


def test_create_scoring_service_does_not_require_vertex_config_at_bootstrap() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=20480,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._DIRECT_SCORING_LLM_MODEL,
        ),
    )

    assert service is not None
    assert service._config.provider == "chutes"
    assert service._config.model == bootstrap._DIRECT_SCORING_LLM_MODEL
    assert service._config.fallback_models == ()
    assert service._config.reasoning_effort == bootstrap._SCORING_LLM_REASONING_EFFORT
    assert service._config.retry_policy == settings.llm.scoring_llm_retry_policy


def test_create_scoring_service_uses_effective_route_model_and_default_provider() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="vertex",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=20480,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="bedrock",
            model="custom/internal-model",
        ),
    )

    assert service._config.provider == "vertex"
    assert service._config.model == "custom/internal-model"
    assert service._config.fallback_models == ()
    assert service._config.retry_policy == settings.llm.scoring_llm_retry_policy


def test_create_scoring_service_uses_explicit_fallback_models() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=0.0,
            scoring_llm_max_output_tokens=20480,
            scoring_llm_timeout_seconds=30.0,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._DIRECT_SCORING_LLM_MODEL,
        ),
        fallback_models=("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"),
    )

    assert service._config.fallback_models == ("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE")


def test_scoring_slot_config_entries_are_hard_coded() -> None:
    assert tuple(bootstrap._SCORING_SLOT_CONFIG.entries) == (
        bootstrap.ScoringSlotConfigEntry(
            model="google/gemma-4-31B-turbo-TEE",
            slot_limit=10,
            fallback_models=(
                "zai-org/GLM-5.2-TEE",
                "zai-org/GLM-5.1-TEE",
                "moonshotai/Kimi-K3-TEE",
            ),
        ),
        bootstrap.ScoringSlotConfigEntry(
            model="Qwen/Qwen3.6-27B-TEE",
            slot_limit=10,
            fallback_models=(
                "zai-org/GLM-5.2-TEE",
                "zai-org/GLM-5.1-TEE",
                "moonshotai/Kimi-K3-TEE",
            ),
        ),
    )


def test_build_services_passes_slot_fallback_models_to_scoring_services() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=0.0,
            scoring_llm_max_output_tokens=20480,
            scoring_llm_timeout_seconds=30.0,
            similarity_llm_provider="chutes",
            similarity_llm_temperature=0.0,
            similarity_llm_max_output_tokens=4096,
            similarity_llm_timeout_seconds=30.0,
        ),
        subtensor=SubtensorSettings.model_construct(netuid=1),
    )

    scoring_services, _, _ = bootstrap._build_services(
        resolved=settings,
        scoring_llm_provider=SimpleNamespace(),
        similarity_llm_provider=SimpleNamespace(),
        scoring_routes={
            "google/gemma-4-31B-turbo-TEE": ResolvedLlmRoute(
                surface="scoring",
                provider="chutes",
                model="google/gemma-4-31B-turbo-TEE",
            ),
            "Qwen/Qwen3.6-27B-TEE": ResolvedLlmRoute(
                surface="scoring",
                provider="chutes",
                model="Qwen/Qwen3.6-27B-TEE",
            ),
        },
        similarity_route=ResolvedLlmRoute(
            surface="duplication_detection",
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
        ),
        similarity_request_extra_by_model={},
        subtensor_client=SimpleNamespace(),
        platform_client=SimpleNamespace(),
    )

    assert scoring_services["google/gemma-4-31B-turbo-TEE"]._config.fallback_models == (
        "zai-org/GLM-5.2-TEE",
        "zai-org/GLM-5.1-TEE",
        "moonshotai/Kimi-K3-TEE",
    )
    assert scoring_services["Qwen/Qwen3.6-27B-TEE"]._config.fallback_models == (
        "zai-org/GLM-5.2-TEE",
        "zai-org/GLM-5.1-TEE",
        "moonshotai/Kimi-K3-TEE",
    )


def test_direct_scoring_service_uses_explicit_direct_model() -> None:
    direct_service = object()
    other_service = object()

    selected = bootstrap._direct_scoring_service(
        {
            "Qwen/Qwen3.6-27B-TEE": other_service,
            bootstrap._DIRECT_SCORING_LLM_MODEL: direct_service,
        }
    )

    assert selected is direct_service


def test_direct_scoring_service_inherits_configured_fallback_models() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=0.0,
            scoring_llm_max_output_tokens=20480,
            scoring_llm_timeout_seconds=30.0,
        ),
    )
    direct_service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._DIRECT_SCORING_LLM_MODEL,
        ),
        fallback_models=("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"),
    )

    selected = bootstrap._direct_scoring_service(
        {
            "Qwen/Qwen3.6-27B-TEE": object(),
            bootstrap._DIRECT_SCORING_LLM_MODEL: direct_service,
        }
    )

    assert selected._config.fallback_models == ("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE")


@pytest.mark.anyio("asyncio")
async def test_score_platform_execution_executor_converts_scoring_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring_service = object()
    execution = object()
    expected = object()
    observed: dict[str, object] = {}

    async def fake_score_platform_execution(
        service: object,
        scoreable_execution: object,
        *,
        convert_scoring_error: bool,
    ) -> object:
        observed["service"] = service
        observed["execution"] = scoreable_execution
        observed["convert_scoring_error"] = convert_scoring_error
        return expected

    monkeypatch.setattr(bootstrap, "score_platform_execution", fake_score_platform_execution)

    executor = bootstrap._score_platform_execution_with(scoring_service)  # type: ignore[arg-type]
    result = await executor(execution)  # type: ignore[arg-type]

    assert result is expected
    assert observed == {
        "service": scoring_service,
        "execution": execution,
        "convert_scoring_error": True,
    }


def test_create_similarity_judge_uses_similarity_llm_config() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=0.9,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            similarity_llm_provider="vertex",
            similarity_llm_temperature=0.0,
            similarity_llm_max_output_tokens=4096,
            similarity_llm_timeout_seconds=300.0,
            similarity_llm_retry_attempts=2,
            similarity_llm_retry_initial_ms=10,
            similarity_llm_retry_max_ms=20,
            similarity_llm_retry_jitter=0.0,
        ),
    )

    judge = _create_similarity_judge(
        settings,
        provider=SimpleNamespace(),
        similarity_route=ResolvedLlmRoute(
            surface="duplication_detection",
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
        ),
        request_extra_by_model={},
    )

    assert judge._config.provider == "vertex"
    assert judge._config.model == "google/gemma-4-31B-turbo-TEE"
    assert judge._config.fallback_models == (
        "deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
        "zai-org/GLM-5.2-TEE",
        "moonshotai/Kimi-K3-TEE",
    )
    assert judge._config.temperature == 0.0
    assert judge._config.max_output_tokens == 4096
    assert judge._config.reasoning_effort == "high"
    assert judge._config.timeout_seconds == 300.0
    assert judge._config.retry_policy == settings.llm.similarity_llm_retry_policy
    assert judge._config.request_extra_by_model == {}


def test_similarity_fallback_tail_only_uses_candidates_after_primary_override() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_model_override="zai-org/GLM-5.2-TEE",
        ),
    )

    assert bootstrap._similarity_judge_fallback_models(settings) == ("moonshotai/Kimi-K3-TEE",)


def test_similarity_deepseek_override_uses_only_later_candidates() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_model_override="deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
        ),
    )

    assert bootstrap._similarity_judge_fallback_models(settings) == (
        "zai-org/GLM-5.2-TEE",
        "moonshotai/Kimi-K3-TEE",
    )


def test_similarity_model_override_participates_in_duplication_detection_route_override() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider="chutes",
            similarity_llm_model_override="moonshotai/Kimi-K2.5-TEE",
            llm_model_provider_overrides_json=json.dumps(
                {"duplication_detection": {"moonshotai/Kimi-K2.5-TEE": "bedrock"}}
            ),
        ),
    )

    route = bootstrap._resolve_similarity_judge_route(settings)

    assert route == ResolvedLlmRoute(
        surface="duplication_detection",
        provider="bedrock",
        model="moonshotai/Kimi-K2.5-TEE",
    )


@pytest.mark.parametrize("provider", ("bedrock", "vertex"))
def test_default_similarity_chain_rejects_incompatible_builtin_provider(provider: str) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider=provider,
            similarity_llm_model_override=None,
            llm_model_provider_overrides_json=None,
        ),
    )

    with pytest.raises(ValueError, match="google/gemma-4-31B-turbo-TEE.*requires a Chutes or custom"):
        bootstrap._resolve_similarity_judge_routes(settings)


def test_default_similarity_chain_allows_explicit_chutes_routes() -> None:
    models = bootstrap._DUPLICATION_DETECTION_MODEL_CHAIN
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider="vertex",
            similarity_llm_model_override=None,
            llm_model_provider_overrides_json=json.dumps(
                {"duplication_detection": {model: "chutes" for model in models}}
            ),
        ),
    )

    routes = bootstrap._resolve_similarity_judge_routes(settings)

    assert tuple(route.provider for route in routes) == tuple("chutes" for _ in models)


def test_default_similarity_chain_allows_explicit_custom_routes() -> None:
    models = bootstrap._DUPLICATION_DETECTION_MODEL_CHAIN
    route_target = "custom-openai-compatible:wide-context"
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider="vertex",
            similarity_llm_model_override=None,
            llm_model_provider_overrides_json=json.dumps(
                {"duplication_detection": {model: route_target for model in models}}
            ),
            openai_compatible_endpoints_json=json.dumps(
                [
                    {
                        "id": "wide-context",
                        "base_url": "https://wide-context.test/v1",
                        "auth": {"type": "none"},
                    }
                ]
            ),
        ),
    )

    routes = bootstrap._resolve_similarity_judge_routes(settings)

    assert tuple(route.provider for route in routes) == tuple(route_target for _ in models)


def test_default_similarity_chain_routes_only_deepseek_through_openrouter() -> None:
    deepseek_model = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider="chutes",
            similarity_llm_model_override=None,
            llm_model_provider_overrides_json=json.dumps(
                {"duplication_detection": {deepseek_model: "openrouter"}}
            ),
        ),
    )

    routes = bootstrap._resolve_similarity_judge_routes(settings)

    assert tuple((route.model, route.provider) for route in routes) == (
        ("google/gemma-4-31B-turbo-TEE", "chutes"),
        (deepseek_model, "openrouter"),
        ("zai-org/GLM-5.2-TEE", "chutes"),
        ("moonshotai/Kimi-K3-TEE", "chutes"),
    )
    assert bootstrap._similarity_request_extra_by_model(routes) == {
        deepseek_model: {
            "provider": {
                "ignore": [
                    "Baidu",
                    "Cloudflare",
                    "Decart",
                    "Io Net",
                    "OpenInference",
                    "Parasail",
                    "Sail Research",
                    "Together",
                ]
            }
        }
    }


def test_default_similarity_chain_rejects_openrouter_for_non_deepseek_model() -> None:
    glm_model = "zai-org/GLM-5.2-TEE"
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            similarity_llm_provider="chutes",
            similarity_llm_model_override=None,
            llm_model_provider_overrides_json=json.dumps(
                {"duplication_detection": {glm_model: "openrouter"}}
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate-preflight similarity OpenRouter route.*DeepSeek-V4-Flash-0731",
    ):
        bootstrap._resolve_similarity_judge_routes(settings)


def test_build_llm_clients_routes_configured_scoring_entries_to_custom_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring_routes = {
        "google/gemma-4-31B-turbo-TEE": "custom-openai-compatible:gemma4-cloud-run-turbo",
        "Qwen/Qwen3.6-27B-TEE": "custom-openai-compatible:qwen36-cloud-run",
    }
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            llm_model_provider_overrides_json=json.dumps(
                {"scoring": scoring_routes}
            ),
            openai_compatible_endpoints_json=json.dumps(
                [
                    {
                        "id": "gemma4-cloud-run-turbo",
                        "base_url": "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app/v1",
                        "auth": {"type": "none"},
                    },
                    {
                        "id": "qwen36-cloud-run",
                        "base_url": "https://qwen3-6-27b-obbrpx3ppa-uc.a.run.app/v1",
                        "auth": {"type": "none"},
                    }
                ]
            ),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    clients = _build_llm_clients(settings)

    assert _routed_surface(clients.scoring_llm_provider) == "scoring"
    assert clients.scoring_routes == {
        model: ResolvedLlmRoute(surface="scoring", provider=provider, model=model)
        for model, provider in scoring_routes.items()
    }


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _CountingClosable:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _ShutdownSpyExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.calls.append((wait, cancel_futures))


@pytest.mark.anyio
async def test_close_runtime_resources_closes_llm_provider_registry() -> None:
    llm_provider_registry = _Closable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        llm_provider_registry=llm_provider_registry,
        platform_tool_proxy_platform_client=None,
        tool_llm_provider=None,
        scoring_llm_provider=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert llm_provider_registry.closed is True


@pytest.mark.anyio
async def test_close_runtime_resources_closes_registry_once() -> None:
    llm_provider_registry = _CountingClosable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        llm_provider_registry=llm_provider_registry,
        platform_tool_proxy_platform_client=None,
        tool_llm_provider=None,
        scoring_llm_provider=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert llm_provider_registry.close_calls == 1


@pytest.mark.anyio
async def test_close_runtime_resources_closes_platform_tool_proxy_client_once() -> None:
    closable = _CountingClosable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        llm_provider_registry=closable,
        platform_tool_proxy_platform_client=closable,
        tool_llm_provider=None,
        scoring_llm_provider=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert closable.close_calls == 1
