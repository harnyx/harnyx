from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import SecretStr

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCallOutcome
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.openrouter import OpenRouterLlmProvider
from harnyx_commons.llm.routing import ResolvedLlmRoute
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_commons.tools.token_semaphore import (
    DEFAULT_TOOL_CONCURRENCY_LIMITS,
    ToolConcurrencyLimiter,
    ToolConcurrencyLimits,
)
from harnyx_commons.tools.types import ToolName
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.infrastructure.http.routes import ToolRouteDeps, add_tool_routes
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.runtime.bootstrap import ALLOWED_TOOL_MODELS, _ProviderTrackingToolExecutor
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

DEMO_SESSION_TOKEN = uuid4().hex


def _invocation(tool: ToolName = "search_web") -> ToolInvocationRequest:
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=DEMO_SESSION_TOKEN,
        tool=tool,
        args=(),
        kwargs={},
    )


def create_test_app(dependency_provider: DemoDependencyProvider) -> FastAPI:
    """Create a test app with tool routes only."""
    app = FastAPI()
    add_tool_routes(app, dependency_provider)
    return app


class RecordingToolInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((tool_name, args, kwargs))
        query = kwargs.get("query", "demo")
        return {
            "data": [
                {
                    "link": f"https://example.com/{query}",
                    "title": "Demo",
                    "snippet": "demo",
                },
            ],
        }


class RecordingToolConcurrencyLimiter(ToolConcurrencyLimiter):
    def __init__(self, limits: ToolConcurrencyLimits = DEFAULT_TOOL_CONCURRENCY_LIMITS) -> None:
        super().__init__(limits)
        self.acquire_calls: list[tuple[str, ToolName]] = []
        self.release_calls: list[tuple[str, ToolName]] = []

    def acquire(self, invocation: ToolInvocationRequest) -> None:
        self.acquire_calls.append((invocation.token, invocation.tool))
        super().acquire(invocation)

    async def acquire_async(self, invocation: ToolInvocationRequest) -> None:
        self.acquire_calls.append((invocation.token, invocation.tool))
        await super().acquire_async(invocation)

    def release(self, invocation: ToolInvocationRequest) -> None:
        self.release_calls.append((invocation.token, invocation.tool))
        super().release(invocation)


class DemoDependencyProvider:
    def __init__(self) -> None:
        self.session_registry = FakeSessionRegistry()
        self.receipt_log = FakeReceiptLog()
        self.tokens = InMemoryTokenRegistry()

        self.session = Session(
            session_id=uuid4(),
            uid=7,
            task_id=uuid4(),
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
            budget_usd=0.1,
            usage=SessionUsage(),
            status=SessionStatus.ACTIVE,
        )
        self.session_registry.create(self.session)
        self.tokens.register(self.session.session_id, DEMO_SESSION_TOKEN)

        usage_tracker = UsageTracker()
        tool_invoker = RecordingToolInvoker()

        self.tool_executor = ToolExecutor(
            session_registry=self.session_registry,
            receipt_log=self.receipt_log,
            usage_tracker=usage_tracker,
            tool_invoker=tool_invoker,
            token_registry=self.tokens,
            clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
        )
        self.invoker = tool_invoker
        self.tool_concurrency_limiter = RecordingToolConcurrencyLimiter()

        self.dependencies = ToolRouteDeps(
            tool_executor=self.tool_executor,
            tool_concurrency_limiter=self.tool_concurrency_limiter,
        )

    def __call__(self) -> ToolRouteDeps:
        return self.dependencies


class _NoopLlmProvider:
    async def invoke(self, request):
        raise AssertionError(f"llm provider should not be called: {request}")


class _SuccessfulLlmProvider:
    async def invoke(self, request):
        return LlmResponse(
            id="resp-success",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class _RetryExhaustedLlmProvider:
    async def invoke(self, request):
        raise LlmRetryExhaustedError("provider timed out")


class _SlowLlmProvider(_SuccessfulLlmProvider):
    async def invoke(self, request):
        await asyncio.sleep(1.0)
        return await super().invoke(request)


class _ProviderTimeoutLlmProvider:
    async def invoke(self, request):
        raise TimeoutError("provider timed out")


class _SlowFetchPageProvider:
    async def fetch_page(self, request: FetchPageRequest) -> FetchPageResponse:
        await asyncio.sleep(1.0)
        return FetchPageResponse(data=[{"url": request.url, "content": "page text"}])


class _SlowSearchProvider(_SlowFetchPageProvider):
    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        await asyncio.sleep(1.0)
        return SearchWebSearchResponse(data=[])

    async def search_ai(self, request: SearchAiSearchRequest) -> SearchAiSearchResponse:
        await asyncio.sleep(1.0)
        return SearchAiSearchResponse(data=[])


class _ProviderTimeoutSearchProvider:
    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse:
        raise TimeoutError("provider timed out")

    async def search_ai(self, request: SearchAiSearchRequest) -> SearchAiSearchResponse:
        raise TimeoutError("provider timed out")

    async def fetch_page(self, request: FetchPageRequest) -> FetchPageResponse:
        raise TimeoutError("provider timed out")


class TrackingDependencyProvider:
    def __init__(self, *, llm_provider=None, llm_provider_name: str = "openai", web_search_client=None) -> None:
        self.session_registry = FakeSessionRegistry()
        self.receipt_log = FakeReceiptLog()
        self.tokens = InMemoryTokenRegistry()
        self._progress_storage = tempfile.TemporaryDirectory(prefix="harnyx-test-run-progress-")
        self.progress_tracker = FileBackedRunProgress(
            storage_root=Path(self._progress_storage.name) / "run-progress"
        )
        self.batch_id = uuid4()
        self.artifact_id = uuid4()

        self.session = Session(
            session_id=uuid4(),
            uid=7,
            task_id=uuid4(),
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
            budget_usd=0.1,
            usage=SessionUsage(),
            status=SessionStatus.ACTIVE,
        )
        self.session_registry.create(self.session)
        self.tokens.register(self.session.session_id, DEMO_SESSION_TOKEN)
        self.progress_tracker.register_task_session(
            batch_id=self.batch_id,
            session_id=self.session.session_id,
        )

        usage_tracker = UsageTracker()
        tool_invoker = RuntimeToolInvoker(
            FakeReceiptLog(),
            web_search_client=web_search_client,
            llm_provider=llm_provider or _NoopLlmProvider(),
            llm_provider_name="openai",
            allowed_models=ALLOWED_TOOL_MODELS,
        )

        self.tool_executor = _ProviderTrackingToolExecutor(
            session_registry=self.session_registry,
            receipt_log=self.receipt_log,
            usage_tracker=usage_tracker,
            tool_invoker=tool_invoker,
            token_registry=self.tokens,
            clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            progress=self.progress_tracker,
            search_provider_name="desearch",
            llm_route_resolver=lambda model: ResolvedLlmRoute(
                surface="tool",
                provider=llm_provider_name,
                model=model,
            ),
        )
        self.tool_concurrency_limiter = RecordingToolConcurrencyLimiter()
        self.dependencies = ToolRouteDeps(
            tool_executor=self.tool_executor,
            tool_concurrency_limiter=self.tool_concurrency_limiter,
        )

    def __call__(self) -> ToolRouteDeps:
        return self.dependencies


def test_execute_tool_endpoint_records_receipt() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 200
    body = response.json()
    receipt_id = body["receipt_id"]
    receipt = provider.receipt_log.lookup(receipt_id)
    assert receipt is not None
    assert body["results"][0]["result_id"] == receipt.details.results[0].result_id
    assert body["result_policy"] == receipt.details.result_policy.value
    assert receipt.details.request_hash
    session_snapshot = provider.session_registry.get(provider.session.session_id)
    assert session_snapshot is not None
    assert session_snapshot.usage.total_cost_usd == 0.0001
    assert provider.tool_concurrency_limiter.acquire_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.release_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.in_flight(_invocation("search_web")) == 0


def test_execute_tool_endpoint_accepts_neutral_headers() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 200
    assert provider.tool_concurrency_limiter.acquire_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.release_calls == [(DEMO_SESSION_TOKEN, "search_web")]


def test_execute_tool_endpoint_releases_semaphore_on_failure() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_FailingToolExecutor(),
        tool_concurrency_limiter=provider.tool_concurrency_limiter,
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert provider.tool_concurrency_limiter.release_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.in_flight(_invocation("search_web")) == 0


def test_execute_tool_endpoint_returns_generic_detail_for_provider_failure() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_ProviderFailingToolExecutor(),
        tool_concurrency_limiter=provider.tool_concurrency_limiter,
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}


def test_execute_tool_endpoint_waits_for_same_token_permit_then_succeeds() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    response_box: dict[str, Response | Exception] = {}
    done = threading.Event()
    held = [
        _invocation("search_web"),
        _invocation("search_ai"),
        _invocation("fetch_page"),
        _invocation("tooling_info"),
        _invocation("test_tool"),
    ]

    def issue_request() -> None:
        try:
            with TestClient(app) as client:
                response_box["response"] = client.post(
                    "/v1/tools/execute",
                    json={
                        "tool": "search_web",
                        "args": ["demo"],
                        "kwargs": {"query": "demo"},
                    },
                    headers={
                        "x-platform-token": DEMO_SESSION_TOKEN,
                        SESSION_ID_HEADER: str(provider.session.session_id),
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive capture
            response_box["error"] = exc
        finally:
            done.set()

    for invocation in held:
        provider.tool_concurrency_limiter.acquire(invocation)
    request_thread = threading.Thread(target=issue_request)
    request_thread.start()
    try:
        deadline = time.monotonic() + 1.0
        while len(provider.tool_concurrency_limiter.acquire_calls) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(provider.tool_concurrency_limiter.acquire_calls) == 6
        assert not done.is_set()
        provider.tool_concurrency_limiter.release(held.pop())
    finally:
        for invocation in held:
            provider.tool_concurrency_limiter.release(invocation)
    request_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert "error" not in response_box
    response = response_box["response"]
    assert isinstance(response, Response)
    assert response.status_code == 200
    assert provider.tool_concurrency_limiter.in_flight(_invocation("search_web")) == 0


def test_execute_tool_endpoint_invalid_llm_payload_does_not_record_provider_call() -> None:
    provider = TrackingDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "not-an-allowed-model",
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == ()


def test_execute_tool_endpoint_rejects_non_string_llm_model_without_provider_call() -> None:
    provider = TrackingDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": 123,
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == ()


def test_execute_tool_endpoint_records_openrouter_missing_key_as_failed_receipt() -> None:
    provider = TrackingDependencyProvider(
        llm_provider=OpenRouterLlmProvider(
            openrouter_api_key=SecretStr(""),
            model_provider_options={},
        ),
        llm_provider_name="openrouter",
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "openai/gpt-oss-120b",
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}
    receipts = tuple(provider.receipt_log.for_session(provider.session.session_id))
    assert len(receipts) == 1
    assert receipts[0].outcome is ToolCallOutcome.PROVIDER_ERROR
    assert receipts[0].details.extra is not None
    assert receipts[0].details.extra["error_type"] == "ToolProviderError"
    expected_failure_reason = (
        "OPENROUTER_API_KEY must be configured to use OpenRouter model "
        "openai/gpt-oss-120b"
    )
    assert receipts[0].details.extra["error_message"] == expected_failure_reason
    assert receipts[0].details.extra["error_cause_type"] == "LlmProviderConfigurationError"
    assert receipts[0].details.extra["error_cause_message"] == expected_failure_reason
    expected_provider_failure = {
        "provider": "openrouter",
        "model": "openai/gpt-oss-120b",
        "total_calls": 1,
        "failed_calls": 1,
        "failure_reason": expected_failure_reason,
    }
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == (
        expected_provider_failure,
    )
    assert provider.progress_tracker.consume_provider_failures(provider.session.session_id) == (
        expected_provider_failure,
    )


def test_execute_tool_endpoint_records_provider_call_on_live_llm_success() -> None:
    provider = TrackingDependencyProvider(llm_provider=_SuccessfulLlmProvider())
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 200
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == (
        {
            "provider": "openai",
            "model": ALLOWED_TOOL_MODELS[0],
            "total_calls": 1,
            "failed_calls": 0,
        },
    )


def test_execute_tool_endpoint_records_custom_openai_compatible_provider_call() -> None:
    provider = TrackingDependencyProvider(
        llm_provider=_SuccessfulLlmProvider(),
        llm_provider_name="custom-openai-compatible:gemma4-cloud-run-turbo",
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": "google/gemma-4-31B-turbo-TEE",
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 200
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == (
        {
            "provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
            "model": "google/gemma-4-31B-turbo-TEE",
            "total_calls": 1,
            "failed_calls": 0,
        },
    )


def test_execute_tool_endpoint_records_provider_failure_on_live_llm_provider_error() -> None:
    provider = TrackingDependencyProvider(llm_provider=_RetryExhaustedLlmProvider())
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == (
        {
            "provider": "openai",
            "model": ALLOWED_TOOL_MODELS[0],
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "provider timed out",
        },
    )


def test_execute_tool_endpoint_records_provider_failure_reason_from_cause() -> None:
    provider = TrackingDependencyProvider(llm_provider=_RetryExhaustedLlmProvider())
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "llm_chat",
            "args": [],
            "kwargs": {
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
            },
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert provider.progress_tracker.consume_provider_failures(provider.session.session_id) == (
        {
            "provider": "openai",
            "model": ALLOWED_TOOL_MODELS[0],
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "provider timed out",
        },
    )


def test_execute_tool_endpoint_does_not_record_provider_failure_for_fetch_page_timeout() -> None:
    provider = TrackingDependencyProvider(web_search_client=_SlowFetchPageProvider())
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "fetch_page",
            "args": [],
            "kwargs": {"url": "https://example.com", "timeout": 0.01},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == ()
    assert provider.progress_tracker.consume_provider_failures(provider.session.session_id) == ()


@pytest.mark.parametrize(
    ("tool", "kwargs"),
    [
        ("search_web", {"search_queries": ["harnyx"], "timeout": 0.01}),
        ("search_ai", {"prompt": "harnyx", "count": 10, "timeout": 0.01}),
        (
            "llm_chat",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
                "timeout": 0.01,
            },
        ),
    ],
)
def test_execute_tool_endpoint_does_not_record_provider_failure_for_tool_timeout(
    tool: ToolName,
    kwargs: dict[str, object],
) -> None:
    provider = TrackingDependencyProvider(
        web_search_client=_SlowSearchProvider(),
        llm_provider=_SlowLlmProvider(),
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": tool,
            "args": [],
            "kwargs": kwargs,
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == ()
    assert provider.progress_tracker.consume_provider_failures(provider.session.session_id) == ()


@pytest.mark.parametrize(
    ("tool", "kwargs", "expected_provider", "expected_model"),
    [
        ("search_web", {"search_queries": ["harnyx"], "timeout": 5}, "desearch", "search_web"),
        ("search_ai", {"prompt": "harnyx", "count": 10, "timeout": 5}, "desearch", "search_ai"),
        ("fetch_page", {"url": "https://example.com", "timeout": 5}, "desearch", "fetch_page"),
        (
            "llm_chat",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model": ALLOWED_TOOL_MODELS[0],
                "timeout": 5,
            },
            "openai",
            ALLOWED_TOOL_MODELS[0],
        ),
    ],
)
def test_execute_tool_endpoint_records_provider_failure_for_provider_timeout(
    tool: ToolName,
    kwargs: dict[str, object],
    expected_provider: str,
    expected_model: str,
) -> None:
    provider = TrackingDependencyProvider(
        web_search_client=_ProviderTimeoutSearchProvider(),
        llm_provider=_ProviderTimeoutLlmProvider(),
    )
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": tool,
            "args": [],
            "kwargs": kwargs,
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "tool execution failed"}
    assert provider.progress_tracker.provider_evidence(provider.batch_id) == (
        {
            "provider": expected_provider,
            "model": expected_model,
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "provider timed out",
        },
    )


class _FailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise RuntimeError("expected failure")


class _ProviderFailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise ToolProviderError("provider failed")
