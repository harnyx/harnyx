from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
from harnyx_commons.tools.token_semaphore import TokenSemaphore
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.infrastructure.http.routes import ToolRouteDeps, add_tool_routes
from harnyx_validator.infrastructure.state.run_progress import InMemoryRunProgress
from harnyx_validator.runtime.bootstrap import ALLOWED_TOOL_MODELS, _ProviderTrackingToolExecutor
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

DEMO_SESSION_TOKEN = uuid4().hex


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


class RecordingTokenSemaphore(TokenSemaphore):
    def __init__(self, max_parallel_calls: int) -> None:
        super().__init__(max_parallel_calls)
        self.acquire_calls: list[str] = []
        self.release_calls: list[str] = []

    def acquire(self, token: str) -> None:
        self.acquire_calls.append(token)
        super().acquire(token)

    async def acquire_async(self, token: str) -> None:
        self.acquire_calls.append(token)
        await super().acquire_async(token)

    def release(self, token: str) -> None:
        self.release_calls.append(token)
        super().release(token)


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
        self.token_semaphore = RecordingTokenSemaphore(max_parallel_calls=1)

        self.dependencies = ToolRouteDeps(
            tool_executor=self.tool_executor,
            token_semaphore=self.token_semaphore,
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


class TrackingDependencyProvider:
    def __init__(self, *, llm_provider=None) -> None:
        self.session_registry = FakeSessionRegistry()
        self.receipt_log = FakeReceiptLog()
        self.tokens = InMemoryTokenRegistry()
        self.progress_tracker = InMemoryRunProgress()
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
            llm_provider_name="openai",
        )
        self.token_semaphore = RecordingTokenSemaphore(max_parallel_calls=1)
        self.dependencies = ToolRouteDeps(
            tool_executor=self.tool_executor,
            token_semaphore=self.token_semaphore,
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
    assert provider.token_semaphore.acquire_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.release_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.in_flight(DEMO_SESSION_TOKEN) == 0


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
    assert provider.token_semaphore.acquire_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.release_calls == [DEMO_SESSION_TOKEN]


def test_execute_tool_endpoint_releases_semaphore_on_failure() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_FailingToolExecutor(),
        token_semaphore=provider.token_semaphore,
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
    assert provider.token_semaphore.release_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.in_flight(DEMO_SESSION_TOKEN) == 0


def test_execute_tool_endpoint_returns_generic_detail_for_provider_failure() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_ProviderFailingToolExecutor(),
        token_semaphore=provider.token_semaphore,
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

    provider.token_semaphore.acquire(DEMO_SESSION_TOKEN)
    request_thread = threading.Thread(target=issue_request)
    request_thread.start()
    try:
        deadline = time.monotonic() + 1.0
        while len(provider.token_semaphore.acquire_calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert provider.token_semaphore.acquire_calls == [DEMO_SESSION_TOKEN, DEMO_SESSION_TOKEN]
        assert not done.is_set()
    finally:
        provider.token_semaphore.release(DEMO_SESSION_TOKEN)
    request_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert "error" not in response_box
    response = response_box["response"]
    assert isinstance(response, Response)
    assert response.status_code == 200


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
        },
    )


class _FailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise RuntimeError("expected failure")


class _ProviderFailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise ToolProviderError("provider failed")
