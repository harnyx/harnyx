from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from harnyx_commons.domain.session import Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.protocol_headers import SESSION_ID_HEADER
from harnyx_commons.tools.dto import ToolBudgetSnapshot, ToolInvocationRequest, ToolInvocationResult
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.runtime_invoker import build_miner_sandbox_tool_invoker
from harnyx_commons.tools.token_semaphore import (
    DEFAULT_TOOL_CONCURRENCY_LIMITS,
    ToolConcurrencyLimiter,
    ToolConcurrencyLimits,
)
from harnyx_commons.tools.types import ToolName
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.infrastructure.http.routes import ToolRouteDeps, add_tool_routes
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

DEMO_SESSION_TOKEN = uuid4().hex
DEFAULT_LIMIT_LLM_MODEL = "openai/gpt-oss-20b"
DEEPSEEK_V32_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"


def _invocation(tool: ToolName = "search_web") -> ToolInvocationRequest:
    kwargs = {"model": DEFAULT_LIMIT_LLM_MODEL} if tool == "llm_chat" else {}
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=DEMO_SESSION_TOKEN,
        tool=tool,
        args=(),
        kwargs=kwargs,
    )


def _mixed_invocations(count: int) -> list[ToolInvocationRequest]:
    tools: tuple[ToolName, ...] = ("search_web", "search_ai", "fetch_page", "tooling_info", "test_tool", "llm_chat")
    return [_invocation(tools[index % len(tools)]) for index in range(count)]


def create_test_app(dependency_provider: DemoDependencyProvider) -> FastAPI:
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
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        self.calls.append((tool_name, args, kwargs))
        query = kwargs.get("query", "demo")
        return ToolInvocationOutput(
            public_payload={
                "data": [
                    {
                        "link": f"https://example.com/{query}",
                        "title": "Demo",
                        "snippet": "demo",
                    },
                ],
            },
            actual_cost_usd=0.005,
            actual_cost_provider="parallel",
        )


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


class ToolingInfoDependencyProvider:
    def __init__(self) -> None:
        self.session_registry = FakeSessionRegistry()
        self.receipt_log = InMemoryReceiptLog()
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

        self.tool_executor = ToolExecutor(
            session_registry=self.session_registry,
            receipt_log=self.receipt_log,
            usage_tracker=UsageTracker(),
            tool_invoker=build_miner_sandbox_tool_invoker(self.receipt_log),
            token_registry=self.tokens,
            clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
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
    assert session_snapshot.usage.total_cost_usd == pytest.approx(0.005)
    invocation = _invocation("search_web")
    assert provider.tool_concurrency_limiter.acquire_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.release_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.in_flight(invocation) == 0


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
    invocation = _invocation("search_web")
    assert provider.tool_concurrency_limiter.release_calls == [(DEMO_SESSION_TOKEN, "search_web")]
    assert provider.tool_concurrency_limiter.in_flight(invocation) == 0


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
    stored = provider.session_registry.get(provider.session.session_id)
    assert stored is not None
    assert stored.failure_code is None


def test_execute_tool_endpoint_returns_platform_provider_failure_detail() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_PlatformProviderFailingToolExecutor("empty_output"),
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
    assert response.json() == {"detail": "empty_output"}


def test_execute_tool_endpoint_coarsens_platform_provider_http_detail() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_PlatformProviderFailingToolExecutor("http_429: provider body with detail"),
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
    assert response.json() == {"detail": "http_429"}


def test_execute_tool_endpoint_keeps_unsafe_platform_provider_failure_detail_generic() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_PlatformProviderFailingToolExecutor("provider body with detail"),
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


def test_execute_tool_endpoint_unexpected_internal_error_uses_generic_500_body() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=_UnexpectedFailingToolExecutor(),
        tool_concurrency_limiter=provider.tool_concurrency_limiter,
    )
    app = create_test_app(provider)
    client = TestClient(app, raise_server_exceptions=False)

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

    assert response.status_code == 500
    assert "tool secret" not in response.text


def test_execute_tool_endpoint_rejects_repo_tools() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "search_repo",
            "args": [],
            "kwargs": {},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 422


def test_execute_tool_endpoint_supports_tooling_info() -> None:
    provider = ToolingInfoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "tool": "tooling_info",
            "args": [],
            "kwargs": {},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_policy"] == "log_only"
    assert body["budget"]["session_budget_usd"] == pytest.approx(provider.session.budget_usd)
    assert body["budget"]["session_hard_limit_usd"] == pytest.approx(provider.session.effective_hard_limit_usd)
    assert body["budget"]["session_used_budget_usd"] == pytest.approx(0.0)
    assert body["budget"]["session_remaining_budget_usd"] == pytest.approx(provider.session.budget_usd)
    response_payload = body["response"]
    assert "search_repo" not in response_payload["tool_names"]
    assert "get_repo_file" not in response_payload["tool_names"]
    assert "search_items" not in response_payload["tool_names"]
    assert "search_repo" not in response_payload["pricing"]
    assert "get_repo_file" not in response_payload["pricing"]
    assert "search_items" not in response_payload["pricing"]

    session_snapshot = provider.session_registry.get(provider.session.session_id)
    assert session_snapshot is not None
    assert session_snapshot.usage.total_cost_usd == pytest.approx(0.0)


def test_execute_tool_endpoint_waits_when_shared_tool_cap_is_full_then_succeeds() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=cast(ToolExecutor, _StaticToolExecutor()),
        tool_concurrency_limiter=provider.tool_concurrency_limiter,
    )
    app = create_test_app(provider)
    held = _mixed_invocations(20)
    for invocation in held:
        provider.tool_concurrency_limiter.acquire(invocation)

    try:
        unblock_invocation = held.pop()
        response = _issue_waiting_tool_request(
            app=app,
            provider=provider,
            tool="llm_chat",
            model=DEFAULT_LIMIT_LLM_MODEL,
            expected_acquire_calls=21,
            unblock_invocation=unblock_invocation,
        )
    finally:
        for invocation in held:
            provider.tool_concurrency_limiter.release(invocation)

    assert response.status_code == 200
    assert provider.tool_concurrency_limiter.in_flight(_invocation("llm_chat")) == 0


def test_execute_tool_endpoint_uses_same_cap_for_different_llm_models() -> None:
    provider = DemoDependencyProvider()
    provider.dependencies = ToolRouteDeps(
        tool_executor=cast(ToolExecutor, _StaticToolExecutor()),
        tool_concurrency_limiter=provider.tool_concurrency_limiter,
    )
    app = create_test_app(provider)
    held = _mixed_invocations(20)
    for invocation in held:
        provider.tool_concurrency_limiter.acquire(invocation)

    try:
        unblock_invocation = held.pop()
        response = _issue_waiting_tool_request(
            app=app,
            provider=provider,
            tool="llm_chat",
            model=DEEPSEEK_V32_MODEL,
            expected_acquire_calls=21,
            unblock_invocation=unblock_invocation,
        )
    finally:
        for invocation in held:
            provider.tool_concurrency_limiter.release(invocation)

    assert response.status_code == 200
    assert provider.tool_concurrency_limiter.in_flight(_invocation("llm_chat")) == 0


def test_execute_tool_endpoint_waits_for_twenty_first_mixed_tool_call_then_succeeds() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    held = _mixed_invocations(20)
    for invocation in held:
        provider.tool_concurrency_limiter.acquire(invocation)

    try:
        unblock_invocation = held.pop()
        response = _issue_waiting_tool_request(
            app=app,
            provider=provider,
            tool="search_web",
            expected_acquire_calls=21,
            unblock_invocation=unblock_invocation,
        )
    finally:
        for invocation in held:
            provider.tool_concurrency_limiter.release(invocation)

    assert response.status_code == 200
    assert provider.tool_concurrency_limiter.in_flight(_invocation("search_web")) == 0


def _issue_waiting_tool_request(
    *,
    app: FastAPI,
    provider: DemoDependencyProvider,
    tool: ToolName,
    model: str = DEFAULT_LIMIT_LLM_MODEL,
    expected_acquire_calls: int,
    unblock_invocation: ToolInvocationRequest,
) -> Response:
    response_box: dict[str, Response | Exception] = {}
    done = threading.Event()

    def issue_request() -> None:
        try:
            with TestClient(app) as client:
                kwargs = (
                    {"model": model, "messages": [{"role": "user", "content": "hi"}]}
                    if tool == "llm_chat"
                    else {"query": "demo"}
                )
                response_box["response"] = client.post(
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
        except Exception as exc:  # pragma: no cover - defensive capture
            response_box["error"] = exc
        finally:
            done.set()

    request_thread = threading.Thread(target=issue_request)
    request_thread.start()
    deadline = time.monotonic() + 1.0
    while len(provider.tool_concurrency_limiter.acquire_calls) < expected_acquire_calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(provider.tool_concurrency_limiter.acquire_calls) == expected_acquire_calls
    assert not done.is_set()

    provider.tool_concurrency_limiter.release(unblock_invocation)
    request_thread.join(timeout=1.0)

    assert not request_thread.is_alive()
    assert "error" not in response_box
    response = response_box["response"]
    assert isinstance(response, Response)
    return response


class _FailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise RuntimeError("expected failure")


class _StaticToolExecutor:
    async def execute(self, invocation: ToolInvocationRequest) -> ToolInvocationResult:
        return ToolInvocationResult(
            receipt=ToolCall(
                receipt_id="receipt-1",
                session_id=invocation.session_id,
                uid=7,
                tool=invocation.tool,
                issued_at=datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
                outcome=ToolCallOutcome.OK,
                details=ToolCallDetails(
                    request_hash="request-hash",
                    response_hash="response-hash",
                    response_payload={"ok": True},
                    result_policy=ToolResultPolicy.LOG_ONLY,
                ),
            ),
            response_payload={"ok": True},
            budget=ToolBudgetSnapshot(
                session_budget_usd=1.0,
                session_hard_limit_usd=1.0,
                session_used_budget_usd=0.0,
                session_remaining_budget_usd=1.0,
            ),
        )


class _ProviderFailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise ToolProviderError("provider failed")


class _PlatformToolProxyProviderError(ToolProviderError):
    status_code = 502
    error_code = "provider_failed"


class _PlatformProviderFailingToolExecutor:
    def __init__(self, message: str) -> None:
        self._message = message

    async def execute(self, _: object) -> object:
        raise _PlatformToolProxyProviderError(self._message)


class _UnexpectedFailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise AssertionError("tool secret exploded")


def test_execute_tool_endpoint_rejects_missing_token_header() -> None:
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
        headers={SESSION_ID_HEADER: str(provider.session.session_id)},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "missing x-platform-token header"}


def test_execute_tool_endpoint_rejects_missing_session_header() -> None:
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
        headers={"x-platform-token": DEMO_SESSION_TOKEN},
    )

    assert response.status_code == 422


def test_execute_tool_endpoint_rejects_malformed_session_header() -> None:
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
            SESSION_ID_HEADER: "not-a-uuid",
        },
    )

    assert response.status_code == 422


def test_execute_tool_endpoint_rejects_legacy_body_session_id_field() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "session_id": str(provider.session.session_id),
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
        headers={
            "x-platform-token": DEMO_SESSION_TOKEN,
            SESSION_ID_HEADER: str(provider.session.session_id),
        },
    )

    assert response.status_code == 422


def test_execute_tool_openapi_declares_platform_token_security() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)

    operation = app.openapi()["paths"]["/v1/tools/execute"]["post"]
    security = operation["security"]
    parameters = operation["parameters"]
    assert {"PlatformToken": []} in security
    assert any(
        parameter.get("name") == SESSION_ID_HEADER
        and parameter.get("in") == "header"
        and parameter.get("required") is True
        and parameter.get("schema", {}).get("format") == "uuid"
        for parameter in parameters
    )


def test_execute_tool_openapi_excludes_repo_tools() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)

    schema = app.openapi()["components"]["schemas"]["ToolExecuteRequestDTO"]
    tool_enum = schema["properties"]["tool"]["enum"]

    assert "search_repo" not in tool_enum
    assert "get_repo_file" not in tool_enum
