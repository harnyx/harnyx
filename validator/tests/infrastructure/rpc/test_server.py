from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from caster_commons.domain.session import Session, SessionStatus, SessionUsage
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.tools.executor import ToolExecutor
from caster_commons.tools.token_semaphore import TokenSemaphore
from caster_commons.tools.usage_tracker import UsageTracker
from caster_validator.infrastructure.http.routes import ToolRouteDeps, add_tool_routes
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
            claim_id=uuid4(),
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
            usage=SessionUsage(),
            status=SessionStatus.ACTIVE,
        )
        self.session_registry.create(self.session)
        self.tokens.register(self.session.session_id, DEMO_SESSION_TOKEN)

        usage_tracker = UsageTracker(cost_limit_usd=0.1)
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


def test_execute_tool_endpoint_records_receipt() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/v1/tools/execute",
        json={
            "session_id": str(provider.session.session_id),
            "token": DEMO_SESSION_TOKEN,
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    receipt_id = body["receipt_id"]
    receipt = provider.receipt_log.lookup(receipt_id)
    assert receipt is not None
    assert body["results"][0]["result_id"] == receipt.metadata.results[0].result_id
    assert body["result_policy"] == receipt.metadata.result_policy.value
    assert receipt.metadata.request_hash
    session_snapshot = provider.session_registry.get(provider.session.session_id)
    assert session_snapshot is not None
    assert session_snapshot.usage.total_cost_usd == 0.0025
    assert provider.token_semaphore.acquire_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.release_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.in_flight(DEMO_SESSION_TOKEN) == 0


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
            "session_id": str(provider.session.session_id),
            "token": DEMO_SESSION_TOKEN,
            "tool": "search_web",
            "args": ["demo"],
            "kwargs": {"query": "demo"},
        },
    )

    assert response.status_code == 400
    assert provider.token_semaphore.release_calls == [DEMO_SESSION_TOKEN]
    assert provider.token_semaphore.in_flight(DEMO_SESSION_TOKEN) == 0


def test_execute_tool_endpoint_rejects_when_concurrency_limit_exceeded() -> None:
    provider = DemoDependencyProvider()
    app = create_test_app(provider)
    client = TestClient(app)

    provider.token_semaphore.acquire(DEMO_SESSION_TOKEN)
    try:
        response = client.post(
            "/v1/tools/execute",
            json={
                "session_id": str(provider.session.session_id),
                "token": DEMO_SESSION_TOKEN,
                "tool": "search_web",
                "args": ["demo"],
                "kwargs": {"query": "demo"},
            },
        )
    finally:
        provider.token_semaphore.release(DEMO_SESSION_TOKEN)

    assert response.status_code == 400


class _FailingToolExecutor:
    async def execute(self, _: object) -> object:
        raise RuntimeError("expected failure")
