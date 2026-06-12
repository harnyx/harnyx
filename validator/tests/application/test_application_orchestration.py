from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

import harnyx_validator.application.evaluate_task_run as evaluate_task_run_module
from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.dto.evaluation import MinerTaskRunRequest
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")

TEST_SESSION_TOKEN = uuid4().hex


class StubSandboxClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object], dict[str, object], str, UUID]] = []
        self.response: dict[str, object] | None = None
        self.on_invoke: Callable[[UUID], None] | None = None

    def set_response(self, response: dict[str, object]) -> None:
        self.response = response

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: dict[str, object],
        context: dict[str, object],
        token: str,
        session_id: UUID,
    ) -> dict[str, object]:
        self.requests.append((entrypoint, payload, context, token, session_id))
        if self.on_invoke is not None:
            self.on_invoke(session_id)
        if self.response is None:
            raise RuntimeError("sandbox response not configured")
        return self.response


class EchoToolInvoker:
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
        return ToolInvocationOutput(
            public_payload={
                "data": [
                    {
                        "link": "https://example.com",
                        "title": "Example",
                        "snippet": "ref",
                    },
                ],
                "cost_usd": 0.01,
            },
            actual_cost_usd=0.01,
            actual_cost_provider="parallel",
        )


class StubScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        assert task.query.text == "Harnyx Subnet demo"
        assert response.text == "A direct answer"
        return ScoreBreakdown(
            comparison_score=1.0,
            total_score=1.0,
            scoring_version="v1",
        )


class TrackingScoringService:
    def __init__(self) -> None:
        self.calls = 0

    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        self.calls += 1
        raise AssertionError("scoring should not run for exhausted sessions")


class FailingScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        raise RuntimeError("scoring failed")


class RetryExhaustedScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        raise LlmRetryExhaustedError("scoring retries exhausted")


class _ClockSequence:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def now(self) -> datetime:
        if not self._values:
            raise AssertionError("clock sequence exhausted")
        return self._values.pop(0)


async def test_application_use_cases_cooperate_for_single_task_run() -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()

    session_manager = SessionManager(session_registry, token_registry)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    tool_invoker = EchoToolInvoker()
    usage_tracker = UsageTracker()

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    await executor.execute(
        ToolInvocationRequest(
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            tool="search_web",
            args=("harnyx subnet",),
            kwargs={"query": "harnyx subnet"},
        ),
    )

    session = session_registry.get(session_request.session_id)
    assert session is not None
    session_registry.update(
        session.with_usage(
            session.usage.update(
                llm_usage_totals={
                    "chutes": {
                        "openai/gpt-oss-20b-TEE": LlmUsageTotals(
                            prompt_tokens=10,
                            completion_tokens=15,
                            total_tokens=25,
                            call_count=1,
                        ),
                    },
                },
                llm_tokens_last_call=25,
            ),
        ),
    )

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})

    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )

    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=StubScoringService(),
        session_registry=session_registry,
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ).now,
    )

    outcome = await orchestrator.evaluate(
        MinerTaskRunRequest(
            batch_id=uuid4(),
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            uid=7,
            artifact_id=uuid4(),
            task=task,
        ),
    )

    assert outcome.run.response == Response(text="A direct answer")
    assert outcome.run.details.score_breakdown is not None
    assert outcome.run.details.score_breakdown.total_score == pytest.approx(1.0)
    assert outcome.run.details.elapsed_ms == pytest.approx(300000.0)
    assert outcome.run.completed_at == datetime(2025, 10, 17, 12, 10, tzinfo=UTC)
    assert outcome.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.01)
    assert sandbox.requests == [
        (
            "query",
            {"text": "Harnyx Subnet demo"},
            {},
            TEST_SESSION_TOKEN,
            session_request.session_id,
        ),
    ]


async def test_task_orchestration_logs_scoring_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()
    session_manager = SessionManager(session_registry, token_registry)
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluate_task_run_module.measurement_logger, "info", capture_info)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    batch_id = uuid4()
    artifact_id = uuid4()
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )

    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=StubScoringService(),
        session_registry=session_registry,
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ).now,
    )

    await orchestrator.evaluate(
        MinerTaskRunRequest(
            batch_id=batch_id,
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            uid=7,
            artifact_id=artifact_id,
            task=task,
        ),
    )

    scoring_logs = [extra for message, extra in captured_logs if message == "miner-task scoring finished"]
    assert len(scoring_logs) == 1
    payload = scoring_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["session_id"] == str(session_request.session_id)
    assert payload["artifact_id"] == str(artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == 7
    assert payload["invocation_ms"] >= 0.0
    assert payload["scoring_ms"] >= 0.0
    assert payload["orchestration_ms"] >= 0.0
    assert payload["comparison_score"] == pytest.approx(1.0)
    assert payload["total_score"] == pytest.approx(1.0)
    assert payload["outcome"] == "ok"
    assert payload["error_code"] is None


async def test_task_orchestration_logs_scoring_summary_on_scoring_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()
    session_manager = SessionManager(session_registry, token_registry)
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluate_task_run_module.measurement_logger, "info", capture_info)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    batch_id = uuid4()
    artifact_id = uuid4()
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=FailingScoringService(),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="scoring failed"):
        await orchestrator.evaluate(
            MinerTaskRunRequest(
                batch_id=batch_id,
                session_id=session_request.session_id,
                token=TEST_SESSION_TOKEN,
                uid=7,
                artifact_id=artifact_id,
                task=task,
            ),
        )

    scoring_logs = [extra for message, extra in captured_logs if message == "miner-task scoring finished"]
    assert len(scoring_logs) == 1
    payload = scoring_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["session_id"] == str(session_request.session_id)
    assert payload["artifact_id"] == str(artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == 7
    assert payload["invocation_ms"] >= 0.0
    assert payload["scoring_ms"] >= 0.0
    assert payload["orchestration_ms"] >= 0.0
    assert payload["comparison_score"] is None
    assert payload["total_score"] is None
    assert payload["outcome"] == "error"
    assert payload["error_code"] == str(MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE)


async def test_task_orchestration_logs_retry_exhausted_scoring_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()
    session_manager = SessionManager(session_registry, token_registry)
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluate_task_run_module.measurement_logger, "info", capture_info)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    batch_id = uuid4()
    artifact_id = uuid4()
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=RetryExhaustedScoringService(),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(LlmRetryExhaustedError, match="scoring retries exhausted"):
        await orchestrator.evaluate(
            MinerTaskRunRequest(
                batch_id=batch_id,
                session_id=session_request.session_id,
                token=TEST_SESSION_TOKEN,
                uid=7,
                artifact_id=artifact_id,
                task=task,
            ),
        )

    scoring_logs = [extra for message, extra in captured_logs if message == "miner-task scoring finished"]
    assert len(scoring_logs) == 1
    payload = scoring_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["session_id"] == str(session_request.session_id)
    assert payload["artifact_id"] == str(artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == 7
    assert payload["invocation_ms"] >= 0.0
    assert payload["scoring_ms"] >= 0.0
    assert payload["orchestration_ms"] >= 0.0
    assert payload["comparison_score"] is None
    assert payload["total_score"] is None
    assert payload["outcome"] == "error"
    assert payload["error_code"] == str(MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED)


async def test_task_orchestration_stops_before_scoring_when_session_exhausts() -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()
    session_manager = SessionManager(session_registry, token_registry)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})

    def exhaust_session(session_id: UUID) -> None:
        session = session_registry.get(session_id)
        assert session is not None
        session_registry.update(session.mark_exhausted())

    sandbox.on_invoke = exhaust_session

    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    scoring = TrackingScoringService()
    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=scoring,
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    with pytest.raises(SessionBudgetExhaustedError):
        await orchestrator.evaluate(
            MinerTaskRunRequest(
                batch_id=uuid4(),
                session_id=session_request.session_id,
                token=TEST_SESSION_TOKEN,
                uid=7,
                artifact_id=uuid4(),
                task=task,
            ),
        )

    assert scoring.calls == 0
