from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from caster_commons.application.dto.session import SessionTokenRequest
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer, Response, ScoreBreakdown
from caster_commons.domain.session import LlmUsageTotals
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.tools.dto import ToolInvocationRequest
from caster_commons.tools.executor import ToolExecutor
from caster_commons.tools.usage_tracker import UsageTracker
from caster_validator.application.dto.evaluation import MinerTaskRunRequest
from caster_validator.application.evaluate_task_run import TaskRunOrchestrator
from caster_validator.application.invoke_entrypoint import EntrypointInvoker
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")

TEST_SESSION_TOKEN = uuid4().hex


class StubSandboxClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object], dict[str, object], str, UUID]] = []
        self.response: dict[str, object] | None = None

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
    ) -> dict[str, object]:
        self.calls.append((tool_name, args, kwargs))
        return {
            "data": [
                {
                    "link": "https://example.com",
                    "title": "Example",
                    "snippet": "ref",
                },
            ],
            "cost_usd": 0.01,
        }


class StubScoringService:
    async def score(self, *, task: MinerTask, response: Response) -> ScoreBreakdown:
        assert task.query.text == "Caster Subnet demo"
        assert response.text == "A direct answer"
        return ScoreBreakdown(
            comparison_score=1.0,
            similarity_score=1.0,
            total_score=1.0,
            scoring_version="v1",
        )


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
        query=Query(text="Caster Subnet demo"),
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
            args=("caster subnet",),
            kwargs={"query": "caster subnet"},
        ),
    )

    session = session_registry.get(session_request.session_id)
    assert session is not None
    session_registry.update(
        session.with_usage(
            session.usage.update(
                llm_usage_totals={
                    "chutes": {
                        "openai/gpt-oss-20b": LlmUsageTotals(
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
    assert outcome.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.0025)
    assert sandbox.requests == [
        (
            "query",
            {"text": "Caster Subnet demo"},
            {},
            TEST_SESSION_TOKEN,
            session_request.session_id,
        ),
    ]
