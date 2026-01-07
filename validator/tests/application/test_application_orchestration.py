from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from caster_commons.application.dto.session import SessionTokenRequest
from caster_commons.application.session_manager import SessionManager
from caster_commons.domain.claim import EvaluationClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.llm.pricing import ALLOWED_TOOL_MODELS
from caster_commons.tools.dto import ToolInvocationRequest
from caster_commons.tools.executor import ToolExecutor
from caster_commons.tools.usage_tracker import UsageTracker
from caster_validator.application.dto.evaluation import EvaluationRequest
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator
from caster_validator.application.invoke_entrypoint import EntrypointInvoker
from caster_validator.application.services.evaluation_scoring import EvaluationScoringService
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry, StubGrader

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


async def test_application_use_cases_cooperate_for_single_evaluation() -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()

    session_manager = SessionManager(session_registry, token_registry)

    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        claim_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    tool_invoker = EchoToolInvoker()
    usage_tracker = UsageTracker(cost_limit_usd=0.5)

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    tool_result = await executor.execute(
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
                                ALLOWED_TOOL_MODELS[0]: LlmUsageTotals(
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
    sandbox.set_response(
        {
            "verdict": 1,
            "justification": "looks good",
            "citations": [
                {
                    "url": "https://example.com",
                    "note": "ref",
                    "receipt_id": tool_result.receipt.receipt_id,
                    "result_id": tool_result.receipt.metadata.results[0].result_id,
                },
            ],
        },
    )

    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )

    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=EvaluationScoringService(receipt_log, grader=StubGrader()),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
    )

    claim = EvaluationClaim(
        claim_id=session_request.claim_id,
        text="Caster Subnet demo",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    evaluation_outcome = await orchestrator.evaluate(
        EvaluationRequest(
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            uid=7,
            entrypoint="evaluate_criterion",
            payload={"query": "caster subnet"},
            context={"claim": claim.text},
            claim=claim,
            evaluation_id=uuid4(),
        ),
    )

    assert evaluation_outcome.score.total == 1.0
