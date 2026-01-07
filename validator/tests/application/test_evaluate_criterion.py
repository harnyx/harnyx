from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from caster_commons.domain.claim import EvaluationClaim, ReferenceAnswer, Rubric
from caster_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from caster_commons.domain.tool_call import (
    ReceiptMetadata,
    SearchToolResult,
    ToolCall,
    ToolCallOutcome,
    ToolResult,
    ToolResultPolicy,
)
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_commons.llm.pricing import ALLOWED_TOOL_MODELS
from caster_validator.application.dto.evaluation import EntrypointInvocationResult, EvaluationRequest
from caster_validator.application.evaluate_criterion import EvaluationOrchestrator, UsageSummarizer
from caster_validator.application.services.evaluation_scoring import EvaluationScoringService
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry, StubGrader

pytestmark = pytest.mark.anyio("asyncio")


class StubEntrypointInvoker:
    def __init__(self, result: EntrypointInvocationResult) -> None:
        self._result = result
        self.invocations: list = []

    async def invoke(self, request):
        self.invocations.append(request)
        return self._result


def make_receipt(
    receipt_id: str,
    *,
    session_id: UUID,
    payload: dict[str, object] | None = None,
    results: tuple[ToolResult, ...] | None = None,
) -> ToolCall:
    response_payload = payload or {
        "response": {
            "data": [
                {
                    "link": "https://example.com",
                    "snippet": "ref",
                },
            ],
        },
    }

    def build_result(
        index: int,
        *,
        url: str,
        note: str,
        title: str | None,
        result_id: str | None = None,
    ) -> SearchToolResult:
        return SearchToolResult(
            index=index,
            result_id=result_id or f"result-{index}",
            url=url,
            note=note,
            title=title,
        )

    return ToolCall(
        receipt_id=receipt_id,
        session_id=session_id,
        uid=7,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        metadata=ReceiptMetadata(
            request_hash="req",
            response_hash="res",
            response_payload=response_payload,
            results=results
            or (
                build_result(
                    0,
                    url="https://example.com",
                    note="ref",
                    title="Example",
                ),
            ),
            result_policy=ToolResultPolicy.REFERENCEABLE,
        ),
    )


def test_usage_summarizer_ignores_non_search_tool_receipts() -> None:
    session_id = uuid4()
    claim_id = uuid4()
    issued = datetime(2025, 10, 17, 11, tzinfo=UTC)
    session = Session(
        session_id=session_id,
        uid=7,
        claim_id=claim_id,
        issued_at=issued,
        expires_at=issued + timedelta(hours=1),
            usage=SessionUsage(
                llm_tokens_last_call=10,
                llm_usage_totals={
                    "chutes": {
                        ALLOWED_TOOL_MODELS[0]: LlmUsageTotals(
                            prompt_tokens=4,
                            completion_tokens=6,
                            total_tokens=10,
                            call_count=1,
                        ),
                    },
                },
                total_cost_usd=0.0,
            ),
    )

    search_receipt = make_receipt("search-receipt", session_id=session_id)
    llm_receipt = ToolCall(
        receipt_id="llm-receipt",
        session_id=session_id,
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        metadata=ReceiptMetadata(
            request_hash="req-llm",
            response_hash="res-llm",
            response_payload={"response": {"messages": []}},
            results=(),
            result_policy=ToolResultPolicy.LOG_ONLY,
        ),
    )

    summarizer = UsageSummarizer()
    _, totals = summarizer.summarize(session, (search_receipt, llm_receipt))

    search_totals = totals["search_tool"]
    assert search_totals["call_count"] == 1


async def test_evaluation_orchestrator_builds_miner_evaluation() -> None:
    session_id = uuid4()
    receipt = make_receipt("receipt-1", session_id=session_id)
    receipt_registry = FakeReceiptLog()
    receipt_registry.record(receipt)
    session_registry = FakeSessionRegistry()
    session_registry = FakeSessionRegistry()
    session_registry = FakeSessionRegistry()

    entrypoint_result = EntrypointInvocationResult(
        result={
            "verdict": 1,
            "justification": "Looks good",
            "citations": [
                {
                    "url": "https://example.com",
                    "note": "ref",
                    "receipt_id": "receipt-1",
                    "result_id": receipt.metadata.results[0].result_id,
                },
            ],
        },
        tool_receipts=(receipt,),
    )
    invoker = StubEntrypointInvoker(entrypoint_result)

    claim = EvaluationClaim(
        claim_id=uuid4(),
        text="Example claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    register_session(
        session_registry,
        session_id=session_id,
        uid=7,
        claim_id=claim.claim_id,
    )

    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_registry,
        scoring_service=EvaluationScoringService(receipt_registry, grader=StubGrader()),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC),
    )

    request = EvaluationRequest(
        session_id=session_id,
        token=uuid4().hex,
        uid=7,
        entrypoint="evaluate_criterion",
        payload={"query": "demo"},
        context={"claim": claim.text},
        claim=claim,
        evaluation_id=uuid4(),
    )

    outcome = await orchestrator.evaluate(request)

    assert outcome.evaluation.miner_answer.verdict == 1
    citation = outcome.evaluation.miner_answer.citations[0]
    assert citation.receipt_id == "receipt-1"
    assert citation.result_id == receipt.metadata.results[0].result_id
    assert citation.url == "https://example.com"
    assert citation.note == "ref"
    assert outcome.tool_receipts == (receipt,)
    assert outcome.score.verdict_score == 0.5
    assert outcome.score.support_score == 0.5
    assert outcome.score.justification_pass is True
    assert outcome.score.failed_citation_ids == ()
    assert outcome.usage.total_tokens == 10


async def test_evaluation_orchestrator_overwrites_with_canonical_null_fields() -> None:
    session_id = uuid4()
    canonical_result = SearchToolResult(
        index=0,
        result_id="canonical",
        url="https://example.com",
        note=None,
        title=None,
    )
    receipt = make_receipt("receipt-1", session_id=session_id, results=(canonical_result,))
    receipt_registry = FakeReceiptLog()
    receipt_registry.record(receipt)
    session_registry = FakeSessionRegistry()

    entrypoint_result = EntrypointInvocationResult(
        result={
            "verdict": 1,
            "justification": "Looks good",
            "citations": [
                {
                    "url": "https://forged.example",
                    "note": "forged note",
                    "receipt_id": "receipt-1",
                    "result_id": canonical_result.result_id,
                },
            ],
        },
        tool_receipts=(receipt,),
    )
    invoker = StubEntrypointInvoker(entrypoint_result)

    claim = EvaluationClaim(
        claim_id=uuid4(),
        text="Example claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    register_session(
        session_registry,
        session_id=session_id,
        uid=7,
        claim_id=claim.claim_id,
    )

    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_registry,
        scoring_service=EvaluationScoringService(receipt_registry, grader=StubGrader()),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC),
    )

    request = EvaluationRequest(
        session_id=session_id,
        token=uuid4().hex,
        uid=7,
        entrypoint="evaluate_criterion",
        payload={"query": "demo"},
        context={"claim": claim.text},
        claim=claim,
        evaluation_id=uuid4(),
    )

    outcome = await orchestrator.evaluate(request)
    citation = outcome.evaluation.miner_answer.citations[0]

    assert citation.url == "https://example.com"
    assert citation.note is None
    assert citation.receipt_id == "receipt-1"
    assert citation.result_id == canonical_result.result_id


async def test_evaluation_orchestrator_drops_invalid_citations_and_fails_support() -> None:
    invoker = StubEntrypointInvoker(
        EntrypointInvocationResult(
            result={
                "verdict": 1,
                "justification": "Looks good",
                "citations": [
                    {
                        "url": "https://example.com",
                        "note": "ref",
                        "receipt_id": "missing",
                        "result_id": "missing-hash",
                    },
                ],
            },
            tool_receipts=(),
        ),
    )
    receipt_log = FakeReceiptLog()
    session_registry = FakeSessionRegistry()
    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=EvaluationScoringService(
            receipt_log,
            grader=StubGrader(support_ok=False, rationale="missing receipts"),
        ),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC),
    )
    claim = EvaluationClaim(
        claim_id=uuid4(),
        text="Example claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )
    session_id = uuid4()
    register_session(
        session_registry,
        session_id=session_id,
        uid=7,
        claim_id=claim.claim_id,
    )
    request = EvaluationRequest(
        session_id=session_id,
        token=uuid4().hex,
        uid=7,
        entrypoint="evaluate_criterion",
        payload={},
        context={},
        claim=claim,
        evaluation_id=uuid4(),
    )

    outcome = await orchestrator.evaluate(request)
    assert outcome.score.verdict_score == 0.5
    assert outcome.score.support_score == 0.0
    assert outcome.score.justification_pass is False
    assert outcome.score.failed_citation_ids == ()
    # citation was dropped because it could not be resolved
    assert outcome.evaluation.miner_answer.citations == ()


async def test_evaluation_orchestrator_drops_other_session_citations_and_fails_support() -> None:
    session_id = uuid4()
    other_session_id = uuid4()
    receipt = make_receipt("receipt-1", session_id=other_session_id)
    receipt_log = FakeReceiptLog()
    receipt_log.record(receipt)
    session_registry = FakeSessionRegistry()

    invoker = StubEntrypointInvoker(
        EntrypointInvocationResult(
            result={
                "verdict": 1,
                "justification": "Looks good",
                "citations": [
                    {
                        "url": "https://example.com",
                        "note": "ref",
                        "receipt_id": "receipt-1",
                        "result_id": receipt.metadata.results[0].result_id,
                    },
                ],
            },
            tool_receipts=(receipt,),
        ),
    )

    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=EvaluationScoringService(
            receipt_log,
            grader=StubGrader(support_ok=False, rationale="wrong session"),
        ),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC),
    )

    claim = EvaluationClaim(
        claim_id=uuid4(),
        text="Example claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    register_session(
        session_registry,
        session_id=session_id,
        uid=7,
        claim_id=claim.claim_id,
    )

    request = EvaluationRequest(
        session_id=session_id,
        token=uuid4().hex,
        uid=7,
        entrypoint="evaluate_criterion",
        payload={},
        context={},
        claim=claim,
        evaluation_id=uuid4(),
    )

    outcome = await orchestrator.evaluate(request)
    assert outcome.score.verdict_score == 0.5
    assert outcome.score.support_score == 0.0
    assert outcome.score.justification_pass is False
    assert outcome.score.failed_citation_ids == ()
    assert outcome.evaluation.miner_answer.citations == ()


async def test_evaluation_orchestrator_drops_unknown_result_id_and_fails_support() -> None:
    session_id = uuid4()
    receipt = make_receipt("receipt-1", session_id=session_id)
    invalid_result_id = "invalid-result"
    receipt_log = FakeReceiptLog()
    receipt_log.record(receipt)
    session_registry = FakeSessionRegistry()

    invoker = StubEntrypointInvoker(
        EntrypointInvocationResult(
            result={
                "verdict": 1,
                "justification": "Looks good",
                "citations": [
                    {
                        "url": "https://example.com",
                        "note": "ref",
                        "receipt_id": "receipt-1",
                        "result_id": invalid_result_id,
                    },
                ],
            },
            tool_receipts=(receipt,),
        ),
    )

    orchestrator = EvaluationOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=EvaluationScoringService(
            receipt_log,
            grader=StubGrader(support_ok=False, rationale="unknown result"),
        ),
        session_registry=session_registry,
        clock=lambda: datetime(2025, 10, 17, 12, tzinfo=UTC),
    )

    claim = EvaluationClaim(
        claim_id=uuid4(),
        text="Example claim",
        rubric=Rubric(
            title="Accuracy",
            description="Check accuracy",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ref", citations=()),
    )

    register_session(
        session_registry,
        session_id=session_id,
        uid=7,
        claim_id=claim.claim_id,
    )

    request = EvaluationRequest(
        session_id=session_id,
        token=uuid4().hex,
        uid=7,
        entrypoint="evaluate_criterion",
        payload={},
        context={},
        claim=claim,
        evaluation_id=uuid4(),
    )

    outcome = await orchestrator.evaluate(request)

    assert outcome.score.verdict_score == 0.5
    assert outcome.score.support_score == 0.0
    assert outcome.score.justification_pass is False
    assert outcome.score.failed_citation_ids == ()
    assert outcome.evaluation.miner_answer.citations == ()
def register_session(
    registry: FakeSessionRegistry,
    *,
    session_id: UUID,
    uid: int,
    claim_id: UUID,
) -> None:
    issued = datetime(2025, 10, 17, 11, tzinfo=UTC)
    budget = SessionUsage(
        llm_tokens_last_call=10,
        llm_usage_totals={
            "chutes": {
                ALLOWED_TOOL_MODELS[0]: LlmUsageTotals(
                    prompt_tokens=4,
                    completion_tokens=6,
                    total_tokens=10,
                    call_count=1,
                ),
            },
        },
        total_cost_usd=0.0,
    )
    registry.create(
        Session(
            session_id=session_id,
            uid=uid,
            claim_id=claim_id,
            issued_at=issued,
            expires_at=issued + timedelta(hours=1),
            usage=budget,
        ),
    )
