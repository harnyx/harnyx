from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from caster_commons.domain.claim import Rubric
from caster_commons.domain.tool_call import (
    ReceiptMetadata,
    SearchToolResult,
    ToolCall,
    ToolCallOutcome,
    ToolResultPolicy,
)
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_validator.domain.evaluation import MinerAnswer, MinerCitation, MinerEvaluation
from caster_validator.domain.exceptions import InvalidCitationError
from caster_validator.domain.services.receipt_registry import ReceiptRegistry


def make_receipt(
    receipt_id: str,
    *,
    outcome: ToolCallOutcome = ToolCallOutcome.OK,
    result_id: str = "result-ok",
) -> ToolCall:
    return ToolCall(
        receipt_id=receipt_id,
        session_id=uuid4(),
        uid=7,
        tool="search_web",
        issued_at=datetime(2025, 10, 15, tzinfo=UTC),
        outcome=outcome,
        metadata=ReceiptMetadata(
            request_hash="req",
            response_hash="res",
            results=(
                SearchToolResult(
                    index=0,
                    result_id=result_id,
                    url="https://example.com",
                    note="reference",
                    title="Example",
                ),
            ),
            result_policy=ToolResultPolicy.REFERENCEABLE,
        ),
    )


def test_receipt_registry_validates_successful_citations() -> None:
    registry = ReceiptRegistry()
    registry.record(make_receipt("receipt-ok"))

    citation = MinerCitation(
        url="https://example.com",
        note="ref",
        receipt_id="receipt-ok",
        result_id="result-ok",
    )

    registry.validate_citations([citation])


def test_receipt_registry_rejects_missing_receipts() -> None:
    registry = ReceiptRegistry()

    citation = MinerCitation(
        url="https://example.com",
        note="ref",
        receipt_id="missing",
        result_id="hash-missing",
    )

    with pytest.raises(InvalidCitationError):
        registry.validate_citations([citation])


def test_receipt_registry_rejects_failed_receipts() -> None:
    registry = ReceiptRegistry()
    registry.record(make_receipt("bad", outcome=ToolCallOutcome.PROVIDER_ERROR))

    citation = MinerCitation(
        url="https://example.com",
        note="ref",
        receipt_id="bad",
        result_id="hash-bad",
    )

    with pytest.raises(InvalidCitationError):
        registry.validate_citations([citation])


def test_receipt_registry_lookup_roundtrip() -> None:
    registry = ReceiptRegistry()
    receipt = make_receipt("receipt-ok")
    registry.record(receipt)

    assert registry.lookup("receipt-ok") == receipt
    assert registry.values() == (receipt,)


def test_miner_evaluation_requires_receipt_on_citation() -> None:
    registry = ReceiptRegistry()
    registry.record(make_receipt("valid"))

    evaluation = MinerEvaluation(
        evaluation_id=uuid4(),
        session_id=uuid4(),
        uid=1,
        claim_id=uuid4(),
        rubric=Rubric(
            title="Accuracy",
            description="Ensure sources are credible.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        miner_answer=MinerAnswer(
            verdict=1,
            justification="Looks good",
            citations=(
                MinerCitation(
                    url="https://example.com",
                    note="ref",
                    receipt_id="valid",
                    result_id="result-ok",
                ),
            ),
        ),
        completed_at=datetime(2025, 10, 15, tzinfo=UTC),
    )

    registry.validate_citations(evaluation.miner_answer.citations)
