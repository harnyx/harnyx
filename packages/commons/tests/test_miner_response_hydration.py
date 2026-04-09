from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import hydrate_miner_response_payload
from harnyx_commons.domain.miner_task import AnswerCitation, Response
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolResultPolicy,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog


def test_hydrate_miner_response_payload_hydrates_same_session_referenceable_results() -> None:
    session_id = uuid4()
    receipt_log = InMemoryReceiptLog()
    receipt_log.record(
        ToolCall(
            receipt_id="receipt-1",
            session_id=session_id,
            uid=42,
            tool="search_web",
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash="req",
                response_hash="res",
                result_policy=ToolResultPolicy.REFERENCEABLE,
                results=(
                    SearchToolResult(
                        index=0,
                        result_id="result-1",
                        url="https://example.com/source",
                        note="Primary source",
                        title="Example source",
                    ),
                ),
            ),
        )
    )

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
        },
        session_id=session_id,
        receipt_log=receipt_log,
    )

    assert response == Response(
        text="Answer",
        citations=(
            AnswerCitation(
                url="https://example.com/source",
                note="Primary source",
                title="Example source",
            ),
        ),
    )


def test_hydrate_miner_response_payload_drops_invalid_or_cross_session_citations() -> None:
    session_id = uuid4()
    receipt_log = InMemoryReceiptLog()
    receipt_log.record(
        ToolCall(
            receipt_id="receipt-1",
            session_id=uuid4(),
            uid=42,
            tool="search_web",
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash="req",
                response_hash="res",
                result_policy=ToolResultPolicy.REFERENCEABLE,
                results=(
                    SearchToolResult(
                        index=0,
                        result_id="result-1",
                        url="https://example.com/source",
                    ),
                ),
            ),
        )
    )

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {"receipt_id": "receipt-1", "result_id": "result-1"},
                {"receipt_id": "missing", "result_id": "result-1"},
            ],
        },
        session_id=session_id,
        receipt_log=receipt_log,
    )

    assert response == Response(text="Answer")


def test_hydrate_miner_response_payload_preserves_existing_list_ingress_shape() -> None:
    response = hydrate_miner_response_payload(
        {"text": "Answer", "citations": []},
        session_id=uuid4(),
        receipt_log=InMemoryReceiptLog(),
    )

    assert response == Response(text="Answer")


def test_hydrate_miner_response_payload_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError):
        hydrate_miner_response_payload(
            {"text": "   "},
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )


def test_hydrate_miner_response_payload_rejects_more_than_fifty_citations() -> None:
    with pytest.raises(ValidationError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [
                    {"receipt_id": f"receipt-{index}", "result_id": f"result-{index}"}
                    for index in range(51)
                ],
            },
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )
