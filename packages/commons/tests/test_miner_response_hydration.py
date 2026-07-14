from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    MinerResponsePayloadError,
)
from harnyx_commons.application.miner_response_hydration import (
    hydrate_miner_response_payload as _hydrate_miner_response_payload,
)
from harnyx_commons.domain.miner_task import AnswerCitation, Query, Response
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolResultPolicy,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog

_LEGACY_QUERY = Query(text="question")


def hydrate_miner_response_payload(
    payload: object,
    *,
    session_id: UUID,
    receipt_log: InMemoryReceiptLog,
    query: Query = _LEGACY_QUERY,
) -> Response:
    return _hydrate_miner_response_payload(
        payload,
        query=query,
        session_id=session_id,
        receipt_log=receipt_log,
    )


def _source_text(length: int = 160) -> str:
    return "".join(str(index % 10) for index in range(length))


def _receipt_log_with_result(
    *,
    session_id: UUID,
    note: str | None,
) -> InMemoryReceiptLog:
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
                        note=note,
                        title="Example source",
                    ),
                ),
            ),
        )
    )
    return receipt_log


def test_hydrate_miner_response_payload_materializes_full_result_when_slices_are_omitted() -> None:
    session_id = uuid4()
    source_text = "Primary source"

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response == Response(
        text="Answer",
        citations=(
            AnswerCitation(
                url="https://example.com/source",
                note=f"[slice 0:{len(source_text)}]\n{source_text}",
                title="Example source",
            ),
        ),
    )


def test_hydrate_miner_response_payload_materializes_targeted_slice() -> None:
    session_id = uuid4()
    source_text = _source_text()

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": 120}],
                }
            ],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response == Response(
        text="Answer",
        citations=(
            AnswerCitation(
                url="https://example.com/source",
                note=f"[slice 0:120]\n{source_text[:120]}",
                title="Example source",
            ),
        ),
    )


def test_hydrate_miner_response_payload_materializes_multiple_slices() -> None:
    session_id = uuid4()
    source_text = _source_text(320)

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": 120}, {"start": 180, "end": 300}],
                }
            ],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response.citations is not None
    assert response.citations[0].note == (
        f"[slice 0:120]\n{source_text[:120]}\n\n"
        f"[slice 180:300]\n{source_text[180:300]}"
    )


def test_hydrate_miner_response_payload_uses_unstripped_source_text_offsets() -> None:
    session_id = uuid4()
    source_text = f"  {_source_text(140)}"

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": 120}],
                }
            ],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response.citations is not None
    assert response.citations[0].note == f"[slice 0:120]\n{source_text[:120]}"


def test_hydrate_miner_response_payload_allows_full_short_source_slice() -> None:
    session_id = uuid4()
    source_text = "short source"

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": len(source_text)}],
                }
            ],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response.citations is not None
    assert response.citations[0].note == f"[slice 0:{len(source_text)}]\n{source_text}"


def test_hydrate_miner_response_payload_rejects_short_slice_from_long_source() -> None:
    session_id = uuid4()

    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [
                    {
                        "receipt_id": "receipt-1",
                        "result_id": "result-1",
                        "slices": [{"start": 0, "end": 99}],
                    }
                ],
            },
            session_id=session_id,
            receipt_log=_receipt_log_with_result(session_id=session_id, note=_source_text()),
        )


def test_hydrate_miner_response_payload_rejects_out_of_bounds_slice() -> None:
    session_id = uuid4()

    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [
                    {
                        "receipt_id": "receipt-1",
                        "result_id": "result-1",
                        "slices": [{"start": 0, "end": 500}],
                    }
                ],
            },
            session_id=session_id,
            receipt_log=_receipt_log_with_result(session_id=session_id, note=_source_text()),
        )


def test_hydrate_miner_response_payload_rejects_citation_when_source_text_is_absent() -> None:
    session_id = uuid4()

    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
            },
            session_id=session_id,
            receipt_log=_receipt_log_with_result(session_id=session_id, note="   "),
        )


def test_hydrate_miner_response_payload_rejects_total_materialized_evidence_over_budget() -> None:
    session_id = uuid4()

    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
            },
            session_id=session_id,
            receipt_log=_receipt_log_with_result(session_id=session_id, note=_source_text(120_001)),
        )


def test_hydrate_miner_response_payload_drops_invalid_or_cross_session_citations() -> None:
    session_id = uuid4()
    receipt_log = _receipt_log_with_result(session_id=uuid4(), note=_source_text())

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
    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            {"text": "   "},
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )


def test_hydrate_miner_response_payload_rejects_more_than_two_hundred_citations() -> None:
    with pytest.raises(ValidationError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [
                    {"receipt_id": f"receipt-{index}", "result_id": f"result-{index}"}
                    for index in range(201)
                ],
            },
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )


def test_hydrate_miner_response_payload_rejects_more_than_four_hundred_segments() -> None:
    with pytest.raises(ValidationError):
        hydrate_miner_response_payload(
            {
                "text": "Answer",
                "citations": [
                    {
                        "receipt_id": "receipt-1",
                        "result_id": "result-1",
                        "slices": [{"start": index * 100, "end": (index + 1) * 100} for index in range(401)],
                    }
                ],
            },
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )


def test_hydrate_miner_response_payload_rejects_text_over_eighty_thousand_chars() -> None:
    with pytest.raises(ValidationError):
        hydrate_miner_response_payload(
            {"text": "x" * 80_001},
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )


def test_hydrate_structured_output_and_citations() -> None:
    session_id = uuid4()
    source_text = "Primary source"
    response = hydrate_miner_response_payload(
        {
            "output": {"answer": [1, None, "  exact  "]},
            "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
        },
        query=Query(
            text="question",
            output_schema={
                "type": "object",
                "properties": {"answer": {"type": "array"}},
                "required": ["answer"],
            },
        ),
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert response.output == {"answer": [1, None, "  exact  "]}
    assert response.text is None
    assert response.citations == (
        AnswerCitation(
            url="https://example.com/source",
            note=f"[slice 0:{len(source_text)}]\n{source_text}",
            title="Example source",
        ),
    )


@pytest.mark.parametrize(
    ("query", "payload"),
    [
        (Query(text="question"), {"output": {"answer": 1}}),
        (Query(text="question", output_schema={}), {"text": "answer"}),
        (Query(text="question", output_schema={"type": "array"}), {"output": {"answer": 1}}),
    ],
)
def test_hydration_rejects_wrong_mode_and_schema_mismatch(query: Query, payload: object) -> None:
    with pytest.raises(MinerResponsePayloadError):
        hydrate_miner_response_payload(
            payload,
            query=query,
            session_id=uuid4(),
            receipt_log=InMemoryReceiptLog(),
        )
