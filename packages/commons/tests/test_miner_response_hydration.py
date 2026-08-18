from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.application.miner_response_hydration import (
    CitationSlice,
    MinerResponsePayloadError,
    materialize_citation_slices,
)
from harnyx_commons.application.miner_response_hydration import (
    hydrate_miner_response_payload as _hydrate_miner_response_payload,
)
from harnyx_commons.domain.miner_task import AnswerCitation, Query, ReferenceAnswer, Response
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolResult,
    ToolResultPolicy,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.tools.types import ToolName

_LEGACY_QUERY = Query(text="question")


def test_reference_answer_loads_legacy_persisted_citation_count_above_judge_cap() -> None:
    """Future failure: nullable slots must not tighten persisted ReferenceAnswer cardinality."""
    citation = {"url": "https://example.com/source", "note": "evidence"}

    reference = ReferenceAnswer.model_validate(
        {"text": "Legacy reference", "citations": [citation] * 201}
    )

    assert reference.citations is not None
    assert len(reference.citations) == 201


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


def _record_receipt(
    receipt_log: InMemoryReceiptLog,
    *,
    receipt_id: str,
    session_id: UUID,
    tool: ToolName = "search_web",
    outcome: ToolCallOutcome = ToolCallOutcome.OK,
    result_policy: ToolResultPolicy = ToolResultPolicy.REFERENCEABLE,
    results: tuple[ToolResult, ...] = (),
) -> None:
    receipt_log.record(
        ToolCall(
            receipt_id=receipt_id,
            session_id=session_id,
            uid=42,
            tool=tool,
            issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
            outcome=outcome,
            details=ToolCallDetails(
                request_hash=f"request-{receipt_id}",
                response_hash=f"response-{receipt_id}",
                result_policy=result_policy,
                results=results,
            ),
        )
    )


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


def test_public_slice_materializer_matches_official_hydration_for_raw_unicode_crlf_slices() -> None:
    """Future failure: reference authoring and miner hydration must share one raw projection rule."""
    session_id = uuid4()
    source_text = ("prefix\r\nCafé 雪 " + _source_text(130) + "\r\nsuffix " + _source_text(130))
    slices = (CitationSlice(0, 120), CitationSlice(140, 260))

    materialized = materialize_citation_slices(source_text, slices)
    hydrated = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": item.start, "end": item.end} for item in slices],
                }
            ],
        },
        session_id=session_id,
        receipt_log=_receipt_log_with_result(session_id=session_id, note=source_text),
    )

    assert hydrated.citations is not None
    assert hydrated.citations[0].note == materialized.text
    assert materialized.char_count == 240


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


def test_hydrate_miner_response_payload_preserves_duplicate_and_unresolved_positions() -> None:
    """Future failure: hydration must not renumber positional citation pointers."""
    session_id = uuid4()
    source_text = _source_text()
    receipt_log = _receipt_log_with_result(session_id=session_id, note=source_text)

    response = hydrate_miner_response_payload(
        {
            "text": "Answer",
            "citations": [
                {"receipt_id": "receipt-1", "result_id": "result-1"},
                {"receipt_id": "missing", "result_id": "result-1"},
                {"receipt_id": "receipt-1", "result_id": "result-1"},
            ],
        },
        session_id=session_id,
        receipt_log=receipt_log,
    )

    resolved = AnswerCitation(
        url="https://example.com/source",
        note=f"[slice 0:{len(source_text)}]\n{source_text}",
        title="Example source",
    )
    assert response == Response(text="Answer", citations=(resolved, None, resolved))


def test_hydrate_miner_response_payload_preserves_every_soft_unresolved_class_as_null() -> None:
    """Future failure: soft unresolved refs must not disappear or weaken later citations."""
    session_id = uuid4()
    source_text = _source_text()
    valid_result = SearchToolResult(
        index=0,
        result_id="result-1",
        url="https://example.com/source",
        note=source_text,
        title="Example source",
    )
    receipt_log = InMemoryReceiptLog()
    _record_receipt(
        receipt_log,
        receipt_id="wrong-session",
        session_id=uuid4(),
        results=(valid_result,),
    )
    _record_receipt(
        receipt_log,
        receipt_id="unsuccessful",
        session_id=session_id,
        outcome=ToolCallOutcome.PROVIDER_ERROR,
        results=(valid_result,),
    )
    _record_receipt(
        receipt_log,
        receipt_id="non-citation-tool",
        session_id=session_id,
        tool="llm_chat",
        results=(valid_result,),
    )
    _record_receipt(
        receipt_log,
        receipt_id="not-referenceable",
        session_id=session_id,
        result_policy=ToolResultPolicy.LOG_ONLY,
        results=(valid_result,),
    )
    _record_receipt(
        receipt_log,
        receipt_id="missing-result",
        session_id=session_id,
        results=(valid_result,),
    )
    _record_receipt(
        receipt_log,
        receipt_id="wrong-result-type",
        session_id=session_id,
        results=(ToolResult(index=0, result_id="result-1"),),
    )
    _record_receipt(
        receipt_log,
        receipt_id="valid",
        session_id=session_id,
        results=(valid_result,),
    )

    response = hydrate_miner_response_payload(
        {
            "text": "Answer [[8]]",
            "citations": [
                {"receipt_id": "missing-receipt", "result_id": "result-1"},
                {"receipt_id": "wrong-session", "result_id": "result-1"},
                {"receipt_id": "unsuccessful", "result_id": "result-1"},
                {"receipt_id": "non-citation-tool", "result_id": "result-1"},
                {"receipt_id": "not-referenceable", "result_id": "result-1"},
                {"receipt_id": "missing-result", "result_id": "absent"},
                {"receipt_id": "wrong-result-type", "result_id": "result-1"},
                {"receipt_id": "valid", "result_id": "result-1"},
            ],
        },
        session_id=session_id,
        receipt_log=receipt_log,
    )

    resolved = AnswerCitation(
        url="https://example.com/source",
        note=f"[slice 0:{len(source_text)}]\n{source_text}",
        title="Example source",
    )
    assert response == Response(
        text="Answer [[8]]",
        citations=(None, None, None, None, None, None, None, resolved),
    )


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
