from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import AnswerCitation, MinerTask, ReferenceAnswer
from harnyx_commons.domain.miner_task import Query as CommonsQuery
from harnyx_commons.domain.miner_task import Response as CommonsResponse
from harnyx_miner_sdk.query import CitationRef, CitationSlice
from harnyx_miner_sdk.query import Query as MinerSdkQuery
from harnyx_miner_sdk.query import Response as MinerSdkResponse


@pytest.mark.parametrize(
    "saved,fast,expected", [(300, False, 300), (300, True, 210), (200, True, 140), (100, True, 70)]
)
def test_saved_timeout_survives_serialization_without_fast_mode_reduction(saved, fast, expected):
    task = MinerTask(
        task_id=uuid4(),
        query=CommonsQuery(text="question", fast=fast),
        reference_answer=ReferenceAnswer(text="answer"),
        timeout_seconds=saved,
    )
    restored = MinerTask.model_validate_json(task.model_dump_json())
    assert restored.timeout_seconds == saved
    assert restored.effective_timeout_seconds() == expected
    legacy = task.model_dump(mode="json")
    legacy.pop("timeout_seconds")
    assert MinerTask.model_validate_json(json.dumps(legacy)).timeout_seconds == 300
    for invalid in (0, -1, True, 1.5):
        payload = task.model_dump(mode="json")
        payload["timeout_seconds"] = invalid
        with pytest.raises(ValidationError):
            MinerTask.model_validate_json(json.dumps(payload))


def _relevant_model_config(model: type[object]) -> tuple[object, object, object, object]:
    config = model.model_config
    return (
        config.get("extra"),
        config.get("frozen"),
        config.get("strict"),
        config.get("str_strip_whitespace"),
    )


def test_query_contract_matches_miner_sdk_boundary() -> None:
    assert CommonsQuery.model_json_schema() == MinerSdkQuery.model_json_schema()
    assert _relevant_model_config(CommonsQuery) == _relevant_model_config(MinerSdkQuery)
    assert CommonsQuery is MinerSdkQuery
    schema = {"type": "string", "const": "  exact  "}
    assert CommonsQuery(text=" question ", output_schema=schema).output_schema == schema
    assert CommonsQuery(text="question").fast is False
    assert CommonsQuery(text="question", fast=True) == MinerSdkQuery(text="question", fast=True)

    for model in (CommonsQuery, MinerSdkQuery):
        with pytest.raises(ValidationError):
            model.model_validate({"text": "question", "fast": 1})


def test_response_contract_matches_miner_sdk_boundary() -> None:
    commons_schema = CommonsResponse.model_json_schema()
    sdk_schema = MinerSdkResponse.model_json_schema()

    assert commons_schema != sdk_schema
    assert _relevant_model_config(CommonsResponse) == _relevant_model_config(MinerSdkResponse)
    assert CommonsResponse(text="hello", citations=(AnswerCitation(url="https://example.com"),))
    assert MinerSdkResponse(
        text="hello",
        citations=[CitationRef(receipt_id="receipt-1", result_id="result-1")],
    )
    assert MinerSdkResponse(
        text="hello",
        citations=[
            CitationRef(
                receipt_id="receipt-1",
                result_id="result-1",
                slices=[CitationSlice(start=0, end=120)],
            )
        ],
    )


def test_response_contracts_share_answer_modes_with_distinct_citation_types() -> None:
    assert CommonsResponse(output={"answer": [1, None]}).answer_text == '{"answer":[1,null]}'
    assert MinerSdkResponse(output={"answer": [1, None]})
    assert CommonsResponse(output=None).answer_text == "null"
    assert CommonsResponse(output=None).model_dump(mode="json", exclude_none=True) == {"output": None}
    assert MinerSdkResponse(output=None).model_dump(mode="json", exclude_none=True) == {"output": None}
    assert CommonsResponse(text="answer", output=None).model_dump(mode="json", exclude_none=True) == {"text": "answer"}
    assert MinerSdkResponse(text="answer", output=None).model_dump(mode="json", exclude_none=True) == {"text": "answer"}


def test_response_contracts_share_optional_note_semantics() -> None:
    assert CommonsResponse(text="hello", note="  qualification  ").note == "qualification"
    assert MinerSdkResponse(text="hello", note="  qualification  ").note == "qualification"
    assert CommonsResponse(text="hello").model_dump(mode="json") == {"text": "hello", "citations": None}

    for model in (CommonsResponse, MinerSdkResponse):
        with pytest.raises(ValidationError):
            model(text="hello", note="   ")
        with pytest.raises(ValidationError):
            model(text="hello", note="x" * 80_001)
