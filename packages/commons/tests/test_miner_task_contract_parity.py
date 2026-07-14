from __future__ import annotations

from harnyx_commons.domain.miner_task import AnswerCitation
from harnyx_commons.domain.miner_task import Query as CommonsQuery
from harnyx_commons.domain.miner_task import Response as CommonsResponse
from harnyx_miner_sdk.query import CitationRef, CitationSlice
from harnyx_miner_sdk.query import Query as MinerSdkQuery
from harnyx_miner_sdk.query import Response as MinerSdkResponse


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
