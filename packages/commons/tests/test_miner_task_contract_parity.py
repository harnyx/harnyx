from __future__ import annotations

from harnyx_commons.domain.miner_task import Query as CommonsQuery
from harnyx_commons.domain.miner_task import Response as CommonsResponse
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


def test_response_contract_matches_miner_sdk_boundary() -> None:
    assert CommonsResponse.model_json_schema() == MinerSdkResponse.model_json_schema()
    assert _relevant_model_config(CommonsResponse) == _relevant_model_config(MinerSdkResponse)
