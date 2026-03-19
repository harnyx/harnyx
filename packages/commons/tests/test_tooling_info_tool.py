from __future__ import annotations

import pytest

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.pricing import (
    MODEL_PRICING,
    SEARCH_AI_PER_REFERENCEABLE_RESULT_USD,
    SEARCH_PRICING,
)
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker, build_miner_sandbox_tool_invoker

pytestmark = pytest.mark.anyio("asyncio")


async def test_tooling_info_sandbox_builder_returns_pricing_metadata() -> None:
    invoker = build_miner_sandbox_tool_invoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert "search_repo" not in payload["tool_names"]
    assert "get_repo_file" not in payload["tool_names"]
    assert payload["pricing"]["search_web"]["usd_per_call"] == pytest.approx(SEARCH_PRICING["search_web"])
    assert payload["pricing"]["search_x"]["usd_per_call"] == pytest.approx(SEARCH_PRICING["search_x"])
    assert "search_repo" not in payload["pricing"]
    assert "get_repo_file" not in payload["pricing"]
    assert payload["pricing"]["search_ai"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_AI_PER_REFERENCEABLE_RESULT_USD
    )
    assert payload["pricing"]["search_items"]["kind"] == "flat_per_call"
    assert payload["pricing"]["search_items"]["usd_per_call"] == pytest.approx(0.0025)

    model_prices = payload["pricing"]["llm_chat"]["models"]
    assert model_prices["openai/gpt-oss-20b"]["input_per_million"] == pytest.approx(
        MODEL_PRICING["openai/gpt-oss-20b"].input_per_million
    )
    assert model_prices["openai/gpt-oss-120b"]["output_per_million"] == pytest.approx(
        MODEL_PRICING["openai/gpt-oss-120b"].output_per_million
    )


async def test_tooling_info_default_surface_matches_miner_contract() -> None:
    invoker = RuntimeToolInvoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert "search_repo" not in payload["tool_names"]
    assert "get_repo_file" not in payload["tool_names"]
    assert "search_repo" not in payload["pricing"]
    assert "get_repo_file" not in payload["pricing"]
