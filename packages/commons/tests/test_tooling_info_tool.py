from __future__ import annotations

import pytest

from caster_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from caster_commons.llm.pricing import (
    MODEL_PRICING,
    SEARCH_AI_PER_REFERENCEABLE_RESULT_USD,
    SEARCH_PRICING,
)
from caster_commons.tools.runtime_invoker import RuntimeToolInvoker

pytestmark = pytest.mark.anyio("asyncio")


async def test_tooling_info_returns_pricing_metadata() -> None:
    invoker = RuntimeToolInvoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert payload["pricing"]["search_web"]["usd_per_call"] == pytest.approx(SEARCH_PRICING["search_web"])
    assert payload["pricing"]["search_x"]["usd_per_call"] == pytest.approx(SEARCH_PRICING["search_x"])
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
