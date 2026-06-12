from __future__ import annotations

import pytest

from harnyx_commons.llm.pricing import ModelPricing
from harnyx_commons.llm.providers.chutes_pricing import ChutesModelPricingCache
from harnyx_commons.llm.schema import LlmUsage

pytestmark = pytest.mark.anyio("asyncio")


async def test_chutes_pricing_cache_uses_cached_model_rate() -> None:
    cache = ChutesModelPricingCache(
        cached_pricing={"deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.10, 0.20, 0.0)}
    )
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    first = await cache.price(model="deepseek-ai/DeepSeek-V3.2-TEE", usage=usage)
    second = await cache.price(model="deepseek-ai/DeepSeek-V3.2-TEE", usage=usage)

    assert first.cost_usd == pytest.approx(0.0005)
    assert first.evidence["settlement_source"] == "cached_provider_pricing"
    assert first.evidence["pricing_origin"] == "chutes_live_snapshot"
    assert second.cost_usd == pytest.approx(0.0005)
    assert second.evidence["settlement_source"] == "cached_provider_pricing"
    assert second.evidence["pricing_origin"] == "chutes_live_snapshot"


async def test_chutes_pricing_cache_falls_back_to_hard_coded_rates_when_cache_unavailable() -> None:
    cache = ChutesModelPricingCache()
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    actual_cost = await cache.price(model="Qwen/Qwen3.6-27B-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.0043)
    assert actual_cost.provider == "chutes"
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"


async def test_chutes_pricing_cache_falls_back_to_hard_coded_rates_when_model_missing() -> None:
    cache = ChutesModelPricingCache(cached_pricing={"other/model": ModelPricing(1.0, 1.0, 0.0)})
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    actual_cost = await cache.price(model="google/gemma-4-31B-turbo-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.00099)
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"


async def test_chutes_pricing_cache_updated_empty_snapshot_uses_fallback_without_live_fetch() -> None:
    cache = ChutesModelPricingCache()
    cache.update_snapshot({})
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    actual_cost = await cache.price(model="google/gemma-4-31B-turbo-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.00099)
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"
