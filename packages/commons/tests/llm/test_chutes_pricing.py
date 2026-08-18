from __future__ import annotations

import pytest

from harnyx_commons.llm.pricing import MINER_TOOL_LLM_PRICING, ModelPricing, price_static_llm_model
from harnyx_commons.llm.provider_types import CHUTES_PROVIDER
from harnyx_commons.llm.providers.chutes_pricing import CHUTES_STATIC_PRICING, ChutesModelPricingCache
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


async def test_chutes_pricing_cache_prices_kimi_validator_judge_model() -> None:
    cache = ChutesModelPricingCache()
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=2_000,
        reasoning_tokens=3_000,
        total_tokens=6_000,
    )

    actual_cost = await cache.price(model="moonshotai/Kimi-K2.6-TEE", usage=usage)

    assert "moonshotai/Kimi-K2.6-TEE" in CHUTES_STATIC_PRICING
    assert actual_cost.cost_usd == pytest.approx(0.01816)
    assert actual_cost.provider == "chutes"
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"
    assert actual_cost.evidence["reasoning_tokens"] == 3_000


async def test_chutes_static_pricing_does_not_use_generic_reference_rates() -> None:
    cache = ChutesModelPricingCache()
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    assert "moonshotai/Kimi-K3-TEE" in CHUTES_STATIC_PRICING
    with pytest.raises(KeyError, match="moonshotai/Kimi-K2.5-TEE"):
        await cache.price(model="moonshotai/Kimi-K2.5-TEE", usage=usage)


async def test_chutes_pricing_cache_keeps_unavailable_reasoning_tokens_in_evidence() -> None:
    cache = ChutesModelPricingCache()
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=2_000,
        reasoning_tokens=None,
        total_tokens=3_000,
    )

    actual_cost = await cache.price(model="moonshotai/Kimi-K2.6-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.00766)
    assert actual_cost.evidence["reasoning_tokens"] is None


def test_static_model_pricing_includes_validator_judge_models() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=2_000,
        reasoning_tokens=3_000,
        total_tokens=6_000,
    )

    assert price_static_llm_model("moonshotai/Kimi-K2.5-TEE", usage) == pytest.approx(0.01044)
    assert price_static_llm_model("moonshotai/Kimi-K2.6-TEE", usage) == pytest.approx(0.01816)
    assert price_static_llm_model(
        "deepseek-ai/DeepSeek-V4-Flash-0731-TEE", usage
    ) == pytest.approx(0.00154)
    assert price_static_llm_model("moonshotai/Kimi-K3-TEE", usage) == pytest.approx(0.078)


def test_chutes_static_pricing_uses_miner_advertised_rates_for_allowed_models() -> None:
    for model, pricing in MINER_TOOL_LLM_PRICING[CHUTES_PROVIDER].items():
        assert CHUTES_STATIC_PRICING[model] == pricing


async def test_chutes_pricing_cache_falls_back_to_hard_coded_rates_when_model_missing() -> None:
    cache = ChutesModelPricingCache(cached_pricing={"other/model": ModelPricing(1.0, 1.0, 0.0)})
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    actual_cost = await cache.price(model="google/gemma-4-31B-turbo-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.00086)
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"


async def test_chutes_pricing_cache_updated_empty_snapshot_uses_fallback_without_live_fetch() -> None:
    cache = ChutesModelPricingCache()
    cache.update_snapshot({})
    usage = LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000)

    actual_cost = await cache.price(model="google/gemma-4-31B-turbo-TEE", usage=usage)

    assert actual_cost.cost_usd == pytest.approx(0.00086)


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate"),
    (
        ("deepseek-ai/DeepSeek-V4-Flash-0731-TEE", 0.14, 0.28),
        ("moonshotai/Kimi-K3-TEE", 3.00, 15.00),
        ("moonshotai/Kimi-K2.6-TEE", 0.66, 3.50),
        ("zai-org/GLM-5.1-TEE", 0.98, 3.08),
        ("zai-org/GLM-5.2-TEE", 1.40, 4.40),
        ("Qwen/Qwen3.5-397B-A17B-TEE", 0.45, 3.00),
        ("Qwen/Qwen3.8-27B-TEE", 0.40, 3.00),
    ),
)
async def test_new_chutes_models_settle_from_approved_static_rates(
    model: str,
    input_rate: float,
    output_rate: float,
) -> None:
    usage = LlmUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)

    actual_cost = await ChutesModelPricingCache().price(model=model, usage=usage)

    assert actual_cost.cost_usd == pytest.approx(input_rate + output_rate)
    assert actual_cost.evidence["model"] == model
    assert actual_cost.evidence["input_per_million"] == pytest.approx(input_rate)
    assert actual_cost.evidence["output_per_million"] == pytest.approx(output_rate)
    assert actual_cost.evidence["settlement_source"] == "static_pricing"
    assert actual_cost.evidence["pricing_origin"] == "chutes_repo_rates"
