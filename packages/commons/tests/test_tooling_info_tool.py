from __future__ import annotations

import pytest

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.pricing import (
    MINER_TOOL_EMBEDDING_PRICING,
    MINER_TOOL_LLM_PRICING,
    MODEL_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    price_llm,
    price_miner_llm,
    price_parallel_search,
)
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.llm.tool_models import (
    ALLOWED_TOOL_MODELS,
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    MinerSelectedLlmProviderName,
    parse_tool_model,
)
from harnyx_commons.tools.embedding_models import (
    MINER_SELECTED_EMBEDDING_PROVIDER_MODELS,
    QWEN3_CHUTES_EMBEDDING_MODEL,
    QWEN3_OPENROUTER_EMBEDDING_MODEL,
)
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker, build_miner_sandbox_tool_invoker

pytestmark = pytest.mark.anyio("asyncio")


def test_tool_model_pricing_covers_every_allowed_tool_model() -> None:
    assert set(MODEL_PRICING) == set(ALLOWED_TOOL_MODELS)
    assert set(MINER_TOOL_LLM_PRICING) == set(MINER_SELECTED_LLM_PROVIDER_MODELS)
    for provider, models in MINER_SELECTED_LLM_PROVIDER_MODELS.items():
        assert set(MINER_TOOL_LLM_PRICING[provider]) == set(models)
    assert set(MINER_TOOL_EMBEDDING_PRICING) == set(MINER_SELECTED_EMBEDDING_PROVIDER_MODELS)
    for provider, models in MINER_SELECTED_EMBEDDING_PROVIDER_MODELS.items():
            assert set(MINER_TOOL_EMBEDDING_PRICING[provider]) == set(models)


@pytest.mark.parametrize(
    ("provider", "model", "input_rate", "output_rate"),
    (
        ("chutes", "deepseek-ai/DeepSeek-V3.2-TEE", 1.00, 1.00),
        ("chutes", "Qwen/Qwen3.6-27B-TEE", 0.30, 2.00),
        ("chutes", "Qwen/Qwen3.8-27B-TEE", 0.40, 3.00),
        ("chutes", "google/gemma-4-31B-turbo-TEE", 0.12, 0.37),
        ("chutes", "zai-org/GLM-5.2-TEE", 1.40, 4.40),
        ("chutes", "Qwen/Qwen3.5-397B-A17B-TEE", 0.45, 3.00),
        ("openrouter", "openai/gpt-oss-120b", 0.037, 0.17),
        ("openrouter", "deepseek/deepseek-v3.2", 0.269, 0.40),
        ("openrouter", "qwen/qwen3.6-27b", 0.30, 2.00),
        ("openrouter", "qwen/qwen3.8-27b", 0.45, 3.20),
        ("openrouter", "google/gemma-4-31b-it", 0.14, 0.40),
        ("openrouter", "deepseek/deepseek-v4-flash", 0.14, 0.28),
        ("openrouter", "deepseek/deepseek-v4-flash-0731", 0.09, 0.18),
        ("openrouter", "deepseek/deepseek-v4-pro", 0.435, 0.87),
        ("openrouter", "z-ai/glm-5.2", 0.8008, 2.5168),
        ("openrouter", "thinkingmachines/inkling", 1.00, 4.05),
        ("openrouter", "qwen/qwen3.5-397b-a17b", 0.39, 2.34),
        ("openrouter", "meta/muse-glimmer-30b", 0.35, 1.50),
        ("ai_gateway", "openai/gpt-oss-20b", 0.05, 0.20),
        ("ai_gateway", "zai/glm-4.7", 0.60, 2.20),
        ("ai_gateway", "deepseek/deepseek-v4-flash", 0.14, 0.28),
        ("ai_gateway", "deepseek/deepseek-v4-flash-0731", 0.13, 0.26),
        ("ai_gateway", "deepseek/deepseek-v4-pro", 0.435, 0.87),
        ("ai_gateway", "meta/muse-glimmer-30b", 0.35, 1.50),
        ("ai_gateway", "alibaba/qwen3.8-27b", 0.10, 0.40),
    ),
)
def test_miner_model_reference_rates_match_approved_snapshot(
    provider: MinerSelectedLlmProviderName,
    model: str,
    input_rate: float,
    output_rate: float,
) -> None:
    pricing = MINER_TOOL_LLM_PRICING[provider][model]

    assert pricing.input_per_million == pytest.approx(input_rate)
    assert pricing.output_per_million == pytest.approx(output_rate)


async def test_tooling_info_sandbox_builder_returns_pricing_metadata() -> None:
    invoker = build_miner_sandbox_tool_invoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert "search_repo" not in payload["tool_names"]
    assert "get_repo_file" not in payload["tool_names"]
    assert "search_ai" not in payload["tool_names"]
    assert payload["pricing"]["search_web"]["kind"] == "per_referenceable_result"
    assert payload["pricing"]["fetch_page"]["kind"] == "per_referenceable_result"
    assert payload["pricing"]["search_web"]["settlement_order"] == [
        "provider_returned",
        "static_pricing",
    ]
    assert payload["pricing"]["fetch_page"]["settlement_order"] == [
        "provider_returned",
        "static_pricing",
    ]
    assert payload["pricing"]["search_web"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_PRICING_PER_REFERENCEABLE_RESULT["search_web"]
    )
    assert payload["pricing"]["fetch_page"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_PRICING_PER_REFERENCEABLE_RESULT["fetch_page"]
    )
    assert "search_repo" not in payload["pricing"]
    assert "get_repo_file" not in payload["pricing"]
    assert "search_ai" not in payload["pricing"]
    assert "search_items" not in payload["tool_names"]
    assert "search_items" not in payload["pricing"]

    with pytest.raises(LookupError, match="search_ai.*not registered"):
        await invoker.invoke("search_ai", args=(), kwargs={})

    assert "allowed_tool_models" not in payload
    assert "models" not in payload["pricing"]["llm_chat"]
    assert payload["pricing"]["llm_chat"]["settlement_order"] == [
        "provider_returned",
        "cached_provider_pricing",
        "static_pricing",
    ]
    model_prices = payload["pricing"]["llm_chat"]["provider_models"]
    provider_models = payload["allowed_llm_provider_models"]
    assert provider_models == {
        provider: list(models)
        for provider, models in MINER_SELECTED_LLM_PROVIDER_MODELS.items()
    }
    assert set(model_prices) == set(provider_models)
    for provider, pricing_by_model in MINER_TOOL_LLM_PRICING.items():
        assert set(model_prices[provider]) == set(provider_models[provider])
        for model, rates in pricing_by_model.items():
            assert model_prices[provider][model]["input_per_million"] == pytest.approx(rates.input_per_million)
            assert model_prices[provider][model]["output_per_million"] == pytest.approx(rates.output_per_million)
            assert model_prices[provider][model]["reasoning_per_million"] == pytest.approx(
                rates.billable_reasoning_per_million
            )
    assert "openai/gpt-oss-20b-TEE" not in provider_models["openrouter"]
    assert "openai/gpt-oss-120b-TEE" not in provider_models["openrouter"]
    assert "openai/gpt-oss-20b" in provider_models["openrouter"]
    assert model_prices["openrouter"]["openai/gpt-oss-20b"]["input_per_million"] == pytest.approx(0.03)
    assert model_prices["openrouter"]["openai/gpt-oss-20b"]["output_per_million"] == pytest.approx(0.14)
    assert model_prices["openrouter"]["openai/gpt-oss-20b"]["reasoning_per_million"] == pytest.approx(0.14)
    assert "openai/gpt-oss-120b" in provider_models["openrouter"]
    assert model_prices["openrouter"]["openai/gpt-oss-120b"]["input_per_million"] == pytest.approx(0.037)
    assert model_prices["openrouter"]["openai/gpt-oss-120b"]["output_per_million"] == pytest.approx(0.17)
    assert model_prices["openrouter"]["openai/gpt-oss-120b"]["reasoning_per_million"] == pytest.approx(0.17)
    assert "zai-org/GLM-5-TEE" not in provider_models["chutes"]
    assert "zai-org/GLM-5-TEE" not in model_prices["chutes"]
    assert "moonshotai/Kimi-K2.6-TEE" in provider_models["chutes"]
    assert model_prices["chutes"]["moonshotai/Kimi-K2.6-TEE"]["input_per_million"] == pytest.approx(0.66)
    assert model_prices["chutes"]["moonshotai/Kimi-K2.6-TEE"]["output_per_million"] == pytest.approx(3.50)
    assert model_prices["chutes"]["moonshotai/Kimi-K2.6-TEE"]["reasoning_per_million"] == pytest.approx(3.50)
    assert "Qwen/Qwen3-Next-80B-A3B-Instruct" not in provider_models["chutes"]
    assert "Qwen/Qwen3-Next-80B-A3B-Instruct" not in model_prices["chutes"]
    assert "Qwen/Qwen3.6-27B-TEE" in provider_models["chutes"]
    assert model_prices["chutes"]["Qwen/Qwen3.6-27B-TEE"]["input_per_million"] == pytest.approx(0.30)
    assert model_prices["chutes"]["Qwen/Qwen3.6-27B-TEE"]["output_per_million"] == pytest.approx(2.00)
    assert model_prices["chutes"]["Qwen/Qwen3.6-27B-TEE"]["reasoning_per_million"] == pytest.approx(2.00)
    assert "Qwen/Qwen3.8-27B-TEE" in provider_models["chutes"]
    assert model_prices["chutes"]["Qwen/Qwen3.8-27B-TEE"]["input_per_million"] == pytest.approx(0.40)
    assert model_prices["chutes"]["Qwen/Qwen3.8-27B-TEE"]["output_per_million"] == pytest.approx(3.00)
    assert model_prices["chutes"]["Qwen/Qwen3.8-27B-TEE"]["reasoning_per_million"] == pytest.approx(3.00)
    assert "deepseek-ai/DeepSeek-V3.1-TEE" not in provider_models["chutes"]
    assert "deepseek-ai/DeepSeek-V3.1-TEE" not in model_prices["chutes"]
    assert "moonshotai/Kimi-K2.5-TEE" not in provider_models["chutes"]
    assert "moonshotai/Kimi-K2.5-TEE" not in model_prices["chutes"]
    assert "google/gemma-4-31B-it" not in provider_models["chutes"]
    assert "google/gemma-4-31B-it" not in model_prices["chutes"]
    assert "google/gemma-4-31B-turbo-TEE" in provider_models["chutes"]
    assert model_prices["chutes"]["google/gemma-4-31B-turbo-TEE"]["input_per_million"] == pytest.approx(0.12)
    assert model_prices["chutes"]["google/gemma-4-31B-turbo-TEE"]["output_per_million"] == pytest.approx(0.37)
    assert model_prices["chutes"]["google/gemma-4-31B-turbo-TEE"]["reasoning_per_million"] == pytest.approx(0.37)
    model = "deepseek-ai/DeepSeek-V3.2-TEE"
    assert model in provider_models["chutes"]
    assert "deepseek/deepseek-v3.2" in provider_models["openrouter"]
    assert model_prices["openrouter"]["deepseek/deepseek-v3.2"]["input_per_million"] == pytest.approx(
        0.269
    )
    assert provider_models["ai_gateway"] == list(MINER_SELECTED_LLM_PROVIDER_MODELS["ai_gateway"])
    assert model_prices["ai_gateway"]["thinkingmachines/inkling"]["input_per_million"] == pytest.approx(1.00)
    assert model_prices["ai_gateway"]["thinkingmachines/inkling"]["output_per_million"] == pytest.approx(4.05)
    assert model_prices["ai_gateway"]["zai/glm-5.2-fast"]["input_per_million"] == pytest.approx(2.10)
    assert model_prices["ai_gateway"]["zai/glm-5.2-fast"]["output_per_million"] == pytest.approx(6.60)
    assert model_prices["ai_gateway"]["openai/gpt-oss-20b"]["input_per_million"] == pytest.approx(0.05)
    assert model_prices["ai_gateway"]["openai/gpt-oss-20b"]["output_per_million"] == pytest.approx(0.20)
    assert model_prices["ai_gateway"]["zai/glm-4.7"]["input_per_million"] == pytest.approx(0.60)
    assert model_prices["ai_gateway"]["zai/glm-4.7"]["output_per_million"] == pytest.approx(2.20)
    assert model_prices["ai_gateway"]["google/gemma-4-31b-it"]["input_per_million"] == pytest.approx(0.14)
    assert model_prices["ai_gateway"]["google/gemma-4-31b-it"]["output_per_million"] == pytest.approx(0.40)
    assert model_prices["ai_gateway"]["openai/gpt-oss-120b"]["input_per_million"] == pytest.approx(0.10)
    assert model_prices["ai_gateway"]["openai/gpt-oss-120b"]["output_per_million"] == pytest.approx(0.50)
    assert "alibaba/qwen3.7-plus" not in provider_models["ai_gateway"]
    assert "alibaba/qwen3.7-plus" not in model_prices["ai_gateway"]
    assert model_prices["ai_gateway"]["minimax/minimax-m2.7"]["input_per_million"] == pytest.approx(0.30)
    assert model_prices["ai_gateway"]["minimax/minimax-m2.7"]["output_per_million"] == pytest.approx(1.20)
    assert model_prices["ai_gateway"]["zai/glm-4.7-flash"]["input_per_million"] == pytest.approx(0.07)
    assert model_prices["ai_gateway"]["zai/glm-4.7-flash"]["output_per_million"] == pytest.approx(0.40)
    assert model_prices["ai_gateway"]["alibaba/qwen3.8-27b"]["input_per_million"] == pytest.approx(0.10)
    assert model_prices["ai_gateway"]["alibaba/qwen3.8-27b"]["output_per_million"] == pytest.approx(0.40)

    assert payload["pricing"]["embed_text"]["kind"] == "provider_specific_static"
    embedding_provider_models = payload["allowed_embedding_provider_models"]
    embedding_prices = payload["pricing"]["embed_text"]["provider_models"]
    assert embedding_provider_models == {
        provider: list(models)
        for provider, models in MINER_SELECTED_EMBEDDING_PROVIDER_MODELS.items()
    }
    assert set(embedding_prices) == set(embedding_provider_models)
    for provider, pricing_by_model in MINER_TOOL_EMBEDDING_PRICING.items():
        assert set(embedding_prices[provider]) == set(embedding_provider_models[provider])
        for embedding_model, rates in pricing_by_model.items():
            if rates.input_per_million is not None:
                assert embedding_prices[provider][embedding_model]["input_per_million"] == pytest.approx(
                    rates.input_per_million
                )
            if rates.usd_per_second is not None:
                assert embedding_prices[provider][embedding_model]["usd_per_second"] == pytest.approx(
                    rates.usd_per_second
                )
    assert embedding_provider_models == {
        "chutes": [QWEN3_CHUTES_EMBEDDING_MODEL],
        "openrouter": [QWEN3_OPENROUTER_EMBEDDING_MODEL],
    }
    assert embedding_prices["chutes"][QWEN3_CHUTES_EMBEDDING_MODEL]["usd_per_second"] == pytest.approx(0.0005)
    assert "input_per_million" not in embedding_prices["chutes"][QWEN3_CHUTES_EMBEDDING_MODEL]
    assert embedding_prices["openrouter"][QWEN3_OPENROUTER_EMBEDDING_MODEL]["input_per_million"] == pytest.approx(
        0.01
    )
    assert "usd_per_second" not in embedding_prices["openrouter"][QWEN3_OPENROUTER_EMBEDDING_MODEL]


async def test_tooling_info_default_surface_matches_miner_contract() -> None:
    invoker = RuntimeToolInvoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert "search_repo" not in payload["tool_names"]
    assert "get_repo_file" not in payload["tool_names"]
    assert "search_items" not in payload["tool_names"]
    assert "search_repo" not in payload["pricing"]
    assert "get_repo_file" not in payload["pricing"]
    assert "search_items" not in payload["pricing"]


def test_zero_reasoning_price_falls_back_to_output_price() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        reasoning_tokens=1_000_000,
    )

    assert price_llm(parse_tool_model("deepseek-ai/DeepSeek-V3.2-TEE"), usage) == pytest.approx(1.12)
    assert price_llm(parse_tool_model("zai-org/GLM-5-TEE"), usage) == pytest.approx(6.05)
    assert price_llm(parse_tool_model("Qwen/Qwen3.6-27B-TEE"), usage) == pytest.approx(4.50)
    assert price_llm(parse_tool_model("google/gemma-4-31B-turbo-TEE"), usage) == pytest.approx(0.89)
    assert price_llm(parse_tool_model("openai/gpt-oss-20b"), usage) == pytest.approx(0.31)
    assert price_llm(parse_tool_model("openai/gpt-oss-120b"), usage) == pytest.approx(0.399)
    assert price_miner_llm("openrouter", "deepseek/deepseek-v3.2", usage) == pytest.approx(1.069)
    assert price_miner_llm("chutes", "deepseek-ai/DeepSeek-V3.2-TEE", usage) == pytest.approx(3.00)
    assert price_miner_llm("ai_gateway", "openai/gpt-oss-20b", usage) == pytest.approx(0.45)
    assert price_miner_llm("ai_gateway", "zai/glm-5.2-fast", usage) == pytest.approx(15.30)


@pytest.mark.parametrize(
    ("billable_results", "expected_cost"),
    ((0, 0.005), (1, 0.005), (10, 0.005), (11, 0.006), (25, 0.02)),
)
def test_parallel_search_actual_pricing_uses_base_price_for_up_to_ten_results(
    billable_results: int,
    expected_cost: float,
) -> None:
    assert price_parallel_search(billable_results=billable_results) == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    ("billable_results", "expected_cost"),
    ((0, 0.001), (1, 0.001), (10, 0.001), (11, 0.002), (25, 0.016)),
)
def test_parallel_turbo_search_actual_pricing_uses_lower_base_price(
    billable_results: int,
    expected_cost: float,
) -> None:
    assert price_parallel_search(billable_results=billable_results, mode="turbo") == pytest.approx(
        expected_cost
    )


@pytest.mark.parametrize(
    "model",
    (
        "openai/gpt-oss-20b-TEE",
        "openai/gpt-oss-120b-TEE",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "deepseek-ai/DeepSeek-V3.1-TEE",
    ),
)
def test_retired_tool_models_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model(model)


def test_old_gemma_cloud_run_tool_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model("google/gemma-4-31B-it")
