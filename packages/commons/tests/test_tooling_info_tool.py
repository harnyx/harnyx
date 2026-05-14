from __future__ import annotations

import pytest

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.pricing import (
    MODEL_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    price_llm,
    price_parallel_search,
)
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.llm.tool_models import ALLOWED_TOOL_MODELS, parse_tool_model
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker, build_miner_sandbox_tool_invoker

pytestmark = pytest.mark.anyio("asyncio")


def test_tool_model_pricing_covers_every_allowed_tool_model() -> None:
    assert set(MODEL_PRICING) == set(ALLOWED_TOOL_MODELS)


async def test_tooling_info_sandbox_builder_returns_pricing_metadata() -> None:
    invoker = build_miner_sandbox_tool_invoker(InMemoryReceiptLog())

    payload = await invoker.invoke("tooling_info", args=(), kwargs={})

    assert "search_repo" not in payload["tool_names"]
    assert "get_repo_file" not in payload["tool_names"]
    assert payload["pricing"]["search_web"]["kind"] == "per_referenceable_result"
    assert payload["pricing"]["fetch_page"]["kind"] == "per_referenceable_result"
    assert payload["pricing"]["search_ai"]["kind"] == "per_referenceable_result"
    assert payload["pricing"]["search_web"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_PRICING_PER_REFERENCEABLE_RESULT["search_web"]
    )
    assert payload["pricing"]["fetch_page"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_PRICING_PER_REFERENCEABLE_RESULT["fetch_page"]
    )
    assert "search_repo" not in payload["pricing"]
    assert "get_repo_file" not in payload["pricing"]
    assert payload["pricing"]["search_ai"]["usd_per_referenceable_result"] == pytest.approx(
        SEARCH_PRICING_PER_REFERENCEABLE_RESULT["search_ai"]
    )
    assert "search_items" not in payload["tool_names"]
    assert "search_items" not in payload["pricing"]

    model_prices = payload["pricing"]["llm_chat"]["models"]
    assert "openai/gpt-oss-20b-TEE" not in payload["allowed_tool_models"]
    assert "openai/gpt-oss-20b-TEE" not in model_prices
    assert "openai/gpt-oss-120b-TEE" not in payload["allowed_tool_models"]
    assert "openai/gpt-oss-120b-TEE" not in model_prices
    assert "openai/gpt-oss-20b" in payload["allowed_tool_models"]
    assert model_prices["openai/gpt-oss-20b"]["input_per_million"] == pytest.approx(0.03)
    assert model_prices["openai/gpt-oss-20b"]["output_per_million"] == pytest.approx(0.14)
    assert model_prices["openai/gpt-oss-20b"]["reasoning_per_million"] == pytest.approx(0.14)
    assert "openai/gpt-oss-120b" in payload["allowed_tool_models"]
    assert model_prices["openai/gpt-oss-120b"]["input_per_million"] == pytest.approx(0.039)
    assert model_prices["openai/gpt-oss-120b"]["output_per_million"] == pytest.approx(0.18)
    assert model_prices["openai/gpt-oss-120b"]["reasoning_per_million"] == pytest.approx(0.18)
    assert "zai-org/GLM-5-TEE" in payload["allowed_tool_models"]
    assert model_prices["zai-org/GLM-5-TEE"]["input_per_million"] == pytest.approx(
        MODEL_PRICING["zai-org/GLM-5-TEE"].input_per_million
    )
    assert model_prices["zai-org/GLM-5-TEE"]["output_per_million"] == pytest.approx(
        MODEL_PRICING["zai-org/GLM-5-TEE"].output_per_million
    )
    assert model_prices["zai-org/GLM-5-TEE"]["reasoning_per_million"] == pytest.approx(
        MODEL_PRICING["zai-org/GLM-5-TEE"].billable_reasoning_per_million
    )
    assert "Qwen/Qwen3-Next-80B-A3B-Instruct" in payload["allowed_tool_models"]
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["input_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].input_per_million
    )
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["output_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].output_per_million
    )
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["reasoning_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].billable_reasoning_per_million
    )
    assert "Qwen/Qwen3.6-27B-TEE" in payload["allowed_tool_models"]
    assert model_prices["Qwen/Qwen3.6-27B-TEE"]["input_per_million"] == pytest.approx(0.50)
    assert model_prices["Qwen/Qwen3.6-27B-TEE"]["output_per_million"] == pytest.approx(2.00)
    assert model_prices["Qwen/Qwen3.6-27B-TEE"]["reasoning_per_million"] == pytest.approx(2.00)
    assert "google/gemma-4-31B-it" not in payload["allowed_tool_models"]
    assert "google/gemma-4-31B-it" not in model_prices
    assert "google/gemma-4-31B-turbo-TEE" in payload["allowed_tool_models"]
    assert model_prices["google/gemma-4-31B-turbo-TEE"]["input_per_million"] == pytest.approx(0.13)
    assert model_prices["google/gemma-4-31B-turbo-TEE"]["output_per_million"] == pytest.approx(0.38)
    assert model_prices["google/gemma-4-31B-turbo-TEE"]["reasoning_per_million"] == pytest.approx(0.38)
    for model in ("deepseek-ai/DeepSeek-V3.1-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"):
        assert model in payload["allowed_tool_models"]
        assert MODEL_PRICING[model].reasoning_per_million == pytest.approx(0.0)
        assert model_prices[model]["input_per_million"] == pytest.approx(MODEL_PRICING[model].input_per_million)
        assert model_prices[model]["output_per_million"] == pytest.approx(MODEL_PRICING[model].output_per_million)
        assert model_prices[model]["reasoning_per_million"] == pytest.approx(
            MODEL_PRICING[model].billable_reasoning_per_million
        )


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

    assert price_llm(parse_tool_model("Qwen/Qwen3-Next-80B-A3B-Instruct"), usage) == pytest.approx(1.70)
    assert price_llm(parse_tool_model("deepseek-ai/DeepSeek-V3.1-TEE"), usage) == pytest.approx(2.27)
    assert price_llm(parse_tool_model("deepseek-ai/DeepSeek-V3.2-TEE"), usage) == pytest.approx(1.12)
    assert price_llm(parse_tool_model("zai-org/GLM-5-TEE"), usage) == pytest.approx(6.05)
    assert price_llm(parse_tool_model("Qwen/Qwen3.6-27B-TEE"), usage) == pytest.approx(4.50)
    assert price_llm(parse_tool_model("google/gemma-4-31B-turbo-TEE"), usage) == pytest.approx(0.89)
    assert price_llm(parse_tool_model("openai/gpt-oss-20b"), usage) == pytest.approx(0.31)
    assert price_llm(parse_tool_model("openai/gpt-oss-120b"), usage) == pytest.approx(0.399)


@pytest.mark.parametrize(
    ("requested_results", "expected_cost"),
    ((None, 0.005), (1, 0.005), (10, 0.005), (11, 0.006), (25, 0.02)),
)
def test_parallel_search_actual_pricing_uses_base_price_for_up_to_ten_results(
    requested_results: int | None,
    expected_cost: float,
) -> None:
    assert price_parallel_search(requested_results=requested_results) == pytest.approx(expected_cost)


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b-TEE", "openai/gpt-oss-120b-TEE"))
def test_retired_openai_gpt_oss_tool_models_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model(model)


def test_old_gemma_cloud_run_tool_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model("google/gemma-4-31B-it")
