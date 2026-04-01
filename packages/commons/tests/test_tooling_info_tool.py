from __future__ import annotations

import pytest

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.pricing import (
    MODEL_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    parse_tool_model,
    price_llm,
)
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker, build_miner_sandbox_tool_invoker

pytestmark = pytest.mark.anyio("asyncio")


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
    assert model_prices["openai/gpt-oss-20b-TEE"]["input_per_million"] == pytest.approx(0.03)
    assert model_prices["openai/gpt-oss-20b-TEE"]["output_per_million"] == pytest.approx(0.11)
    assert model_prices["openai/gpt-oss-20b-TEE"]["reasoning_per_million"] == pytest.approx(0.11)
    assert model_prices["openai/gpt-oss-120b-TEE"]["input_per_million"] == pytest.approx(0.09)
    assert model_prices["openai/gpt-oss-120b-TEE"]["output_per_million"] == pytest.approx(0.36)
    assert model_prices["openai/gpt-oss-120b-TEE"]["reasoning_per_million"] == pytest.approx(0.36)
    assert "openai/gpt-oss-20b" not in payload["allowed_tool_models"]
    assert "openai/gpt-oss-120b" not in payload["allowed_tool_models"]
    assert "Qwen/Qwen3-Next-80B-A3B-Instruct" in payload["allowed_tool_models"]
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["input_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].input_per_million
    )
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["output_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].output_per_million
    )
    assert model_prices["Qwen/Qwen3-Next-80B-A3B-Instruct"]["reasoning_per_million"] == pytest.approx(
        MODEL_PRICING["Qwen/Qwen3-Next-80B-A3B-Instruct"].reasoning_per_million
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


def test_qwen_tool_model_pricing_ignores_reasoning_tokens() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        reasoning_tokens=1_000_000,
    )

    cost = price_llm(parse_tool_model("Qwen/Qwen3-Next-80B-A3B-Instruct"), usage)

    assert cost == pytest.approx(0.90)


def test_openai_tool_model_pricing_matches_current_chutes_rates() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        reasoning_tokens=1_000_000,
    )

    cost_20b = price_llm(parse_tool_model("openai/gpt-oss-20b-TEE"), usage)
    cost_120b = price_llm(parse_tool_model("openai/gpt-oss-120b-TEE"), usage)

    assert cost_20b == pytest.approx(0.25)
    assert cost_120b == pytest.approx(0.81)
