from __future__ import annotations

import pytest

from harnyx_commons.llm.pricing import (
    GENERATION_MODEL_PRICING,
    MODEL_PRICING,
    generation_usage_cost_breakdown,
)
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.llm.tool_models import ALLOWED_TOOL_MODELS


def test_generation_usage_cost_breakdown_normalizes_vertex_claude_publisher_path_models() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=2_000,
        total_tokens=3_000,
        web_search_calls=5,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="publishers/anthropic/models/claude-sonnet-4-5@20250929",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == "vertex:claude-sonnet-4-5"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.05)
    assert breakdown["usd_cost"] == pytest.approx(0.0863)


def test_generation_usage_cost_breakdown_normalizes_vertex_gemini_publisher_path_models() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        reasoning_tokens=200,
        web_search_calls=1,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="publishers/google/models/gemini-2.5-pro",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == "vertex:gemini-2.5-pro"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.035)
    assert breakdown["usd_cost"] == pytest.approx(0.04325)


def test_generation_usage_cost_breakdown_multiplies_generic_vertex_grounding_by_search_calls() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        reasoning_tokens=200,
        web_search_calls=5,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="gemini-2.5-pro",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == "vertex:gemini-2.5-pro"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.175)
    assert breakdown["usd_cost"] == pytest.approx(0.18325)


def test_generation_usage_cost_breakdown_normalizes_vertex_gemini_full_resource_models() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        reasoning_tokens=200,
        web_search_calls=1,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="projects/test/locations/us-central1/publishers/google/models/gemini-2.5-pro",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == "vertex:gemini-2.5-pro"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.035)
    assert breakdown["usd_cost"] == pytest.approx(0.04325)


def test_generation_usage_cost_breakdown_prices_default_domain_tweak_model() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        web_search_calls=2,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="gemini-3.1-pro-preview",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == "vertex:gemini-3.1-pro-preview"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.028)
    assert breakdown["usd_cost"] == pytest.approx(14.028)


@pytest.mark.parametrize(
    ("provider", "expected_grounded_cost", "expected_cost"),
    (
        ("vertex", 0.014, 1.764),
        ("google", 0.0, 1.75),
    ),
)
def test_generation_usage_cost_breakdown_prices_domain_tweak_flash_lite_model(
    provider: str,
    expected_grounded_cost: float,
    expected_cost: float,
) -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        web_search_calls=1,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider=provider,
        model="gemini-3.1-flash-lite",
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["usd_cost_grounded"] == pytest.approx(expected_grounded_cost)
    assert breakdown["usd_cost"] == pytest.approx(expected_cost)


def test_generation_usage_cost_breakdown_does_not_normalize_malformed_vertex_gemini_paths() -> None:
    usage = LlmUsage(
        prompt_tokens=1_000,
        completion_tokens=500,
        total_tokens=1_500,
        reasoning_tokens=200,
        web_search_calls=1,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="vertex",
        model="foo/publishers/google/models/gemini-2.5-pro",
    )

    assert breakdown["pricing_missing"] is True
    assert breakdown["pricing_key"] == "vertex:foo/publishers/google/models/gemini-2.5-pro"
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.035)
    assert breakdown["usd_cost"] == pytest.approx(0.035)


@pytest.mark.parametrize(
    ("model", "expected_cost"),
    (
        ("openai/gpt-oss-20b", 0.17),
        ("openai/gpt-oss-120b", 0.219),
    ),
)
def test_generation_usage_cost_breakdown_prices_openrouter_gpt_oss(model: str, expected_cost: float) -> None:
    usage = LlmUsage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    breakdown = generation_usage_cost_breakdown(
        usage,
        provider="openrouter",
        model=model,
    )

    assert breakdown["pricing_missing"] is False
    assert breakdown["pricing_key"] == f"openrouter:{model}"
    assert breakdown["usd_cost"] == pytest.approx(expected_cost)


def test_generation_pricing_is_separate_from_validator_tool_model_pricing() -> None:
    assert set(MODEL_PRICING) == set(ALLOWED_TOOL_MODELS)
    assert "vertex:gemini-3-pro-preview" in GENERATION_MODEL_PRICING
    assert "vertex:gemini-3.1-pro-preview" in GENERATION_MODEL_PRICING
    assert "vertex:gemini-3-pro-preview" not in MODEL_PRICING
