from __future__ import annotations

import pytest

from harnyx_commons.llm.pricing import (
    generation_usage_cost_breakdown,
)
from harnyx_commons.llm.schema import LlmUsage


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
    assert breakdown["usd_cost"] == pytest.approx(22.028)


def test_luna_prices_disjoint_cache_writes_and_reads_without_double_billing_reasoning() -> None:
    breakdown = generation_usage_cost_breakdown(
        LlmUsage(
            prompt_tokens=100_000,
            prompt_cached_tokens=20_000,
            prompt_cache_write_tokens=30_000,
            completion_tokens=10_000,
            reasoning_tokens=9_000,
            web_search_calls=3,
        ),
        provider="openai",
        model="gpt-6-luna",
    )
    assert breakdown["usd_cost_input"] == pytest.approx(0.00895)
    assert breakdown["usd_cost_output"] == pytest.approx(0.0005)
    assert breakdown["usd_cost_reasoning"] == pytest.approx(0.0045)
    assert breakdown["usd_cost_grounded"] == pytest.approx(0.03)
    assert breakdown["usd_cost"] == pytest.approx(0.04395)


@pytest.mark.parametrize(("tokens", "expected"), [(272_000, 0.0277), (272_001, 0.0551502)])
def test_luna_long_context_prices_the_entire_request(tokens: int, expected: float) -> None:
    breakdown = generation_usage_cost_breakdown(
        LlmUsage(prompt_tokens=tokens, completion_tokens=1_000, reasoning_tokens=800),
        provider="openai",
        model="gpt-6-luna",
    )
    assert breakdown["usd_cost"] == pytest.approx(expected)


@pytest.mark.parametrize(("tokens", "expected"), [(200_000, 0.412), (200_001, 0.818004)])
def test_pro_long_context_uses_prompt_count_and_bills_reasoning_separately(tokens: int, expected: float) -> None:
    breakdown = generation_usage_cost_breakdown(
        LlmUsage(prompt_tokens=tokens, completion_tokens=200, reasoning_tokens=800),
        provider="vertex",
        model="gemini-3.1-pro-preview",
    )
    assert breakdown["usd_cost"] == pytest.approx(expected)


@pytest.mark.parametrize(("day", "expected"), [("2026-12-31", 0.00615), ("2027-01-01", 0.0123)])
def test_flash_cache_discount_and_promotion_expiry(day: str, expected: float) -> None:
    from datetime import date

    breakdown = generation_usage_cost_breakdown(
        LlmUsage(prompt_tokens=10_000, prompt_cached_tokens=2_000),
        provider="vertex",
        model="gemini-3.8-flash",
        pricing_date=date.fromisoformat(day),
    )
    assert breakdown["usd_cost"] == pytest.approx(expected)


def test_overlapping_cached_token_counts_do_not_produce_negative_input_cost() -> None:
    with pytest.raises(ValueError, match="disjoint subsets"):
        generation_usage_cost_breakdown(
            LlmUsage(prompt_tokens=10, prompt_cached_tokens=8, prompt_cache_write_tokens=8),
            provider="openai",
            model="gpt-6-luna",
        )


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
    (("openai/gpt-oss-20b", 0.17),),
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
