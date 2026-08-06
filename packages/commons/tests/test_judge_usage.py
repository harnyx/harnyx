from __future__ import annotations

from typing import cast

import pytest

from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
from harnyx_commons.llm.judge_usage import (
    JudgeUsageMetadataError,
    judge_usage_from_response,
    judge_usage_without_actual_cost_from_response,
    merge_judge_usage,
)
from harnyx_commons.llm.schema import LlmResponse, LlmUsage


def _response(
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    reasoning_tokens: int | None = 2,
    metadata: dict[str, object] | None = None,
) -> LlmResponse:
    return LlmResponse(
        id="response-id",
        choices=(),
        usage=LlmUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        metadata=metadata,
    )


def test_judge_usage_from_chutes_response_uses_actual_cost_metadata() -> None:
    summary = judge_usage_from_response(
        _response(
            metadata={
                "selected_provider": "chutes",
                "selected_model": "google/gemma-4-31B-turbo-TEE",
                "actual_cost_usd": 0.0123,
                "actual_cost_provider": "chutes",
                "actual_cost_evidence": "pricing-cache",
            }
        ),
        default_provider="openrouter",
        default_model="fallback-model",
    )

    assert summary.call_count == 1
    assert summary.prompt_tokens == 10
    assert summary.completion_tokens == 5
    assert summary.total_tokens == 15
    assert summary.reasoning_tokens == 2
    assert summary.actual_cost_usd == 0.0123
    assert summary.models[0].provider == "chutes"
    assert summary.models[0].model == "google/gemma-4-31B-turbo-TEE"
    assert summary.models[0].actual_cost_source == "provider_actual"
    assert summary.models[0].actual_cost_provider == "chutes"
    assert summary.models[0].actual_cost_evidence == "pricing-cache"


def test_judge_usage_keeps_model_reasoning_tokens_unavailable_and_coalesces_summary() -> None:
    summary = judge_usage_from_response(
        _response(reasoning_tokens=None),
        default_provider="chutes",
        default_model="judge",
    )

    assert summary.reasoning_tokens == 0
    assert summary.models[0].reasoning_tokens is None


def test_judge_usage_from_retried_response_uses_billable_response_count() -> None:
    summary = judge_usage_from_response(
        _response(
            prompt_tokens=21,
            completion_tokens=5,
            total_tokens=26,
            metadata={
                "actual_cost_usd": 0.02,
                "actual_cost_usd_total": 0.03,
                "attempts": 3,
                "billable_response_count": 2,
            },
        ),
        default_provider="chutes",
        default_model="judge",
    )

    assert summary.call_count == 2
    assert summary.models[0].call_count == 2
    assert summary.prompt_tokens == 21
    assert summary.completion_tokens == 5
    assert summary.total_tokens == 26
    assert summary.actual_cost_usd == pytest.approx(0.03)


def test_judge_usage_from_retried_response_keeps_actual_cost_unavailable_without_total() -> None:
    summary = judge_usage_from_response(
        _response(
            prompt_tokens=21,
            completion_tokens=5,
            total_tokens=26,
            metadata={
                "actual_cost_usd": 0.02,
                "billable_response_count": 2,
            },
        ),
        default_provider="chutes",
        default_model="judge",
    )

    assert summary.call_count == 2
    assert summary.prompt_tokens == 21
    assert summary.completion_tokens == 5
    assert summary.total_tokens == 26
    assert summary.actual_cost_usd is None
    assert summary.models[0].actual_cost_source == "unavailable"


def test_judge_usage_does_not_price_unknown_cost_response() -> None:
    summary = judge_usage_from_response(
        _response(),
        default_provider="chutes",
        default_model="unknown/model",
    )

    assert summary.call_count == 1
    assert summary.prompt_tokens == 10
    assert summary.completion_tokens == 5
    assert summary.actual_cost_usd is None
    assert summary.models[0].actual_cost_source == "unavailable"


def test_judge_usage_ignores_structured_actual_cost_evidence() -> None:
    summary = judge_usage_from_response(
        _response(
            metadata={
                "actual_cost_usd": 0.0123,
                "actual_cost_provider": "chutes",
                "actual_cost_evidence": {"settlement_source": "static_pricing"},
            }
        ),
        default_provider="chutes",
        default_model="judge",
    )

    assert summary.actual_cost_usd == pytest.approx(0.0123)
    assert summary.models[0].actual_cost_evidence is None


def test_judge_usage_does_not_treat_raw_attempts_as_billable_calls() -> None:
    summary = judge_usage_from_response(
        _response(metadata={"attempts": 3}),
        default_provider="chutes",
        default_model="judge",
    )

    assert summary.call_count == 1
    assert summary.models[0].call_count == 1


@pytest.mark.parametrize("metadata_key", ["actual_cost_usd", "actual_cost_usd_total"])
@pytest.mark.parametrize("bad_cost", [True, float("nan"), float("inf"), float("-inf"), -0.01])
def test_judge_usage_rejects_invalid_actual_cost(metadata_key: str, bad_cost: object) -> None:
    with pytest.raises(JudgeUsageMetadataError, match="actual_cost_usd"):
        judge_usage_from_response(
            _response(metadata={metadata_key: bad_cost}),
            default_provider="chutes",
            default_model="judge",
        )


def test_judge_usage_without_actual_cost_retains_valid_tokens_and_call_count() -> None:
    summary = judge_usage_without_actual_cost_from_response(
        _response(
            prompt_tokens=21,
            completion_tokens=5,
            total_tokens=26,
            reasoning_tokens=3,
            metadata={
                "selected_provider": "chutes",
                "selected_model": "judge",
                "billable_response_count": 2,
                "actual_cost_usd": True,
            },
        ),
        default_provider="openrouter",
        default_model="fallback-model",
    )

    assert summary.call_count == 2
    assert summary.prompt_tokens == 21
    assert summary.completion_tokens == 5
    assert summary.total_tokens == 26
    assert summary.reasoning_tokens == 3
    assert summary.actual_cost_usd is None
    assert summary.models[0].provider == "chutes"
    assert summary.models[0].model == "judge"
    assert summary.models[0].actual_cost_source == "unavailable"


@pytest.mark.parametrize(
    "usage_parser",
    [judge_usage_from_response, judge_usage_without_actual_cost_from_response],
)
def test_judge_usage_parsers_reject_invalid_billable_response_count(usage_parser: object) -> None:
    with pytest.raises(JudgeUsageMetadataError, match="billable_response_count"):
        usage_parser(
            _response(metadata={"billable_response_count": 0}),
            default_provider="chutes",
            default_model="judge",
        )


@pytest.mark.parametrize(
    "usage_parser",
    [judge_usage_from_response, judge_usage_without_actual_cost_from_response],
)
def test_judge_usage_parsers_reject_negative_token_values(usage_parser: object) -> None:
    with pytest.raises(JudgeUsageMetadataError, match="prompt_tokens"):
        usage_parser(
            _response(prompt_tokens=-1),
            default_provider="chutes",
            default_model="judge",
        )


@pytest.mark.parametrize("bad_cost", [True, float("nan"), float("inf"), float("-inf"), -0.01])
def test_judge_model_usage_rejects_invalid_actual_cost(bad_cost: object) -> None:
    with pytest.raises(ValueError, match="actual_cost_usd"):
        JudgeModelUsage(
            provider="chutes",
            model="judge",
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=0,
            actual_cost_usd=cast(float, bad_cost),
            actual_cost_source="provider_actual",
        )


@pytest.mark.parametrize("bad_cost", [True, float("nan"), float("inf"), float("-inf"), -0.01])
def test_judge_usage_summary_rejects_invalid_actual_cost(bad_cost: object) -> None:
    with pytest.raises(ValueError, match="actual_cost_usd"):
        JudgeUsageSummary(
            call_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
            actual_cost_usd=cast(float, bad_cost),
            models=(),
        )


def test_judge_usage_summary_rejects_actual_cost_total_that_does_not_match_models() -> None:
    model_usage = JudgeModelUsage(
        provider="chutes",
        model="judge",
        call_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=0,
        actual_cost_usd=1.0,
        actual_cost_source="provider_actual",
    )

    with pytest.raises(ValueError, match="actual_cost_usd must equal model actual costs"):
        JudgeUsageSummary(
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            reasoning_tokens=0,
            actual_cost_usd=0.01,
            models=(model_usage,),
        )


def test_judge_usage_summary_requires_unavailable_cost_when_any_model_cost_is_unavailable() -> None:
    known_cost = JudgeModelUsage(
        provider="chutes",
        model="known-cost-judge",
        call_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        reasoning_tokens=0,
        actual_cost_usd=0.01,
        actual_cost_source="provider_actual",
    )
    unknown_cost = JudgeModelUsage(
        provider="chutes",
        model="unknown-cost-judge",
        call_count=1,
        prompt_tokens=7,
        completion_tokens=3,
        total_tokens=10,
        reasoning_tokens=1,
        actual_cost_usd=None,
        actual_cost_source="unavailable",
    )

    summary = JudgeUsageSummary(
        call_count=2,
        prompt_tokens=17,
        completion_tokens=8,
        total_tokens=25,
        reasoning_tokens=1,
        actual_cost_usd=None,
        models=(known_cost, unknown_cost),
    )

    assert summary.actual_cost_usd is None
    assert summary.models[0].actual_cost_usd == pytest.approx(0.01)
    assert summary.models[1].actual_cost_usd is None

    with pytest.raises(ValueError, match="actual_cost_usd must be unavailable"):
        JudgeUsageSummary(
            call_count=2,
            prompt_tokens=17,
            completion_tokens=8,
            total_tokens=25,
            reasoning_tokens=1,
            actual_cost_usd=0.01,
            models=(known_cost, unknown_cost),
        )


def test_merge_judge_usage_sums_tokens_and_actual_costs_by_model() -> None:
    first = judge_usage_from_response(
        _response(metadata={"selected_provider": "chutes", "selected_model": "judge", "actual_cost_usd": 0.01}),
        default_provider="chutes",
        default_model="judge",
    )
    second = judge_usage_from_response(
        _response(
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
            reasoning_tokens=1,
            metadata={"selected_provider": "chutes", "selected_model": "judge", "actual_cost_usd": 0.02},
        ),
        default_provider="chutes",
        default_model="judge",
    )

    merged = merge_judge_usage((first, second))

    assert merged.call_count == 2
    assert merged.prompt_tokens == 17
    assert merged.completion_tokens == 8
    assert merged.total_tokens == 25
    assert merged.reasoning_tokens == 3
    assert merged.actual_cost_usd == 0.03
    assert len(merged.models) == 1
    assert merged.models[0].call_count == 2


def test_merge_judge_usage_coalesces_unavailable_model_reasoning_tokens() -> None:
    first = judge_usage_from_response(
        _response(reasoning_tokens=None),
        default_provider="chutes",
        default_model="judge",
    )
    second = judge_usage_from_response(
        _response(prompt_tokens=7, completion_tokens=3, total_tokens=10, reasoning_tokens=1),
        default_provider="chutes",
        default_model="judge",
    )

    merged = merge_judge_usage((first, second))

    assert merged.reasoning_tokens == 1
    assert merged.models[0].reasoning_tokens == 1


def test_merge_judge_usage_keeps_tokens_when_actual_cost_is_missing() -> None:
    with_actual = judge_usage_from_response(
        _response(metadata={"selected_provider": "chutes", "selected_model": "judge", "actual_cost_usd": 0.01}),
        default_provider="chutes",
        default_model="judge",
    )
    without_actual = judge_usage_from_response(
        _response(prompt_tokens=7, completion_tokens=3, total_tokens=10, reasoning_tokens=1),
        default_provider="chutes",
        default_model="judge",
    )

    merged = merge_judge_usage((with_actual, without_actual))

    assert merged.call_count == 2
    assert merged.total_tokens == 25
    assert merged.actual_cost_usd is None
    assert merged.models[0].actual_cost_usd is None
    assert merged.models[0].actual_cost_source == "unavailable"
