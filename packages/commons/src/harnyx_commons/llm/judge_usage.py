"""Build judge usage summaries from LLM provider responses."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
from harnyx_commons.llm.schema import LlmResponse


class JudgeUsageMetadataError(ValueError):
    """Provider response metadata cannot produce trustworthy judge usage."""


def judge_usage_from_response(
    response: LlmResponse,
    *,
    default_provider: str,
    default_model: str,
) -> JudgeUsageSummary:
    call_count = _metadata_positive_int(response, "billable_response_count", fallback=1)
    return _build_judge_usage(
        response,
        default_provider=default_provider,
        default_model=default_model,
        call_count=call_count,
        actual_cost=_actual_cost_for_judge(response, call_count=call_count),
        actual_cost_provider=_metadata_optional_string(response, "actual_cost_provider"),
        actual_cost_evidence=_metadata_optional_string(response, "actual_cost_evidence"),
    )


def judge_usage_without_actual_cost_from_response(
    response: LlmResponse,
    *,
    default_provider: str,
    default_model: str,
) -> JudgeUsageSummary:
    """Build judge usage while deliberately treating actual cost as unavailable."""
    return _build_judge_usage(
        response,
        default_provider=default_provider,
        default_model=default_model,
        call_count=_metadata_positive_int(response, "billable_response_count", fallback=1),
        actual_cost=None,
        actual_cost_provider=None,
        actual_cost_evidence=None,
    )


def _build_judge_usage(
    response: LlmResponse,
    *,
    default_provider: str,
    default_model: str,
    call_count: int,
    actual_cost: float | None,
    actual_cost_provider: str | None,
    actual_cost_evidence: str | None,
) -> JudgeUsageSummary:
    model_usage = JudgeModelUsage(
        provider=_metadata_string(response, "selected_provider", default_provider),
        model=_metadata_string(response, "selected_model", default_model),
        call_count=call_count,
        prompt_tokens=_usage_token(response.usage.prompt_tokens, "prompt_tokens"),
        completion_tokens=_usage_token(response.usage.completion_tokens, "completion_tokens"),
        total_tokens=_usage_token(response.usage.total_tokens, "total_tokens"),
        reasoning_tokens=_optional_usage_token(response.usage.reasoning_tokens, "reasoning_tokens"),
        actual_cost_usd=actual_cost,
        actual_cost_source="provider_actual" if actual_cost is not None else "unavailable",
        actual_cost_provider=actual_cost_provider,
        actual_cost_evidence=actual_cost_evidence,
    )
    return JudgeUsageSummary(
        call_count=model_usage.call_count,
        prompt_tokens=model_usage.prompt_tokens,
        completion_tokens=model_usage.completion_tokens,
        total_tokens=model_usage.total_tokens,
        reasoning_tokens=_coalesce_reasoning_tokens((model_usage.reasoning_tokens,)),
        actual_cost_usd=model_usage.actual_cost_usd,
        models=(model_usage,),
    )


def merge_judge_usage(usages: Iterable[JudgeUsageSummary | None]) -> JudgeUsageSummary:
    present = tuple(usage for usage in usages if usage is not None)
    model_groups: dict[tuple[str, str], list[JudgeModelUsage]] = defaultdict(list)
    for usage in present:
        for model in usage.models:
            model_groups[(model.provider, model.model)].append(model)

    merged_models = tuple(
        _merge_model_usage(provider=provider, model=model, usages=tuple(group))
        for (provider, model), group in sorted(model_groups.items())
    )
    return JudgeUsageSummary(
        call_count=sum(model.call_count for model in merged_models),
        prompt_tokens=sum(model.prompt_tokens for model in merged_models),
        completion_tokens=sum(model.completion_tokens for model in merged_models),
        total_tokens=sum(model.total_tokens for model in merged_models),
        reasoning_tokens=_coalesce_reasoning_tokens(model.reasoning_tokens for model in merged_models),
        actual_cost_usd=_sum_actual_costs(tuple(model.actual_cost_usd for model in merged_models)),
        models=merged_models,
    )


@dataclass(frozen=True, slots=True)
class _ActualCostMetadata:
    provider: str | None
    evidence: str | None


def _merge_model_usage(
    *,
    provider: str,
    model: str,
    usages: tuple[JudgeModelUsage, ...],
) -> JudgeModelUsage:
    actual_cost = _sum_actual_costs(tuple(usage.actual_cost_usd for usage in usages))
    actual_metadata = (
        _merge_actual_cost_metadata(usages)
        if actual_cost is not None
        else _ActualCostMetadata(None, None)
    )
    return JudgeModelUsage(
        provider=provider,
        model=model,
        call_count=sum(usage.call_count for usage in usages),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        reasoning_tokens=_coalesce_reasoning_tokens(usage.reasoning_tokens for usage in usages),
        actual_cost_usd=actual_cost,
        actual_cost_source="provider_actual" if actual_cost is not None else "unavailable",
        actual_cost_provider=actual_metadata.provider,
        actual_cost_evidence=actual_metadata.evidence,
    )


def _coalesce_reasoning_tokens(values: Iterable[int | None]) -> int:
    return sum(value or 0 for value in values)


def _sum_actual_costs(costs: tuple[float | None, ...]) -> float | None:
    if any(cost is None for cost in costs):
        return None
    return round(sum(costs), 12)


def _merge_actual_cost_metadata(usages: tuple[JudgeModelUsage, ...]) -> _ActualCostMetadata:
    providers = {usage.actual_cost_provider for usage in usages if usage.actual_cost_provider is not None}
    evidence = {usage.actual_cost_evidence for usage in usages if usage.actual_cost_evidence is not None}
    return _ActualCostMetadata(
        provider=providers.pop() if len(providers) == 1 else None,
        evidence=evidence.pop() if len(evidence) == 1 else None,
    )


def _metadata_string(response: LlmResponse, key: str, fallback: str) -> str:
    value = (response.metadata or {}).get(key)
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _metadata_optional_string(response: LlmResponse, key: str) -> str | None:
    value = (response.metadata or {}).get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _metadata_float(response: LlmResponse, key: str) -> float | None:
    value = (response.metadata or {}).get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JudgeUsageMetadataError(f"{key} must be numeric when supplied")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise JudgeUsageMetadataError(f"{key} must be finite when supplied")
    if numeric < 0.0:
        raise JudgeUsageMetadataError(f"{key} must be non-negative when supplied")
    return numeric


def _metadata_positive_int(response: LlmResponse, key: str, *, fallback: int) -> int:
    value = (response.metadata or {}).get(key)
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeUsageMetadataError(f"{key} must be an integer when supplied")
    if value < 1:
        raise JudgeUsageMetadataError(f"{key} must be positive when supplied")
    return value


def _usage_token(value: int | None, key: str) -> int:
    if value is None:
        return 0
    return _non_negative_int(value, key)


def _optional_usage_token(value: int | None, key: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, key)


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeUsageMetadataError(f"{key} must be an integer when supplied")
    if value < 0:
        raise JudgeUsageMetadataError(f"{key} must be non-negative when supplied")
    return value


def _actual_cost_for_judge(response: LlmResponse, *, call_count: int) -> float | None:
    single_response_cost = _metadata_float(response, "actual_cost_usd")
    total = _metadata_float(response, "actual_cost_usd_total")
    if total is not None:
        return total
    if call_count == 1:
        return single_response_cost
    return None


__all__ = [
    "JudgeUsageMetadataError",
    "judge_usage_from_response",
    "judge_usage_without_actual_cost_from_response",
    "merge_judge_usage",
]
