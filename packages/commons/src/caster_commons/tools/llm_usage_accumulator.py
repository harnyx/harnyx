"""LLM usage accumulation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from caster_commons.domain.session import LlmUsageTotals

if TYPE_CHECKING:
    from caster_commons.tools.usage_tracker import ToolCallUsage


def accumulate_llm_usage(
    existing: dict[str, dict[str, LlmUsageTotals]],
    *,
    usage: ToolCallUsage | None,
    llm_tokens: int,
) -> dict[str, dict[str, LlmUsageTotals]]:
    if usage is None:
        return existing

    if usage.provider is None or usage.model is None:
        raise ValueError("provider and model must be supplied when recording LLM usage")

    prompt = usage.prompt_tokens or 0
    completion = usage.completion_tokens or 0
    total = usage.total_tokens if usage.total_tokens is not None else llm_tokens

    if prompt < 0 or completion < 0 or total < 0:
        raise ValueError("token counts must be non-negative")

    providers: dict[str, dict[str, LlmUsageTotals]] = {
        provider_name: dict(models) for provider_name, models in existing.items()
    }
    model_totals = providers.get(usage.provider, {})
    updated_model_totals = dict(model_totals)

    aggregate = updated_model_totals.get(usage.model, LlmUsageTotals())
    updated_model_totals[usage.model] = aggregate.accumulate(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    providers[usage.provider] = updated_model_totals
    return providers
