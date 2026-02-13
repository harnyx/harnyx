"""Pricing helpers for validator tool budgeting.

All LLM prices here are **Chutes-based** reference rates, used solely for
budgeting miner tool calls (even if the runtime executes against another
provider like Vertex). External benchmarking uses its own pricing
(`platform/scripts/criterion_evaluation_benchmark.py`) and must not import this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from caster_commons.llm.schema import LlmUsage
from caster_commons.tools.types import SearchToolName

# Canonical model ids allowed for tool LLM calls.
ToolModelName = Literal["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

ALLOWED_TOOL_MODELS: tuple[ToolModelName, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
)

def parse_tool_model(raw: str | None) -> ToolModelName:
    """Parse and validate a tool LLM model identifier.

    Only canonical model ids from ALLOWED_TOOL_MODELS are accepted.
    """
    if raw is None:
        raise ValueError("model must be provided for validator tools")
    value = raw.strip()
    if not value or value not in ALLOWED_TOOL_MODELS:
        raise ValueError(f"model {value!r} is not allowed for validator tools")
    return cast(ToolModelName, value)


# Per-call flat rates for search tools, keyed by tool name.
SEARCH_PRICING: dict[SearchToolName, float] = {
    "search_web": 0.0025,
    "search_x": 0.003,
}

SEARCH_AI_PER_REFERENCEABLE_RESULT_USD = 0.004
SEARCH_SIMILAR_FEED_ITEMS_PER_CALL_USD = 0.0025


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    reasoning_per_million: float


# Chutes reference rates keyed by canonical model id.
MODEL_PRICING: Mapping[ToolModelName, ModelPricing] = {
    "openai/gpt-oss-20b": ModelPricing(0.25, 2.0, 2.0),
    "openai/gpt-oss-120b": ModelPricing(1.25, 10.0, 10.0),
}


def price_llm(model: ToolModelName, usage: LlmUsage) -> float:
    """Return USD cost for a single LLM call using Chutes reference pricing."""
    pricing = MODEL_PRICING[model]

    prompt_tokens = float(usage.prompt_tokens or 0)
    completion_tokens = float(usage.completion_tokens or 0)
    reasoning_tokens = float(usage.reasoning_tokens or 0)

    cost_input = (prompt_tokens / 1_000_000) * pricing.input_per_million
    cost_output = (completion_tokens / 1_000_000) * pricing.output_per_million
    cost_reasoning = (reasoning_tokens / 1_000_000) * pricing.reasoning_per_million
    return cost_input + cost_output + cost_reasoning


def price_search(tool_name: SearchToolName) -> float:
    """Return USD cost for a search call based on tool name."""
    if tool_name == "search_ai":
        raise ValueError("search_ai pricing depends on referenceable results; use price_search_ai")
    return float(SEARCH_PRICING[tool_name])


def price_search_ai(*, referenceable_results: int) -> float:
    """Return USD cost for AI search calls based on referenceable results."""
    if referenceable_results < 0:
        raise ValueError("referenceable_results must be non-negative")
    return float(referenceable_results) * SEARCH_AI_PER_REFERENCEABLE_RESULT_USD


__all__ = [
    "ALLOWED_TOOL_MODELS",
    "ToolModelName",
    "price_llm",
    "price_search",
    "price_search_ai",
    "MODEL_PRICING",
    "SEARCH_PRICING",
    "SEARCH_AI_PER_REFERENCEABLE_RESULT_USD",
    "SEARCH_SIMILAR_FEED_ITEMS_PER_CALL_USD",
    "ModelPricing",
    "parse_tool_model",
]
