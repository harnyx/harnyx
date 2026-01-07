"""Pricing helpers for validator tool budgeting.

All LLM prices here are **Chutes-based** reference rates, used solely for
budgeting miner tool calls (even if the runtime executes against another
provider like Vertex). External benchmarking uses its own pricing
(`platform/scripts/eval_benchmark.py`) and must not import this module.
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
    return float(SEARCH_PRICING[tool_name])


__all__ = [
    "ALLOWED_TOOL_MODELS",
    "ToolModelName",
    "price_llm",
    "price_search",
    "MODEL_PRICING",
    "SEARCH_PRICING",
    "ModelPricing",
    "parse_tool_model",
]
