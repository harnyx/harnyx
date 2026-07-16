"""Pricing helpers for validator tool budgeting and generation accounting.

LLM prices here are reference rates for budgeting miner tool calls. Model rates
follow the configured reference provider for each canonical tool model. Product
generation uses separate provider/model pricing in this module.
External benchmarking scripts use their own pricing
(`apps/platform/scripts/miner_task_benchmark.py`) and must not import generation
pricing as a replacement for benchmark-specific rates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.provider_types import (
    AI_GATEWAY_PROVIDER,
    CHUTES_PROVIDER,
    OPENROUTER_PROVIDER,
    VERTEX_PROVIDER,
)
from harnyx_commons.llm.providers.vertex.anthropic import is_claude_model, normalize_claude_model
from harnyx_commons.llm.schema import LlmUsage, extract_vertex_gemini_model_id
from harnyx_commons.llm.tool_models import (
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    MinerSelectedLlmProviderName,
    ToolModelName,
)
from harnyx_commons.tools.embedding_models import (
    QWEN3_CHUTES_EMBEDDING_MODEL,
    QWEN3_OPENROUTER_EMBEDDING_MODEL,
    EmbeddingProviderName,
)
from harnyx_commons.tools.types import SearchToolName

# Per-referenceable-result rates for search tools, keyed by tool name.
SEARCH_PRICING_PER_REFERENCEABLE_RESULT: dict[SearchToolName, float] = {
    "search_web": 0.0001,
    "search_ai": 0.0004,
    "fetch_page": 0.0005,
}

PARALLEL_SEARCH_BASE_RESULTS = 10
PARALLEL_SEARCH_BASE_COST_USD = 0.005
PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD = 0.001
PARALLEL_EXTRACT_URL_COST_USD = 0.001
VERTEX_GROUNDED_PER_1K = 35.0
VERTEX_GEMINI3_GROUNDED_PER_1K = 14.0
VERTEX_CLAUDE_WEB_SEARCH_PER_1K = 10.0


@dataclass(frozen=True)
class EmbeddingPricing:
    input_per_million: float | None = None
    usd_per_second: float | None = None


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    reasoning_per_million: float

    @property
    def billable_reasoning_per_million(self) -> float:
        if self.reasoning_per_million != 0.0:
            return self.reasoning_per_million
        return self.output_per_million


# Reference rates keyed by canonical model id.
MODEL_PRICING: Mapping[ToolModelName, ModelPricing] = {
    "openai/gpt-oss-20b": ModelPricing(0.03, 0.14, 0.0),
    "openai/gpt-oss-120b": ModelPricing(0.039, 0.18, 0.0),
    "deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.28, 0.42, 0.0),
    "zai-org/GLM-5-TEE": ModelPricing(0.95, 2.55, 0.0),
    "Qwen/Qwen3.6-27B-TEE": ModelPricing(0.50, 2.00, 0.0),
    "google/gemma-4-31B-turbo-TEE": ModelPricing(0.13, 0.38, 0.0),
}

STATIC_LLM_PRICING: Mapping[str, ModelPricing] = {
    **MODEL_PRICING,
    "openai/gpt-oss-20b-TEE": MODEL_PRICING["openai/gpt-oss-20b"],
    "openai/gpt-oss-120b-TEE": MODEL_PRICING["openai/gpt-oss-120b"],
    "moonshotai/Kimi-K2.5-TEE": ModelPricing(0.44, 2.00, 0.0),
}

MINER_TOOL_LLM_PRICING: Mapping[MinerSelectedLlmProviderName, Mapping[str, ModelPricing]] = {
    CHUTES_PROVIDER: {
        model: MODEL_PRICING[cast(ToolModelName, model)]
        for model in MINER_SELECTED_LLM_PROVIDER_MODELS[CHUTES_PROVIDER]
    },
    OPENROUTER_PROVIDER: {
        "openai/gpt-oss-20b": MODEL_PRICING["openai/gpt-oss-20b"],
        "openai/gpt-oss-120b": MODEL_PRICING["openai/gpt-oss-120b"],
        "deepseek/deepseek-v3.2": MODEL_PRICING["deepseek-ai/DeepSeek-V3.2-TEE"],
        "z-ai/glm-5": MODEL_PRICING["zai-org/GLM-5-TEE"],
        "qwen/qwen3.6-27b": MODEL_PRICING["Qwen/Qwen3.6-27B-TEE"],
        "google/gemma-4-31b-it": MODEL_PRICING["google/gemma-4-31B-turbo-TEE"],
    },
    AI_GATEWAY_PROVIDER: {
        "thinkingmachines/inkling": ModelPricing(1.00, 4.05, 0.0),
        "zai/glm-5.2-fast": ModelPricing(2.10, 6.60, 0.0),
        "openai/gpt-oss-20b": ModelPricing(0.03, 0.14, 0.0),
        "zai/glm-4.7": ModelPricing(0.43, 1.75, 0.0),
        "google/gemma-4-31b-it": ModelPricing(0.14, 0.40, 0.0),
        "openai/gpt-oss-120b": ModelPricing(0.10, 0.50, 0.0),
        "minimax/minimax-m2.7": ModelPricing(0.30, 1.20, 0.0),
        "zai/glm-4.7-flash": ModelPricing(0.07, 0.40, 0.0),
    },
}

MINER_TOOL_EMBEDDING_PRICING: Mapping[EmbeddingProviderName, Mapping[str, EmbeddingPricing]] = {
    "chutes": {
        QWEN3_CHUTES_EMBEDDING_MODEL: EmbeddingPricing(usd_per_second=0.0005),
    },
    "openrouter": {
        QWEN3_OPENROUTER_EMBEDDING_MODEL: EmbeddingPricing(input_per_million=0.01),
    },
}

GENERATION_MODEL_PRICING: Mapping[str, ModelPricing] = {
    "openrouter:openai/gpt-oss-20b": ModelPricing(0.03, 0.14, 0.0),
    "openrouter:openai/gpt-oss-120b": ModelPricing(0.039, 0.18, 0.0),
    # Historical/accounting prices only; Vertex runtime rejects Gemini models earlier than 3.
    "vertex:gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.0),
    "vertex:gemini-2.5-flash": ModelPricing(0.30, 2.50, 0.0),
    "vertex:gemini-2.5-flash-lite": ModelPricing(0.10, 0.40, 0.0),
    "vertex:gemini-3-pro-preview": ModelPricing(2.0, 12.0, 0.0),
    "vertex:gemini-3.1-pro-preview": ModelPricing(2.0, 12.0, 0.0),
    "vertex:claude-opus-4-5": ModelPricing(5.50, 27.50, 0.0),
    "vertex:claude-sonnet-4-5": ModelPricing(3.30, 16.50, 0.0),
    "vertex:claude-haiku-4-5": ModelPricing(1.10, 5.50, 0.0),
    "anthropic:sonnet-4.5": ModelPricing(3.0, 15.0, 0.0),
    "anthropic:haiku-4.5": ModelPricing(1.0, 5.0, 0.0),
    "gpt-5": ModelPricing(1.25, 10.0, 0.0),
    "gpt-5.1": ModelPricing(1.25, 10.0, 0.0),
    "gpt-5-pro": ModelPricing(15.0, 120.0, 0.0),
    "gpt-5-mini": ModelPricing(0.25, 2.0, 0.0),
    "gpt-5-nano": ModelPricing(0.05, 0.4, 0.0),
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, 0.0),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50, 0.0),
    "gemini-3-pro-preview": ModelPricing(2.0, 12.0, 0.0),
    "gemini-3.1-pro-preview": ModelPricing(2.0, 12.0, 0.0),
    "claude-opus-4-5": ModelPricing(5.50, 27.50, 0.0),
    "claude-sonnet-4-5": ModelPricing(3.30, 16.50, 0.0),
    "claude-haiku-4-5": ModelPricing(1.10, 5.50, 0.0),
    "sonnet-4.5": ModelPricing(3.0, 15.0, 0.0),
    "haiku-4.5": ModelPricing(1.0, 5.0, 0.0),
}


def price_llm(model: ToolModelName, usage: LlmUsage) -> float:
    """Return USD cost for a single LLM call using reference pricing."""
    pricing = MODEL_PRICING[model]
    return _price_tokens(pricing, usage)


def price_miner_llm(provider: str, model: str, usage: LlmUsage) -> float:
    """Return USD cost for a miner-selected provider/model LLM call."""
    if provider not in MINER_TOOL_LLM_PRICING:
        raise KeyError(provider)
    pricing_by_model = MINER_TOOL_LLM_PRICING[cast(MinerSelectedLlmProviderName, provider)]
    pricing = pricing_by_model[model]
    return _price_tokens(pricing, usage)


def price_static_llm_model(model: str, usage: LlmUsage) -> float | None:
    """Return USD cost for a configured static LLM model, if available."""
    pricing = STATIC_LLM_PRICING.get(model)
    if pricing is None:
        return None
    return _price_tokens(pricing, usage)


def pricing_key(provider: str, model: str) -> str:
    """Return the normalized generation pricing key for a provider/model route."""
    provider_key = provider.strip().lower()
    model_raw = _normalized_generation_model(provider=provider, model=model)
    model_base = model_raw.split("@", 1)[0]
    return f"{provider_key}:{model_base}"


def lookup_pricing(provider: str, model: str) -> ModelPricing | None:
    """Return generation pricing for an arbitrary provider/model route."""
    key = pricing_key(provider, model)
    pricing = GENERATION_MODEL_PRICING.get(key)
    if pricing is not None:
        return pricing
    return GENERATION_MODEL_PRICING.get(_normalized_generation_model(provider=provider, model=model))


def grounded_cost_usd(*, provider: str, model: str, web_search_calls: int) -> float:
    """Return provider-billed grounding/search cost for generation LLM calls."""
    if web_search_calls <= 0:
        return 0.0
    provider_key = provider.strip().lower()
    if provider_key != VERTEX_PROVIDER:
        return 0.0

    model_name = _normalized_generation_model(provider=provider, model=model)
    if model_name.startswith("gemini-3"):
        return (float(web_search_calls) * VERTEX_GEMINI3_GROUNDED_PER_1K) / 1000.0
    if model_name.startswith("claude-"):
        return (float(web_search_calls) * VERTEX_CLAUDE_WEB_SEARCH_PER_1K) / 1000.0
    return (float(web_search_calls) * VERTEX_GROUNDED_PER_1K) / 1000.0


def generation_usage_cost_breakdown(usage: LlmUsage, *, provider: str, model: str) -> JsonObject:
    """Return reference-cost details for a generation LLM call."""
    pricing = lookup_pricing(provider, model)

    prompt_tokens = float(usage.prompt_tokens or 0)
    prompt_cached_tokens = float(usage.prompt_cached_tokens or 0)
    completion_tokens = float(usage.completion_tokens or 0)
    reasoning_tokens = float(usage.reasoning_tokens or 0)
    total_tokens = float(usage.total_tokens or 0)
    web_search_calls = int(usage.web_search_calls or 0)

    grounded_cost = grounded_cost_usd(provider=provider, model=model, web_search_calls=web_search_calls)
    base_result: JsonObject = {
        "provider": provider,
        "model": model,
        "pricing_key": pricing_key(provider, model),
        "prompt_tokens": prompt_tokens,
        "prompt_cached_tokens": prompt_cached_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "web_search_calls": float(web_search_calls),
        "usd_cost_grounded": grounded_cost,
    }
    if pricing is None:
        return {
            **base_result,
            "pricing_missing": True,
            "usd_cost_input": 0.0,
            "usd_cost_output": 0.0,
            "usd_cost_reasoning": 0.0,
            "usd_cost": grounded_cost,
        }

    cost_input = (prompt_tokens / 1_000_000) * pricing.input_per_million
    cost_output = (completion_tokens / 1_000_000) * pricing.output_per_million
    cost_reasoning = (reasoning_tokens / 1_000_000) * pricing.billable_reasoning_per_million
    return {
        **base_result,
        "pricing_missing": False,
        "usd_cost_input": cost_input,
        "usd_cost_output": cost_output,
        "usd_cost_reasoning": cost_reasoning,
        "usd_cost": cost_input + cost_output + cost_reasoning + grounded_cost,
    }


def _price_tokens(pricing: ModelPricing, usage: LlmUsage) -> float:
    """Return USD cost for token usage under the supplied per-model rates."""

    prompt_tokens = float(usage.prompt_tokens or 0)
    completion_tokens = float(usage.completion_tokens or 0)
    reasoning_tokens = float(usage.reasoning_tokens or 0)

    cost_input = (prompt_tokens / 1_000_000) * pricing.input_per_million
    cost_output = (completion_tokens / 1_000_000) * pricing.output_per_million
    cost_reasoning = (reasoning_tokens / 1_000_000) * pricing.billable_reasoning_per_million
    return cost_input + cost_output + cost_reasoning


def _normalized_generation_model(*, provider: str, model: str) -> str:
    normalized_provider = provider.strip().lower()
    trimmed_model = model.strip()
    if normalized_provider == VERTEX_PROVIDER and is_claude_model(trimmed_model):
        return normalize_claude_model(trimmed_model).lower()
    if normalized_provider == VERTEX_PROVIDER:
        normalized_gemini = extract_vertex_gemini_model_id(trimmed_model)
        if normalized_gemini is not None:
            return normalized_gemini
    return trimmed_model.lower()


def price_search(tool_name: SearchToolName, *, referenceable_results: int) -> float:
    """Return USD cost for a search call based on referenceable result count."""
    if referenceable_results < 0:
        raise ValueError("referenceable_results must be non-negative")
    return float(referenceable_results) * SEARCH_PRICING_PER_REFERENCEABLE_RESULT[tool_name]


def price_embedding(
    provider: EmbeddingProviderName,
    model: str,
    *,
    input_tokens: int | None = None,
    elapsed_seconds: float | None = None,
) -> float:
    """Return USD cost for a miner embedding call under provider-specific static pricing."""
    pricing = MINER_TOOL_EMBEDDING_PRICING[provider][model]
    if pricing.input_per_million is not None:
        if input_tokens is None:
            raise ValueError("input_tokens must be provided for input-token embedding pricing")
        if input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        return (float(input_tokens) / 1_000_000) * pricing.input_per_million
    if pricing.usd_per_second is not None:
        if elapsed_seconds is None:
            raise ValueError("elapsed_seconds must be provided for elapsed-time embedding pricing")
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        return elapsed_seconds * pricing.usd_per_second
    raise ValueError(f"embedding pricing is not configured for provider={provider!r} model={model!r}")


def price_parallel_search(*, billable_results: int) -> float:
    """Return provider-billed USD cost for one Parallel Search request."""
    if billable_results < 0:
        raise ValueError("billable_results must be non-negative")
    extra_results = max(0, billable_results - PARALLEL_SEARCH_BASE_RESULTS)
    return PARALLEL_SEARCH_BASE_COST_USD + (
        float(extra_results) * PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD
    )


def price_parallel_extract(*, url_count: int) -> float:
    """Return provider-billed USD cost for one Parallel Extract request."""
    if url_count < 0:
        raise ValueError("url_count must be non-negative")
    return float(url_count) * PARALLEL_EXTRACT_URL_COST_USD


__all__ = [
    "PARALLEL_EXTRACT_URL_COST_USD",
    "PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD",
    "PARALLEL_SEARCH_BASE_COST_USD",
    "PARALLEL_SEARCH_BASE_RESULTS",
    "GENERATION_MODEL_PRICING",
    "VERTEX_CLAUDE_WEB_SEARCH_PER_1K",
    "VERTEX_GEMINI3_GROUNDED_PER_1K",
    "VERTEX_GROUNDED_PER_1K",
    "generation_usage_cost_breakdown",
    "grounded_cost_usd",
    "lookup_pricing",
    "price_llm",
    "price_miner_llm",
    "price_parallel_extract",
    "price_parallel_search",
    "price_search",
    "pricing_key",
    "MODEL_PRICING",
    "EmbeddingPricing",
    "MINER_TOOL_EMBEDDING_PRICING",
    "MINER_TOOL_LLM_PRICING",
    "SEARCH_PRICING_PER_REFERENCEABLE_RESULT",
    "STATIC_LLM_PRICING",
    "ModelPricing",
    "price_embedding",
    "price_static_llm_model",
]
