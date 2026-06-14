"""Miner llm_chat tool model contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from harnyx_commons.llm.provider_types import (
    CHUTES_PROVIDER,
    CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG,
    VERTEX_PROVIDER,
)

ToolModelName = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "zai-org/GLM-5-TEE",
    "Qwen/Qwen3.6-27B-TEE",
    "google/gemma-4-31B-turbo-TEE",
]

ToolModelThinkingField = Literal[
    "chat_template_kwargs.thinking",
    "chat_template_kwargs.enable_thinking",
]
ToolModelThinkingProvider = Literal["chutes", "vertex", "custom-openai-compatible"]
MinerSelectedLlmProviderName = Literal["chutes", "openrouter"]
MinerSelectedLlmModelName = str

ALLOWED_TOOL_MODELS: tuple[ToolModelName, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "zai-org/GLM-5-TEE",
    "Qwen/Qwen3.6-27B-TEE",
    "google/gemma-4-31B-turbo-TEE",
)

MINER_SELECTED_LLM_PROVIDERS: tuple[MinerSelectedLlmProviderName, ...] = (
    "chutes",
    "openrouter",
)

MINER_SELECTED_LLM_PROVIDER_MODELS: Mapping[
    MinerSelectedLlmProviderName,
    tuple[MinerSelectedLlmModelName, ...],
] = {
    "chutes": (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "zai-org/GLM-5-TEE",
        "Qwen/Qwen3.6-27B-TEE",
        "google/gemma-4-31B-turbo-TEE",
    ),
    "openrouter": (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek/deepseek-v3.2",
        "z-ai/glm-5",
        "qwen/qwen3.6-27b",
        "google/gemma-4-31b-it",
    ),
}


@dataclass(frozen=True)
class ToolModelThinkingCapability:
    field: ToolModelThinkingField

    def chat_template_kwargs(self, *, enabled: bool) -> dict[str, bool]:
        match self.field:
            case "chat_template_kwargs.thinking":
                return {"thinking": enabled}
            case "chat_template_kwargs.enable_thinking":
                return {"enable_thinking": enabled}
        raise AssertionError(f"unsupported thinking field: {self.field}")


@dataclass(frozen=True, slots=True)
class MinerSelectedLlmProviderModel:
    provider: MinerSelectedLlmProviderName
    model: MinerSelectedLlmModelName


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


def parse_miner_selected_llm_provider(raw: str | None) -> MinerSelectedLlmProviderName:
    if raw is None:
        raise ValueError("miner-selected llm provider must be specified")
    value = raw.strip().lower()
    if value not in MINER_SELECTED_LLM_PROVIDERS:
        raise ValueError(f"miner-selected llm provider {value!r} is not supported")
    return cast(MinerSelectedLlmProviderName, value)


def parse_miner_selected_llm_provider_model(
    *,
    provider: str | None,
    model: str | None,
) -> MinerSelectedLlmProviderModel:
    selected_provider = parse_miner_selected_llm_provider(provider)
    if model is None:
        raise ValueError("model must be provided for validator tools")
    selected_model = model.strip()
    if not selected_model:
        raise ValueError("model must be provided for validator tools")
    if selected_model not in MINER_SELECTED_LLM_PROVIDER_MODELS[selected_provider]:
        raise ValueError(
            f"model {selected_model!r} is not supported for miner-selected provider {selected_provider!r}"
        )
    return MinerSelectedLlmProviderModel(provider=selected_provider, model=selected_model)


# Canonical thinking controls for internal llm_chat routes. Miner-selected
# OpenRouter-native ids use OpenRouter reasoning controls directly in the
# OpenRouter provider.
TOOL_MODEL_THINKING_CAPABILITIES: Mapping[
    ToolModelName,
    Mapping[ToolModelThinkingProvider, ToolModelThinkingCapability],
] = {
    "deepseek-ai/DeepSeek-V3.2-TEE": {
        "chutes": ToolModelThinkingCapability("chat_template_kwargs.thinking"),
        "vertex": ToolModelThinkingCapability("chat_template_kwargs.thinking"),
    },
    "zai-org/GLM-5-TEE": {
        "chutes": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "vertex": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "google/gemma-4-31B-turbo-TEE": {
        "chutes": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "custom-openai-compatible": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "Qwen/Qwen3.6-27B-TEE": {
        "chutes": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "custom-openai-compatible": ToolModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
}

_NORMALIZED_TOOL_MODELS: Mapping[str, ToolModelName] = {
    model.lower(): model
    for model in ALLOWED_TOOL_MODELS
}


def resolve_tool_model(raw: str | None) -> ToolModelName | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value in ALLOWED_TOOL_MODELS:
        return cast(ToolModelName, value)
    return _NORMALIZED_TOOL_MODELS.get(value.lower())


def tool_model_thinking_capability(
    raw: str | None,
    *,
    provider_name: str,
) -> ToolModelThinkingCapability | None:
    tool_model = resolve_tool_model(raw)
    if tool_model is None:
        return None
    provider = _tool_model_thinking_provider(provider_name)
    if provider is None:
        return None
    return TOOL_MODEL_THINKING_CAPABILITIES.get(tool_model, {}).get(provider)


def _tool_model_thinking_provider(provider_name: str) -> ToolModelThinkingProvider | None:
    provider = provider_name.strip().lower()
    if provider in {CHUTES_PROVIDER, VERTEX_PROVIDER}:
        return cast(ToolModelThinkingProvider, provider)
    if provider == CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG or provider.startswith(
        f"{CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG}:"
    ):
        return "custom-openai-compatible"
    return None


__all__ = [
    "ALLOWED_TOOL_MODELS",
    "MINER_SELECTED_LLM_PROVIDERS",
    "MINER_SELECTED_LLM_PROVIDER_MODELS",
    "MinerSelectedLlmModelName",
    "MinerSelectedLlmProviderModel",
    "MinerSelectedLlmProviderName",
    "TOOL_MODEL_THINKING_CAPABILITIES",
    "ToolModelName",
    "ToolModelThinkingCapability",
    "ToolModelThinkingField",
    "ToolModelThinkingProvider",
    "parse_miner_selected_llm_provider",
    "parse_miner_selected_llm_provider_model",
    "parse_tool_model",
    "resolve_tool_model",
    "tool_model_thinking_capability",
]
