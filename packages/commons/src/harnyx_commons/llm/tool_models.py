"""Miner llm_chat tool model contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from harnyx_commons.llm.provider_types import (
    AI_GATEWAY_PROVIDER,
    CHUTES_PROVIDER,
    CUSTOM_OPENAI_COMPATIBLE_PROVIDER_TAG,
    OPENROUTER_PROVIDER,
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

ModelThinkingField = Literal[
    "chat_template_kwargs.thinking",
    "chat_template_kwargs.enable_thinking",
]
ModelThinkingProvider = Literal["chutes", "vertex", "custom-openai-compatible"]
ToolModelThinkingField: TypeAlias = ModelThinkingField
ToolModelThinkingProvider: TypeAlias = ModelThinkingProvider
MinerSelectedLlmProviderName = Literal["chutes", "openrouter", "ai_gateway"]
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
    CHUTES_PROVIDER,
    OPENROUTER_PROVIDER,
    AI_GATEWAY_PROVIDER,
)

MINER_SELECTED_LLM_PROVIDER_MODELS: Mapping[
    MinerSelectedLlmProviderName,
    tuple[MinerSelectedLlmModelName, ...],
] = {
    CHUTES_PROVIDER: (
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "moonshotai/Kimi-K2.6-TEE",
        "Qwen/Qwen3.6-27B-TEE",
        "Qwen/Qwen3.8-27B-TEE",
        "google/gemma-4-31B-turbo-TEE",
        "zai-org/GLM-5.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
    ),
    OPENROUTER_PROVIDER: (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek/deepseek-v3.2",
        "z-ai/glm-5",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro",
        "z-ai/glm-5.2",
        "thinkingmachines/inkling",
        "qwen/qwen3.5-397b-a17b",
        "meta/muse-glimmer-30b",
    ),
    AI_GATEWAY_PROVIDER: (
        "thinkingmachines/inkling",
        "zai/glm-5.2-fast",
        "openai/gpt-oss-20b",
        "zai/glm-4.7",
        "google/gemma-4-31b-it",
        "openai/gpt-oss-120b",
        "minimax/minimax-m2.7",
        "zai/glm-4.7-flash",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro",
        "meta/muse-glimmer-30b",
        "alibaba/qwen3.8-27b",
    ),
}


@dataclass(frozen=True)
class ModelThinkingCapability:
    field: ModelThinkingField

    def chat_template_kwargs(self, *, enabled: bool) -> dict[str, bool]:
        match self.field:
            case "chat_template_kwargs.thinking":
                return {"thinking": enabled}
            case "chat_template_kwargs.enable_thinking":
                return {"enable_thinking": enabled}
        raise AssertionError(f"unsupported thinking field: {self.field}")


ToolModelThinkingCapability: TypeAlias = ModelThinkingCapability


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


# Verified provider/model thinking controls. This capability registry is
# independent from validator-tool and miner-selected model authorization.
# OpenRouter-native ids use OpenRouter reasoning controls directly in the
# OpenRouter provider.
MODEL_THINKING_CAPABILITIES: Mapping[
    str,
    Mapping[ModelThinkingProvider, ModelThinkingCapability],
] = {
    "deepseek-ai/DeepSeek-V3.2-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.thinking"),
        "vertex": ModelThinkingCapability("chat_template_kwargs.thinking"),
    },
    "deepseek-ai/DeepSeek-V4-Flash-0731-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "zai-org/GLM-5-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "vertex": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "google/gemma-4-31B-turbo-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "custom-openai-compatible": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "Qwen/Qwen3.6-27B-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
        "custom-openai-compatible": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
    "Qwen/Qwen3.8-27B-TEE": {
        "chutes": ModelThinkingCapability("chat_template_kwargs.enable_thinking"),
    },
}

# Backward-compatible authorization-restricted view for consumers of the
# original tool-model capability API. Provider capabilities outside
# ALLOWED_TOOL_MODELS intentionally remain absent.
TOOL_MODEL_THINKING_CAPABILITIES: Mapping[
    ToolModelName,
    Mapping[ToolModelThinkingProvider, ToolModelThinkingCapability],
] = {
    model: MODEL_THINKING_CAPABILITIES[model]
    for model in ALLOWED_TOOL_MODELS
    if model in MODEL_THINKING_CAPABILITIES
}

_NORMALIZED_TOOL_MODELS: Mapping[str, ToolModelName] = {
    model.lower(): model
    for model in ALLOWED_TOOL_MODELS
}

_NORMALIZED_THINKING_CAPABILITY_MODELS: Mapping[str, str] = {
    model.lower(): model for model in MODEL_THINKING_CAPABILITIES
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


def model_thinking_capability(
    raw: str | None,
    *,
    provider_name: str,
) -> ModelThinkingCapability | None:
    model = _resolve_thinking_capability_model(raw)
    if model is None:
        return None
    provider = _model_thinking_provider(provider_name)
    if provider is None:
        return None
    return MODEL_THINKING_CAPABILITIES[model].get(provider)


def tool_model_thinking_capability(
    raw: str | None,
    *,
    provider_name: str,
) -> ToolModelThinkingCapability | None:
    tool_model = resolve_tool_model(raw)
    if tool_model is None:
        return None
    return model_thinking_capability(tool_model, provider_name=provider_name)


def _resolve_thinking_capability_model(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return _NORMALIZED_THINKING_CAPABILITY_MODELS.get(value.lower())


def _model_thinking_provider(provider_name: str) -> ModelThinkingProvider | None:
    provider = provider_name.strip().lower()
    if provider in {CHUTES_PROVIDER, VERTEX_PROVIDER}:
        return cast(ModelThinkingProvider, provider)
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
    "MODEL_THINKING_CAPABILITIES",
    "TOOL_MODEL_THINKING_CAPABILITIES",
    "ModelThinkingCapability",
    "ModelThinkingField",
    "ModelThinkingProvider",
    "ToolModelName",
    "ToolModelThinkingCapability",
    "ToolModelThinkingField",
    "ToolModelThinkingProvider",
    "parse_miner_selected_llm_provider",
    "parse_miner_selected_llm_provider_model",
    "parse_tool_model",
    "resolve_tool_model",
    "model_thinking_capability",
    "tool_model_thinking_capability",
]
