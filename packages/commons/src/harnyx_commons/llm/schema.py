"""Provider-facing LLM request types shared by platform and validator."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER, parse_custom_openai_compatible_target
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_miner_sdk.llm import (
    LlmChoice,
    LlmChoiceMessage,
    LlmCitation,
    LlmInputContentPart,
    LlmInputImageData,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmThinkingConfig,
    LlmTool,
    LlmToolCall,
    LlmUsage,
    PostprocessRecovery,
    PostprocessResult,
    ToolLlmRequest,
)

ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class AbstractLlmRequest(ToolLlmRequest, ABC):
    """Base class for all explicit-provider LLM requests (non-instantiable)."""

    grounded: bool = False
    output_mode: str = "text"
    output_schema: type[BaseModel] | None = None
    postprocessor: Callable[[LlmResponse], PostprocessResult] | None = None
    use_case: str | None = None
    internal_metadata: Mapping[str, object] | None = None
    extra: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None
    include: Sequence[str] | None = None
    timeout_seconds: float | None = None
    retry_policy: RetryPolicy | None = None
    allow_postprocess_recovery: bool = True

    def __post_init__(self) -> None:
        if self.internal_metadata is not None and "use_case" in self.internal_metadata:
            raise ValueError("LLM request use_case must be provided via the typed use_case field")
        if self.use_case is None:
            return
        normalized_use_case = self.use_case.strip()
        if not normalized_use_case:
            raise ValueError("LLM request use_case must not be blank")
        object.__setattr__(self, "use_case", normalized_use_case)


@dataclass(frozen=True)
class GroundedLlmRequest(AbstractLlmRequest):
    """LLM request executed with provider-managed grounding/tools enabled."""

    grounded: Literal[True] = True
    output_mode: Literal["text"] = "text"
    output_schema: None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple guard
        super().__post_init__()
        if not supports_grounded_requests(provider=self.provider, model=self.model):
            raise ValueError(f"grounded mode not supported for provider/model '{self.provider}:{self.model}'")
        if self.tools and _is_vertex_claude_model(self.model):
            raise ValueError("grounded requests with additional tools are not supported for Vertex Claude models")
        if self.model.startswith("gpt-5") and self.temperature is not None:
            raise ValueError("gpt-5 models do not support temperature parameter")


@dataclass(frozen=True)
class LlmRequest(AbstractLlmRequest):
    """Ungrounded LLM request that may ask for JSON/structured output."""

    grounded: Literal[False] = False
    output_mode: Literal["text", "json_object", "structured"] = "text"
    output_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_mode == "structured" and self.output_schema is None:
            raise ValueError("structured output requires output_schema")
        if self.output_mode != "structured" and self.output_schema is not None:
            raise ValueError("output_schema is only allowed with structured output_mode")
        if self.output_schema is not None:
            schema_type = self.output_schema
            if not isinstance(schema_type, type) or not issubclass(schema_type, BaseModel):
                raise ValueError("output_schema must be a Pydantic BaseModel subclass")
        if self.model.startswith("gpt-5") and self.temperature is not None:
            raise ValueError("gpt-5 models do not support temperature parameter")


__all__ = [
    "ReasoningEffort",
    "LlmMessage",
    "LlmTool",
    "LlmThinkingConfig",
    "ToolLlmRequest",
    "AbstractLlmRequest",
    "GroundedLlmRequest",
    "LlmRequest",
    "LlmToolCall",
    "LlmMessageToolCall",
    "LlmInputContentPart",
    "LlmInputImageData",
    "LlmInputImagePart",
    "LlmInputTextPart",
    "LlmInputToolResultPart",
    "LlmMessageContentPart",
    "LlmChoiceMessage",
    "LlmChoice",
    "LlmUsage",
    "LlmCitation",
    "LlmResponse",
    "PostprocessRecovery",
    "PostprocessResult",
    "extract_vertex_gemini_model_id",
    "supports_grounded_requests",
    "supports_tool_result_messages",
    "supports_grounded_additional_tools",
]


def supports_grounded_requests(*, provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower()
    if normalized_provider != "vertex":
        return False
    if _is_vertex_claude_model(model):
        return _is_vertex_claude_web_search_model(model)
    return _is_vertex_gemini_model(model)


def supports_grounded_additional_tools(*, provider: str, model: str) -> bool:
    return supports_grounded_requests(provider=provider, model=model) and not _is_vertex_claude_model(model)


def supports_tool_result_messages(*, provider: str, model: str) -> bool:
    normalized_provider = provider.strip().lower()
    if parse_custom_openai_compatible_target(normalized_provider) is not None:
        return True
    if normalized_provider == OPENROUTER_PROVIDER:
        return True
    if normalized_provider == "chutes":
        return True
    if normalized_provider == "vertex":
        return not _is_vertex_claude_model(model)
    return False


def _is_vertex_claude_model(model: str) -> bool:
    extracted = _extract_vertex_claude_model_id(model)
    return bool(extracted and extracted.startswith("claude-"))


def _extract_vertex_claude_model_id(model: str) -> str | None:
    normalized = model.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("claude-"):
        return normalized

    for prefix in ("publishers/anthropic/models/", "anthropic/models/"):
        idx = normalized.find(prefix)
        if idx == -1:
            continue
        extracted = normalized[idx + len(prefix) :].strip()
        return extracted or None

    idx = normalized.find("claude-")
    if idx != -1 and (idx == 0 or normalized[idx - 1] == "/"):
        extracted = normalized[idx:].strip()
        return extracted or None
    return None


def _is_vertex_gemini_model(model: str) -> bool:
    return extract_vertex_gemini_model_id(model) is not None


def extract_vertex_gemini_model_id(model: str) -> str | None:
    normalized = model.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("gemini-"):
        return normalized

    for official_prefix in (
        "publishers/google/models/",
        "google/models/",
    ):
        if normalized.startswith(official_prefix):
            extracted = normalized.removeprefix(official_prefix).strip()
            return extracted if extracted.startswith("gemini-") else None

    parts = normalized.split("/")
    if len(parts) != 8:
        return None
    if parts[0] != "projects" or parts[2] != "locations":
        return None
    if parts[4:7] != ["publishers", "google", "models"]:
        return None
    extracted = parts[7].strip()
    return extracted if extracted.startswith("gemini-") else None


def _is_vertex_claude_web_search_model(model: str) -> bool:
    normalized = _extract_vertex_claude_model_id(model)
    if not normalized:
        return False
    for prefix in (
        "claude-opus-4-5@",
        "claude-opus-4-1@",
        "claude-opus-4@",
        "claude-sonnet-4-5@",
        "claude-sonnet-4@",
        "claude-haiku-4-5@",
    ):
        if normalized.startswith(prefix):
            return True
    return False
