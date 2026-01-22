"""Provider-facing LLM request types shared by platform and validator."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from caster_miner_sdk.llm import (
    LlmChoice,
    LlmChoiceMessage,
    LlmCitation,
    LlmInputContentPart,
    LlmInputImageData,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmTool,
    LlmToolCall,
    LlmUsage,
    PostprocessResult,
    ToolLlmRequest,
)

ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class AbstractLlmRequest(ToolLlmRequest, ABC):
    """Base class for all explicit-provider LLM requests (non-instantiable)."""

    provider: str = ""
    grounded: bool = False
    output_mode: str = "text"
    output_schema: type[BaseModel] | None = None
    postprocessor: Callable[[LlmResponse], PostprocessResult] | None = None
    extra: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None
    include: Sequence[str] | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class GroundedLlmRequest(AbstractLlmRequest):
    """LLM request executed with provider-managed grounding/tools enabled."""

    grounded: Literal[True] = True
    output_mode: Literal["text"] = "text"
    output_schema: None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple guard
        provider = self.provider
        if provider not in {"openai", "vertex"}:
            raise ValueError(f"grounded mode not supported for provider '{self.provider}'")
        if self.model.startswith("gpt-5") and self.temperature is not None:
            raise ValueError("gpt-5 models do not support temperature parameter")


@dataclass(frozen=True)
class LlmRequest(AbstractLlmRequest):
    """Ungrounded LLM request that may ask for JSON/structured output."""

    grounded: Literal[False] = False
    output_mode: Literal["text", "json_object", "structured"] = "text"
    output_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
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
    "LlmMessageContentPart",
    "LlmChoiceMessage",
    "LlmChoice",
    "LlmUsage",
    "LlmCitation",
    "LlmResponse",
    "PostprocessResult",
]
