from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Final, Literal, TypeAlias

from pydantic import TypeAdapter

INPUT_TEXT_PART_TYPE: Final[Literal["input_text"]] = "input_text"
INPUT_IMAGE_PART_TYPE: Final[Literal["input_image"]] = "input_image"


@dataclass(frozen=True)
class LlmMessage:
    """Single message in a chat-style prompt (request payload)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Sequence[LlmInputContentPart]

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("LlmMessage.content must include at least one content part")
        for part in self.content:
            if isinstance(part, (LlmInputTextPart, LlmInputImagePart)):
                continue
            raise TypeError(f"unsupported request content part type: {type(part).__name__}")


@dataclass(frozen=True)
class LlmTool:
    """Definition of a tool that can be invoked by the model."""

    type: str
    function: Mapping[str, Any] | None = None
    config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolLlmRequest:
    """Miner-facing LLM invocation payload (provider selected by host)."""

    model: str
    messages: Sequence[LlmMessage]
    temperature: float | None
    max_output_tokens: int | None
    tools: Sequence[LlmTool] | None = None
    tool_choice: Literal["auto", "required"] | None = None


@dataclass(frozen=True)
class LlmMessageToolCall:
    """Raw tool call object associated with a single choice message."""

    id: str
    type: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LlmMessageContentPart:
    """Structured content part from a choice message."""

    type: str
    text: str | None = None
    data: Mapping[str, Any] | None = None

    @classmethod
    def input_text(cls, text: str) -> LlmInputTextPart:
        return LlmInputTextPart(text=text)

    @classmethod
    def input_image_url(
        cls,
        url: str,
        *,
        mime_type: str | None = None,
    ) -> LlmInputImagePart:
        return LlmInputImagePart(data=LlmInputImageData(url=url, mime_type=mime_type))


@dataclass(frozen=True)
class LlmInputTextPart:
    text: str
    type: Literal["input_text"] = field(init=False, default=INPUT_TEXT_PART_TYPE)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("input_text parts require text")


@dataclass(frozen=True)
class LlmInputImageData:
    url: str
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise TypeError("input_image url must be a string")
        if not self.url.strip():
            raise ValueError("input_image url must be non-empty")
        if self.mime_type is not None and not isinstance(self.mime_type, str):
            raise TypeError("input_image mime_type must be a string")
        if self.mime_type is not None and not self.mime_type.strip():
            raise ValueError("input_image mime_type must be non-empty when provided")


@dataclass(frozen=True)
class LlmInputImagePart:
    data: LlmInputImageData
    type: Literal["input_image"] = field(init=False, default=INPUT_IMAGE_PART_TYPE)

    def __post_init__(self) -> None:
        if not isinstance(self.data, LlmInputImageData):
            raise TypeError("input_image parts require data")


LlmInputContentPart: TypeAlias = LlmInputTextPart | LlmInputImagePart


@dataclass(frozen=True)
class LlmChoiceMessage:
    """Message payload returned with each choice."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Sequence[LlmMessageContentPart]
    tool_calls: Sequence[LlmMessageToolCall] | None = None
    refusal: Mapping[str, Any] | None = None
    reasoning: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LlmChoice:
    """Single choice entry in an OpenAI-style response."""

    index: int
    message: LlmChoiceMessage
    finish_reason: str | None = None
    logprobs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LlmToolCall:
    """Flattened tool call info retained for compatibility with legacy code."""

    name: str
    arguments: Mapping[str, Any]
    output: str | None = None


@dataclass(frozen=True)
class LlmUsage:
    """Token accounting metadata returned by the provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    web_search_calls: int | None = None

    def __add__(self, other: LlmUsage | None) -> LlmUsage:
        """Combine usages fieldwise, treating ``None`` as zero when present."""

        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        other_usage = other or LlmUsage()
        return LlmUsage(
            prompt_tokens=_sum(self.prompt_tokens, other_usage.prompt_tokens),
            completion_tokens=_sum(self.completion_tokens, other_usage.completion_tokens),
            total_tokens=_sum(self.total_tokens, other_usage.total_tokens),
            prompt_cached_tokens=_sum(self.prompt_cached_tokens, other_usage.prompt_cached_tokens),
            reasoning_tokens=_sum(self.reasoning_tokens, other_usage.reasoning_tokens),
            web_search_calls=_sum(self.web_search_calls, other_usage.web_search_calls),
        )

    def __radd__(self, other: object) -> LlmUsage:
        if other in (0, None):
            return self
        raise TypeError(f"unsupported operand type(s) for +: {type(other)!r} and 'LlmUsage'")

    def __iadd__(self, other: LlmUsage | None) -> LlmUsage:
        return self + other


@dataclass(frozen=True)
class LlmCitation:
    """Provider-agnostic citation entry surfaced alongside responses."""

    url: str
    note: str


@dataclass(frozen=True)
class LlmResponse:
    """Normalized response returned by an LLM provider."""

    id: str
    choices: Sequence[LlmChoice]
    usage: LlmUsage
    metadata: Mapping[str, object] | None = None
    postprocessed: object | None = None
    finish_reason: str | None = None

    @cached_property
    def raw_text(self) -> str | None:
        parts: list[str] = []
        for choice in self.choices:
            for part in choice.message.content:
                if part.text:
                    text = part.text.strip()
                    if text:
                        parts.append(text)
        if not parts:
            return None
        return "\n".join(parts)

    @cached_property
    def tool_calls(self) -> tuple[LlmToolCall, ...]:
        converted: list[LlmToolCall] = []
        for choice in self.choices:
            calls = choice.message.tool_calls or ()
            for call in calls:
                try:
                    args = json.loads(call.arguments)
                    if not isinstance(args, dict):
                        args = {"raw": call.arguments}
                except json.JSONDecodeError:
                    args = {"raw": call.arguments}
                converted.append(
                    LlmToolCall(
                        name=call.name,
                        arguments=dict(args),
                        output=None,
                    ),
                )
        return tuple(converted)

    @cached_property
    def payload(self) -> dict[str, Any]:
        payload = _LLM_RESPONSE_ADAPTER.dump_python(
            self,
            mode="json",
            fallback=lambda value: repr(value),
        )
        if not isinstance(payload, dict):
            msg = "LLM response serialization must produce a JSON object"
            raise TypeError(msg)
        return payload

    def to_payload(self) -> dict[str, Any]:
        return self.payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LlmResponse:
        if not isinstance(payload, Mapping):
            raise TypeError("LLM response payload must be a JSON object")
        return _LLM_RESPONSE_ADAPTER.validate_python(payload)


@dataclass(frozen=True)
class PostprocessResult:
    """Outcome of applying a request-specific postprocessor."""

    ok: bool
    retryable: bool
    reason: str | None = None
    processed: object | None = None


_LLM_RESPONSE_ADAPTER = TypeAdapter(LlmResponse)


__all__ = [
    "LlmInputContentPart",
    "LlmInputImageData",
    "LlmInputImagePart",
    "LlmInputTextPart",
    "LlmMessage",
    "LlmTool",
    "ToolLlmRequest",
    "LlmMessageToolCall",
    "LlmMessageContentPart",
    "LlmChoiceMessage",
    "LlmChoice",
    "LlmToolCall",
    "LlmUsage",
    "LlmCitation",
    "LlmResponse",
    "PostprocessResult",
]
