"""Chutes request and response codec models."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from harnyx_commons.llm.providers.openai_chat_codec import OpenAiChatRequestParts
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiChoiceState,
    OpenAiStreamState,
    OpenAiToolCall,
    _OpenAiStreamEvent,
)
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmUsage,
)

_CHUTES_CONTENT_PARTS_ADAPTER = TypeAdapter(list[object])
_CHUTES_TOOL_CALLS_ADAPTER = TypeAdapter(list[object])
_CHUTES_CHOICES_ADAPTER = TypeAdapter(list[object])


class _ChutesChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    provider: str
    model: str
    messages: list[dict[str, Any]]
    stream: bool = True
    temperature: float | None = None
    max_output_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    include: list[str] | None = None
    response_format: dict[str, Any] | None = None

    @classmethod
    def from_request(cls, request: AbstractLlmRequest) -> _ChutesChatRequest:
        if request.grounded:
            raise ValueError("grounded mode is not supported for chutes provider")
        request_parts = OpenAiChatRequestParts.from_request(
            request,
            provider_name="chutes",
            image_error_message="chutes provider does not support image content parts",
            tool_mix_error_message="chutes input_tool_result messages cannot mix text parts",
            tool_count_error_message="chutes input_tool_result messages must include exactly one part",
        )
        payload = cls(
            provider=request.provider or "chutes",
            model=request.model,
            messages=[message.model_dump(mode="python", exclude_none=True) for message in request_parts.messages],
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            tools=(
                [tool.model_dump(mode="python", exclude_none=True) for tool in request_parts.tools]
                if request_parts.tools
                else None
            ),
            tool_choice=request_parts.tool_choice,
            include=request_parts.include,
            response_format=(
                request_parts.response_format.model_dump(mode="python", exclude_none=True)
                if request_parts.response_format is not None
                else None
            ),
        )
        if request.extra:
            payload = payload.model_copy(update=dict(request.extra))
        return payload.model_copy(update={"stream": True})


class _ChutesTextContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str | None = None
    text: str | None = None


class _ChutesReasoningObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    thought_text_parts: list[str] = Field(default_factory=list)
    has_thought_signature: bool = False
    text: str | None = None
    summary: str | None = None
    content: str | None = None

    @property
    def reasoning_text(self) -> str | None:
        normalized_parts = tuple(part.strip() for part in self.thought_text_parts if part.strip())
        if normalized_parts:
            return "\n\n".join(normalized_parts)

        for candidate in (self.text, self.summary, self.content):
            normalized_text = _normalize_reasoning_text(candidate)
            if normalized_text is not None:
                return normalized_text
        return None

    @property
    def stream_fallback_text(self) -> str | None:
        for candidate in (self.text, self.summary, self.content):
            normalized_text = _normalize_stream_reasoning_text(candidate)
            if normalized_text is not None:
                return normalized_text
        return None

    @classmethod
    def from_stream_state(cls, state: _ChutesReasoningChoiceState) -> _ChutesReasoningObject | None:
        if not state.thought_text_parts and not state.fallback_parts:
            return None
        if state.thought_text_parts:
            return cls(
                thought_text_parts=list(state.thought_text_parts),
                has_thought_signature=state.has_thought_signature,
            )
        fallback_text = "".join(state.fallback_parts)
        return cls(
            text=fallback_text,
            has_thought_signature=state.has_thought_signature,
        )


class _ChutesToolFunctionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = Field(min_length=1)
    arguments: str | dict[str, Any] | None = None


class _ChutesToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    type: str | None = None
    function: _ChutesToolFunctionPayload

    @classmethod
    def from_openai_tool_call(cls, tool_call: OpenAiToolCall) -> _ChutesToolCallPayload:
        return cls(
            id=tool_call.id,
            type=tool_call.type,
            function=_ChutesToolFunctionPayload(
                name=tool_call.name,
                arguments=tool_call.arguments,
            ),
        )

    def to_tool_call(self, *, index: int) -> LlmMessageToolCall:
        match self.function.arguments:
            case str() as arguments:
                serialized_arguments = arguments
            case dict() as arguments:
                serialized_arguments = json.dumps(arguments)
            case _:
                serialized_arguments = ""
        return LlmMessageToolCall(
            id=self.id or f"toolcall-{index}",
            type=self.type or "function",
            name=self.function.name,
            arguments=serialized_arguments,
        )


class _ChutesMessagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    content: str | list[_ChutesTextContentPart] | None = None
    reasoning: str | _ChutesReasoningObject | None = None
    tool_calls: list[_ChutesToolCallPayload] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: object) -> str | list[_ChutesTextContentPart] | None:
        match value:
            case None:
                return None
            case str() as text:
                return text
            case _:
                try:
                    raw_parts = _CHUTES_CONTENT_PARTS_ADAPTER.validate_python(value)
                except ValidationError:
                    return None
                parts: list[_ChutesTextContentPart] = []
                for payload in raw_parts:
                    try:
                        part = _ChutesTextContentPart.model_validate(payload)
                    except ValidationError:
                        continue
                    if part.text is None:
                        continue
                    parts.append(part)
                return parts

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: object) -> str | _ChutesReasoningObject | None:
        match value:
            case None:
                return None
            case str() as text:
                return _normalize_reasoning_text(text)
            case _:
                try:
                    return _ChutesReasoningObject.model_validate(value, strict=True)
                except ValidationError as exc:
                    raise RuntimeError("chutes message reasoning object shape is invalid") from exc

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _normalize_tool_calls(cls, value: object) -> list[_ChutesToolCallPayload] | None:
        if value is None:
            return None
        try:
            raw_tool_calls = _CHUTES_TOOL_CALLS_ADAPTER.validate_python(value)
        except ValidationError as exc:
            raise RuntimeError("chutes message tool_calls must be an array") from exc
        tool_calls: list[_ChutesToolCallPayload] = []
        for payload in raw_tool_calls:
            try:
                tool_call = _ChutesToolCallPayload.model_validate(payload)
            except ValidationError:
                continue
            tool_calls.append(tool_call)
        return tool_calls

    def content_parts(self) -> tuple[LlmMessageContentPart, ...]:
        match self.content:
            case None:
                return ()
            case str() as text:
                return (LlmMessageContentPart(type="text", text=text),)
            case list() as parts:
                return tuple(
                    LlmMessageContentPart(
                        type=part.type or "text",
                        text=part.text,
                    )
                    for part in parts
                    if part.text is not None
                )

    def tool_call_parts(self) -> tuple[LlmMessageToolCall, ...]:
        if not self.tool_calls:
            return ()
        return tuple(tool_call.to_tool_call(index=index) for index, tool_call in enumerate(self.tool_calls))

    def reasoning_text(self) -> str | None:
        match self.reasoning:
            case None:
                return None
            case str() as text:
                return _normalize_reasoning_text(text)
            case _ChutesReasoningObject() as reasoning:
                return reasoning.reasoning_text

    def to_choice_message(self) -> LlmChoiceMessage:
        parts = self.content_parts() or (LlmMessageContentPart(type="text", text=""),)
        return LlmChoiceMessage(
            role="assistant",
            content=parts,
            tool_calls=self.tool_call_parts(),
            reasoning=self.reasoning_text(),
        )


class _ChutesChoicePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    index: int | None = None
    message: _ChutesMessagePayload
    finish_reason: str | None = None

    def to_choice(self, *, index: int) -> LlmChoice:
        return LlmChoice(
            index=self.index if self.index is not None else index,
            message=self.message.to_choice_message(),
            finish_reason=self.finish_reason or "stop",
        )


class _ChutesUsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_usage(self) -> LlmUsage:
        return LlmUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
        )


class _ChutesChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    choices: list[_ChutesChoicePayload] = Field(default_factory=list)
    usage: _ChutesUsagePayload | None = None

    @field_validator("choices", mode="before")
    @classmethod
    def _normalize_choices(cls, value: object) -> list[_ChutesChoicePayload]:
        try:
            raw_choices = _CHUTES_CHOICES_ADAPTER.validate_python(value)
        except ValidationError:
            return []
        choices: list[_ChutesChoicePayload] = []
        for index, payload in enumerate(raw_choices):
            try:
                choice = _ChutesChoicePayload.model_validate(payload)
            except ValidationError:
                continue
            if choice.index is None:
                choice = choice.model_copy(update={"index": index})
            choices.append(choice)
        return choices

    @field_validator("usage", mode="before")
    @classmethod
    def _normalize_usage(cls, value: object) -> _ChutesUsagePayload | None:
        if value is None:
            return None
        try:
            return _ChutesUsagePayload.model_validate(value)
        except ValidationError as exc:
            raise RuntimeError("chutes usage payload must be a JSON object") from exc

    @classmethod
    def from_payload(cls, payload: object) -> _ChutesChatResponse:
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError("chutes chat completions payload must be a JSON object") from exc

    @classmethod
    def from_stream_state(
        cls,
        state: OpenAiStreamState,
        *,
        reasoning_state: _ChutesReasoningStreamState,
    ) -> _ChutesChatResponse:
        choices = [
            _ChutesChoicePayload(
                index=index,
                message=_ChutesMessagePayload(
                    content=choice_state.content_text,
                    reasoning=_ChutesReasoningObject.from_stream_state(reasoning_state.choice(index)),
                    tool_calls=_chutes_stream_tool_calls(choice_state),
                ),
                finish_reason=choice_state.finish_reason,
            )
            for index, choice_state in sorted(state.choices.items())
        ]
        usage = _ChutesUsagePayload.model_validate(state.usage) if state.usage is not None else None
        return cls(id=state.response_id or None, choices=choices, usage=usage)

    def to_llm_response(self) -> LlmResponse:
        choices = tuple(choice.to_choice(index=index) for index, choice in enumerate(self.choices))
        usage = self.usage.to_usage() if self.usage is not None else LlmUsage()
        finish_reason = choices[0].finish_reason if choices else None
        return LlmResponse(
            id=self.id or "",
            choices=choices,
            usage=usage,
            finish_reason=finish_reason,
        )


class _ChutesReasoningChoiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    thought_text_parts: list[str] = Field(default_factory=list)
    fallback_parts: list[str] = Field(default_factory=list)
    has_thought_signature: bool = False

    def merge(self, value: object) -> None:
        match value:
            case None:
                return
            case str() as text:
                normalized = _normalize_stream_reasoning_text(text)
                if normalized is not None:
                    self.fallback_parts.append(normalized)
            case _:
                try:
                    normalized = _ChutesReasoningObject.model_validate(value, strict=True)
                except ValidationError as exc:
                    raise RuntimeError("chutes message reasoning object shape is invalid") from exc
                self.thought_text_parts.extend(normalized.thought_text_parts)
                self.has_thought_signature = self.has_thought_signature or normalized.has_thought_signature
                if not normalized.thought_text_parts and normalized.stream_fallback_text is not None:
                    self.fallback_parts.append(normalized.stream_fallback_text)


class _ChutesReasoningStreamState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    choices: dict[int, _ChutesReasoningChoiceState] = Field(default_factory=dict)

    def choice(self, index: int) -> _ChutesReasoningChoiceState:
        state = self.choices.get(index)
        if state is None:
            state = _ChutesReasoningChoiceState()
            self.choices[index] = state
        return state

    def merge_event(self, event: _OpenAiStreamEvent) -> None:
        for fallback_index, choice_payload in enumerate(event.choices):
            index = choice_payload.index if choice_payload.index is not None else fallback_index
            message_payload = choice_payload.message_delta(reasoning_keys=("reasoning",))
            if message_payload is None:
                continue
            reasoning_payload = (message_payload.model_extra or {}).get("reasoning")
            self.choice(index).merge(reasoning_payload)


def _chutes_stream_tool_calls(choice_state: OpenAiChoiceState) -> list[_ChutesToolCallPayload] | None:
    tool_calls = choice_state.tool_call_values()
    if not tool_calls:
        return None
    result: list[_ChutesToolCallPayload] = []
    for tool_call in tool_calls:
        try:
            result.append(_ChutesToolCallPayload.from_openai_tool_call(tool_call))
        except ValidationError:
            continue
    return result or None


def _normalize_reasoning_text(value: object) -> str | None:
    match value:
        case str() as text if text.strip():
            return text.strip()
        case _:
            return None


def _normalize_stream_reasoning_text(value: object) -> str | None:
    match value:
        case str() as text if text != "":
            return text
        case _:
            return None


def _parse_chutes_response_payload(value: object) -> _ChutesChatResponse:
    return _ChutesChatResponse.from_payload(value)


__all__ = [
    "_ChutesChatRequest",
    "_ChutesChatResponse",
    "_ChutesReasoningStreamState",
    "_parse_chutes_response_payload",
]
