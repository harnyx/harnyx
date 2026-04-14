"""Shared OpenAI-compatible SSE parsing and accumulation helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class _OpenAiTextFragment(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    text: str | None = None


class _OpenAiFunctionDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str | None = None
    arguments: str | dict[str, Any] | None = None


class _OpenAiToolCallDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: _OpenAiFunctionDelta | None = None


class _OpenAiMessageDelta(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    content: str | list[_OpenAiTextFragment] | None = None
    tool_calls: list[_OpenAiToolCallDelta] | None = None


class _OpenAiChoiceDelta(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    index: int | None = None
    finish_reason: str | None = None
    delta: _OpenAiMessageDelta | None = None
    message: _OpenAiMessageDelta | None = None

    def message_delta(self, *, reasoning_keys: tuple[str, ...]) -> _OpenAiMessageDelta | None:
        if self.delta is not None:
            return self.delta
        if self.message is not None:
            return self.message
        extra = self.model_extra or {}
        payload = {
            key: extra[key]
            for key in ("content", "tool_calls", *reasoning_keys)
            if key in extra
        }
        if not payload:
            return None
        return _OpenAiMessageDelta.model_validate(payload)


class _OpenAiStreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str | None = None
    usage: dict[str, Any] | None = None
    choices: list[_OpenAiChoiceDelta] = Field(default_factory=list)


class _OpenAiStreamErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: str
    type: str | None = None
    code: str | int | None = None


class _OpenAiStreamEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    error: _OpenAiStreamErrorPayload | None = None

    @classmethod
    def from_json(cls, raw_payload: str) -> tuple[_OpenAiStreamEnvelope, object]:
        payload = json.loads(raw_payload)
        match payload:
            case {"error": error_payload}:
                return cls(error=_OpenAiStreamErrorPayload.model_validate(error_payload)), payload
            case {"event": _}:
                raise ValueError("wrapped event envelopes are not supported")
        return cls(), payload


_TEXT_OR_PARTS_ADAPTER = TypeAdapter(str | list[_OpenAiTextFragment] | None)
FragmentNormalizer = Callable[[object], tuple[str, ...]]


def _stream_error_status_code(code: str | int | None) -> int | None:
    match code:
        case int() as status_code:
            return status_code
        case str() as raw if raw.isdigit():
            return int(raw)
        case _:
            return None


class OpenAiStreamError(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        error_type: str | None = None,
        code: str | int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.code = code

    @property
    def retryable(self) -> bool:
        status_code = _stream_error_status_code(self.code)
        if status_code is not None:
            return status_code == 429 or status_code >= 500
        return self.error_type in {"rate_limit_error", "server_error", "overloaded_error"}

    @property
    def reason(self) -> str:
        parts = ["stream_error"]
        if self.code is not None:
            parts.append(str(self.code))
        if self.error_type:
            parts.append(self.error_type)
        parts.append(self.message)
        return ":".join(parts)


class OpenAiToolCallState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments_text: str = ""

    def merge_delta(self, payload: _OpenAiToolCallDelta) -> bool:
        saw_output = False
        if payload.id:
            self.id = payload.id
        if payload.type:
            self.type = payload.type
        function = payload.function
        if function is None:
            return bool(self.id or self.name)
        if function.name:
            self.name = function.name
        match function.arguments:
            case str() as arguments if arguments:
                self.arguments_text += arguments
                saw_output = True
            case dict() as arguments:
                self.arguments_text = json.dumps(arguments)
                saw_output = True
            case _ if self.id or self.name:
                saw_output = True
        return saw_output or bool(self.id or self.name)

    def to_tool_call(self, *, index: int) -> OpenAiToolCall | None:
        if not self.name:
            return None
        return OpenAiToolCall(
            id=self.id or f"toolcall-{index}",
            type=self.type or "function",
            name=self.name,
            arguments=self.arguments_text,
        )


class OpenAiToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    type: str
    name: str
    arguments: str


class OpenAiChoiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content_parts: list[str] = Field(default_factory=list)
    reasoning_parts: list[str] = Field(default_factory=list)
    tool_calls: dict[int, OpenAiToolCallState] = Field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def content_text(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning_parts)

    def merge_delta(
        self,
        payload: _OpenAiChoiceDelta,
        *,
        reasoning_keys: tuple[str, ...],
        normalize_content_fragment: FragmentNormalizer,
        normalize_reasoning_fragment: FragmentNormalizer,
    ) -> bool:
        message_payload = payload.message_delta(reasoning_keys=reasoning_keys)
        if message_payload is None:
            if payload.finish_reason:
                self.finish_reason = payload.finish_reason
            return False

        saw_output = False
        for text in normalize_content_fragment(message_payload.content):
            self.content_parts.append(text)
            saw_output = True
        extra = message_payload.model_extra or {}
        appended_reasoning: set[str] = set()
        for key in reasoning_keys:
            for reasoning in normalize_reasoning_fragment(extra.get(key)):
                if reasoning in appended_reasoning:
                    continue
                appended_reasoning.add(reasoning)
                self.reasoning_parts.append(reasoning)
                saw_output = True
        if self._merge_tool_calls(message_payload.tool_calls):
            saw_output = True
        if payload.finish_reason:
            self.finish_reason = payload.finish_reason
        return saw_output

    def tool_call_values(self) -> tuple[OpenAiToolCall, ...] | None:
        if not self.tool_calls:
            return None
        tool_calls: list[OpenAiToolCall] = []
        for index in sorted(self.tool_calls):
            tool_call = self.tool_calls[index].to_tool_call(index=index)
            if tool_call is None:
                continue
            tool_calls.append(tool_call)
        return tuple(tool_calls) or None

    def _merge_tool_calls(self, payloads: list[_OpenAiToolCallDelta] | None) -> bool:
        if not payloads:
            return False
        saw_output = False
        for fallback_index, payload in enumerate(payloads):
            index = payload.index if payload.index is not None else fallback_index
            state = self.tool_calls.get(index)
            if state is None:
                state = OpenAiToolCallState()
                self.tool_calls[index] = state
            if state.merge_delta(payload):
                saw_output = True
        return saw_output


class OpenAiStreamState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    response_id: str = ""
    usage: dict[str, Any] | None = None
    choices: dict[int, OpenAiChoiceState] = Field(default_factory=dict)

    def choice(self, index: int) -> OpenAiChoiceState:
        choice = self.choices.get(index)
        if choice is None:
            choice = OpenAiChoiceState()
            self.choices[index] = choice
        return choice

    def merge_event(
        self,
        event: _OpenAiStreamEvent,
        *,
        reasoning_keys: tuple[str, ...],
        normalize_content_fragment: FragmentNormalizer | None = None,
        normalize_reasoning_fragment: FragmentNormalizer | None = None,
    ) -> bool:
        if event.id is not None:
            self.response_id = event.id
        if event.usage is not None:
            self.usage = dict(event.usage)

        content_fragment = normalize_content_fragment or normalize_openai_text_fragments
        reasoning_fragment = normalize_reasoning_fragment or normalize_openai_text_fragments
        saw_output = False
        for fallback_index, choice_payload in enumerate(event.choices):
            index = choice_payload.index if choice_payload.index is not None else fallback_index
            if self.choice(index).merge_delta(
                choice_payload,
                reasoning_keys=reasoning_keys,
                normalize_content_fragment=content_fragment,
                normalize_reasoning_fragment=reasoning_fragment,
            ):
                saw_output = True
        return saw_output


async def iter_openai_sse_events(
    response: httpx.Response,
    *,
    invalid_data_message: str,
    invalid_event_message: str,
) -> AsyncIterator[_OpenAiStreamEvent]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            event = _parse_sse_event_payload(
                data_lines,
                invalid_data_message=invalid_data_message,
                invalid_event_message=invalid_event_message,
            )
            data_lines.clear()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    event = _parse_sse_event_payload(
        data_lines,
        invalid_data_message=invalid_data_message,
        invalid_event_message=invalid_event_message,
    )
    if event is not None:
        yield event

def _parse_sse_event_payload(
    data_lines: list[str],
    *,
    invalid_data_message: str,
    invalid_event_message: str,
) -> _OpenAiStreamEvent | None:
    if not data_lines:
        return None
    raw_payload = "\n".join(data_lines).strip()
    if not raw_payload or raw_payload == "[DONE]":
        return None
    try:
        envelope, payload = _OpenAiStreamEnvelope.from_json(raw_payload)
    except ValidationError as exc:
        raise RuntimeError(invalid_event_message) from exc
    except ValueError as exc:
        raise RuntimeError(invalid_data_message) from exc
    if envelope.error is not None:
        raise OpenAiStreamError(
            message=envelope.error.message,
            error_type=envelope.error.type,
            code=envelope.error.code,
        )
    return _OpenAiStreamEvent.model_validate(payload)


def normalize_openai_text_fragments(
    value: object,
    *,
    multipart_joiner: str | None = None,
) -> tuple[str, ...]:
    try:
        normalized = _TEXT_OR_PARTS_ADAPTER.validate_python(value)
    except ValidationError:
        return ()
    match normalized:
        case None:
            return ()
        case str() as text:
            return (text,) if text else ()
        case list() as fragments:
            values: list[str] = []
            for fragment in fragments:
                text = fragment.text
                if not text:
                    continue
                values.append(text)
            if multipart_joiner is None:
                return tuple(values)
            joined = multipart_joiner.join(values)
            return (joined,) if joined else ()
    return ()


__all__ = [
    "OpenAiChoiceState",
    "OpenAiStreamError",
    "OpenAiStreamState",
    "OpenAiToolCall",
    "OpenAiToolCallState",
    "iter_openai_sse_events",
    "normalize_openai_text_fragments",
]
