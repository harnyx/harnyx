"""Encoding/decoding helpers for Vertex provider."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Sequence
from typing import Any

import httpx
from google.genai import types
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from harnyx_commons.llm.provider_types import normalize_reasoning_effort
from harnyx_commons.llm.providers.openai_chat_codec import (
    OpenAiChatRequestParts,
    json_schema_from_model,
)
from harnyx_commons.llm.providers.openai_stream import OpenAiChoiceState, OpenAiStreamState
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputContentPart,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmTool,
    LlmUsage,
)

_IMAGE_FETCH_TIMEOUT_SECONDS = 20.0
_STRING_KEY_MAPPING_ADAPTER = TypeAdapter(dict[str, object])
class _VertexMaasChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str
    stream: bool = True
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    reasoning_effort: str | None = None
    include: list[str] | None = None
    response_format: dict[str, Any] | None = None

    @classmethod
    def from_request(cls, request: AbstractLlmRequest) -> _VertexMaasChatRequest:
        request_parts = OpenAiChatRequestParts.from_request(
            request,
            provider_name="Vertex MaaS",
            image_error_message="vertex-maas GPT OSS requests do not support image content parts",
            tool_mix_error_message="vertex-maas tool messages cannot mix text and input_tool_result parts",
            tool_count_error_message="vertex-maas tool messages must include exactly one input_tool_result part",
        )
        payload = cls(
            model=vertex_maas_openai_chat_model_name(request.model),
            messages=[message.model_dump(mode="python", exclude_none=True) for message in request_parts.messages],
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            tools=(
                [tool.model_dump(mode="python", exclude_none=True) for tool in request_parts.tools]
                if request_parts.tools
                else None
            ),
            tool_choice=request_parts.tool_choice,
            reasoning_effort=request.reasoning_effort,
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


class _VertexMaasToolFunctionPayload(BaseModel):
    name: str
    arguments: object | None = None


class _VertexMaasToolCallPayload(BaseModel):
    id: str | None = None
    function: _VertexMaasToolFunctionPayload


class _VertexMaasTextPart(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str | None = None
    text: str | None = None


_VERTEX_MAAS_TEXT_ADAPTER = TypeAdapter(str | list[_VertexMaasTextPart] | None)


class _VertexMaasChoicePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    index: int | None = None
    content: str | list[_VertexMaasTextPart] | None = None
    reasoning_content: str | list[_VertexMaasTextPart] | None = None
    reasoning: str | None = None
    tool_calls: list[_VertexMaasToolCallPayload] | None = None
    finish_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_message_payload(cls, value: object) -> object:
        try:
            payload = _STRING_KEY_MAPPING_ADAPTER.validate_python(value)
        except ValidationError:
            return value
        message_payload = payload.get("message")
        if message_payload is None:
            return payload
        try:
            message = _STRING_KEY_MAPPING_ADAPTER.validate_python(message_payload)
        except ValidationError:
            return payload
        return {**payload, **message}

    @classmethod
    def from_stream_choice(cls, *, index: int, state: OpenAiChoiceState) -> _VertexMaasChoicePayload:
        return cls(
            index=index,
            content=state.content_text,
            reasoning_content=state.reasoning_text or None,
            tool_calls=_vertex_maas_tool_call_payloads(state),
            finish_reason=state.finish_reason,
        )

    def text_content(self) -> str | None:
        return _extract_vertex_maas_text(self.content, require_text_type=True)

    def reasoning_text(self) -> str | None:
        return _extract_vertex_maas_text(self.reasoning_content) or _extract_vertex_maas_text(self.reasoning)

    def to_choice(self, *, index: int) -> LlmChoice:
        return LlmChoice(
            index=self.index if self.index is not None else index,
            message=LlmChoiceMessage(
                role="assistant",
                content=((_text_part(text_content),) if (text_content := self.text_content()) else ()),
                tool_calls=_vertex_maas_tool_calls(self.tool_calls),
                reasoning=self.reasoning_text(),
            ),
            finish_reason=self.finish_reason or "stop",
        )

    def raw_payload(self) -> dict[str, Any]:
        message_payload = {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "reasoning": self.reasoning,
            "tool_calls": (
                [tool_call.model_dump(mode="python", exclude_none=True) for tool_call in self.tool_calls]
                if self.tool_calls
                else None
            ),
        }
        payload = {
            "index": self.index,
            "finish_reason": self.finish_reason,
            "message": {
                key: value
                for key, value in message_payload.items()
                if value is not None
            },
        }
        return {key: value for key, value in payload.items() if value is not None}


class _VertexMaasUsagePayload(BaseModel):
    prompt_tokens: int | None = None
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class _VertexMaasChatResponse(BaseModel):
    id: str | None = None
    choices: list[_VertexMaasChoicePayload]
    usage: _VertexMaasUsagePayload | None = None

    @classmethod
    def from_stream_state(cls, state: OpenAiStreamState) -> _VertexMaasChatResponse:
        choices = [
            _VertexMaasChoicePayload.from_stream_choice(index=index, state=choice_state)
            for index, choice_state in sorted(state.choices.items())
        ]
        usage = _VertexMaasUsagePayload.model_validate(state.usage) if state.usage is not None else None
        return cls(id=state.response_id or None, choices=choices, usage=usage)

    def to_llm_response(self) -> LlmResponse:
        choices = tuple(choice.to_choice(index=index) for index, choice in enumerate(self.choices))
        usage_payload = self.usage
        usage = LlmUsage(
            prompt_tokens=usage_payload.prompt_tokens if usage_payload else None,
            prompt_cached_tokens=(
                usage_payload.prompt_tokens_details.get("cached_tokens")
                if usage_payload and usage_payload.prompt_tokens_details
                else None
            ),
            completion_tokens=usage_payload.completion_tokens if usage_payload else None,
            total_tokens=usage_payload.total_tokens if usage_payload else None,
            reasoning_tokens=usage_payload.reasoning_tokens if usage_payload else None,
        )
        response_id = self.id or ""
        finish_reason = choices[0].finish_reason if choices else None
        return LlmResponse(
            id=response_id,
            choices=choices,
            usage=usage,
            finish_reason=finish_reason,
        )

    def raw_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "choices": [choice.raw_payload() for choice in self.choices],
            "usage": self.usage.model_dump(mode="python", exclude_none=True) if self.usage is not None else None,
        }


def _to_vertex_request_role(role: str) -> str:
    if role == "user":
        return "user"
    if role == "assistant":
        return "model"
    if role == "tool":
        return "user"
    raise ValueError(f"unsupported Vertex request role: {role!r}")


def normalize_messages(messages: Sequence[LlmMessage]) -> tuple[str | None, list[Any]]:
    system_instruction: str | None = None
    converted: list[Any] = []
    for message in messages:
        if message.role == "system":
            system_instruction = _join_text_parts(message.content, label="system")
            continue
        parts = _serialize_vertex_parts(message.content)
        content = types.Content(role=_to_vertex_request_role(message.role), parts=parts)
        converted.append(content)
    return system_instruction, converted


def serialize_tools(tools: Sequence[LlmTool] | None) -> list[types.Tool] | None:
    if not tools:
        return []
    serialized: list[types.Tool] = []
    for tool in tools:
        if tool.type == "function":
            if tool.function is None:
                raise ValueError("function tool requires 'function' metadata")
            serialized.append(types.Tool(function_declarations=[types.FunctionDeclaration(**tool.function)]))
        else:
            serialized.append(types.Tool())
    return serialized


def serialize_provider_native_tools(tools: Sequence[LlmTool] | None) -> list[types.Tool]:
    if not tools:
        return []
    serialized: list[types.Tool] = []
    for tool in tools:
        if tool.config is None:
            raise ValueError("provider-native Vertex tools require config payload")
        serialized.append(types.Tool(**dict(tool.config)))
    return serialized


def vertex_maas_openai_chat_model_name(model: str) -> str:
    normalized = model.strip()
    prefix = "publishers/"
    models_marker = "/models/"
    if normalized.startswith(prefix) and models_marker in normalized:
        publisher_and_model = normalized[len(prefix) :]
        publisher, model_name = publisher_and_model.split(models_marker, 1)
        return f"{publisher}/{model_name}"
    return normalized


def _extract_vertex_maas_text(
    value: object,
    *,
    require_text_type: bool = False,
) -> str | None:
    try:
        normalized = _VERTEX_MAAS_TEXT_ADAPTER.validate_python(value)
    except ValidationError:
        return None
    match normalized:
        case None:
            return None
        case str() as text:
            stripped = text.strip()
            return stripped or None
        case list() as parts:
            text_fragments = [
                text.strip()
                for part in parts
                if (not require_text_type or part.type == "text")
                if (text := part.text)
                if text.strip()
            ]
            return "\n\n".join(text_fragments) or None
    return None


def _vertex_maas_tool_calls(
    value: list[_VertexMaasToolCallPayload] | None,
) -> tuple[LlmMessageToolCall, ...] | None:
    if not value:
        return None
    return tuple(
        LlmMessageToolCall(
            id=tool_call.id or f"toolcall-{index}",
            type="function",
            name=tool_call.function.name,
            arguments=(
                tool_call.function.arguments
                if isinstance(tool_call.function.arguments, str)
                else json.dumps(tool_call.function.arguments or {})
            ),
        )
        for index, tool_call in enumerate(value)
    )


def _vertex_maas_tool_call_payloads(state: OpenAiChoiceState) -> list[_VertexMaasToolCallPayload] | None:
    tool_calls = state.tool_call_values()
    if not tool_calls:
        return None
    return [
        _VertexMaasToolCallPayload(
            id=tool_call.id,
            function=_VertexMaasToolFunctionPayload(
                name=tool_call.name,
                arguments=tool_call.arguments,
            ),
        )
        for tool_call in tool_calls
    ]


def resolve_tool_config(
    choice: str | None,
    tools: Sequence[types.Tool] | None,
) -> types.ToolConfig | None:
    if not tools:
        return None
    if not choice:
        return types.ToolConfig()
    if choice == "auto":
        return types.ToolConfig()
    if choice == "required":
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
            ),
        )
    return None


def supports_thinking_config(*, model: str) -> bool:
    normalized_model = model.strip().lower()
    return "gemini" in normalized_model


def resolve_thinking_config(
    *, model: str, reasoning_effort: str | None,
) -> types.ThinkingConfig | None:
    include_thoughts = True
    normalized_effort = normalize_reasoning_effort(reasoning_effort)
    if normalized_effort is None:
        return None

    try:
        effort_value = int(normalized_effort)
        return types.ThinkingConfig(
            thinking_budget=effort_value,
            include_thoughts=include_thoughts,
        )
    except ValueError:
        # Named reasoning levels intentionally fall through to the enum lookup below.
        return types.ThinkingConfig(
            thinking_level=types.ThinkingLevel[normalized_effort.upper()],
            include_thoughts=include_thoughts,
        )


def build_choices(response: types.GenerateContentResponse) -> tuple[LlmChoice, ...]:
    return tuple(
        _choice_from_candidate(idx, candidate)
        for idx, candidate in enumerate(response.candidates or ())
    )


def _choice_from_candidate(index: int, candidate: Any) -> LlmChoice:
    parts, tool_calls, reasoning = _candidate_parts_and_calls(candidate)
    finish_reason = candidate.finish_reason.value.lower() if candidate.finish_reason is not None else "stop"
    return LlmChoice(
        index=index,
        message=LlmChoiceMessage(
            role="assistant",
            content=tuple(parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
            reasoning=reasoning,
        ),
        finish_reason=finish_reason,
    )


def _candidate_parts_and_calls(
    candidate: Any,
) -> tuple[list[LlmMessageContentPart], list[LlmMessageToolCall], str | None]:
    candidate_content = candidate.content
    if candidate_content is None or not candidate_content.parts:
        return [], [], None

    parts: list[LlmMessageContentPart] = []
    tool_calls: list[LlmMessageToolCall] = []
    thought_text_parts: list[str] = []
    for part in candidate_content.parts:
        if part.function_call is not None:
            _append_tool_call(tool_calls, parts, part)
            continue

        is_reasoning_part = bool(part.thought)
        if is_reasoning_part:
            text_value = str(part.text or "")
            if text_value:
                thought_text_parts.append(text_value)
            continue

        parts.append(_text_part(str(part.text or "")))
    reasoning = _reasoning_text(thought_text_parts)
    return parts, tool_calls, reasoning


def _reasoning_text(thought_text_parts: list[str]) -> str | None:
    normalized_parts = tuple(part.strip() for part in thought_text_parts if part.strip())
    if not normalized_parts:
        return None
    return "\n\n".join(normalized_parts)


def _append_tool_call(
    tool_calls: list[LlmMessageToolCall],
    parts: list[LlmMessageContentPart],
    part: Any,
) -> None:
    fn = part.function_call
    tool_call_id = fn.id or fn.name or f"toolcall-{len(tool_calls)}"
    tool_name = fn.name or ""
    tool_calls.append(
        LlmMessageToolCall(
            id=str(tool_call_id),
            type="function",
            name=str(tool_name),
            arguments=json.dumps(fn.args or {}),
        ),
    )
    text_value = part.text
    if text_value:
        parts.append(_text_part(str(text_value)))


def _text_part(text: str) -> LlmMessageContentPart:
    return LlmMessageContentPart(type="text", text=text)


def _join_text_parts(parts: Sequence[LlmInputContentPart], *, label: str) -> str:
    fragments: list[str] = []
    for part in parts:
        match part:
            case LlmInputTextPart(text=text):
                fragments.append(text)
            case LlmInputImagePart():
                raise ValueError(f"{label} messages do not support input_image content parts")
            case LlmInputToolResultPart():
                raise ValueError(f"{label} messages do not support input_tool_result content parts")
            case _:
                raise ValueError(f"unsupported request content part type: {part!r}")
    return "\n".join(fragments)


def _serialize_vertex_parts(parts: Sequence[LlmInputContentPart]) -> list[Any]:
    converted: list[Any] = []
    for part in parts:
        match part:
            case LlmInputTextPart(text=text):
                converted.append(types.Part.from_text(text=text))
            case LlmInputImagePart() as image_part:
                converted.append(_serialize_vertex_image_part(image_part))
            case LlmInputToolResultPart() as tool_result_part:
                converted.append(_serialize_vertex_tool_result_part(tool_result_part))
            case _:
                raise ValueError(f"unsupported request content part type: {part!r}")
    return converted


def _serialize_vertex_tool_result_part(part: LlmInputToolResultPart) -> Any:
    try:
        parsed = json.loads(part.output_json)
    except json.JSONDecodeError as exc:
        raise ValueError("input_tool_result output_json must be valid JSON") from exc

    if isinstance(parsed, dict):
        payload: dict[str, Any] = dict(parsed)
    else:
        payload = {"value": parsed}

    payload["tool_call_id"] = part.tool_call_id
    return types.Part.from_function_response(
        name=part.name,
        response=payload,
    )


def _serialize_vertex_image_part(part: LlmInputImagePart) -> Any:
    data = part.data
    url = data.url
    mime_type = data.mime_type
    if url.startswith("gs://"):
        mime_type = mime_type or _infer_mime_type_from_url(url)
        if mime_type is None:
            raise ValueError(f"unable to infer mime_type for GCS image url: {url!r}")
        return types.Part.from_uri(file_uri=url, mime_type=mime_type)

    image_bytes, resolved_mime_type = _download_image(url)
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type or resolved_mime_type)


def _download_image(url: str) -> tuple[bytes, str]:
    with httpx.Client(follow_redirects=True, timeout=_IMAGE_FETCH_TIMEOUT_SECONDS) as client:
        response = client.get(url)
        response.raise_for_status()
        content = response.content
        if not content:
            raise RuntimeError(f"empty image response from url: {url!r}")

        content_type = response.headers.get("content-type")
        mime_type = _parse_mime_type(content_type) or _infer_mime_type_from_url(url)
        if mime_type is None:
            raise ValueError(f"unable to determine mime_type for image url: {url!r}")
        return content, mime_type


def _parse_mime_type(value: str | None) -> str | None:
    if not value:
        return None
    mime = value.split(";", 1)[0].strip().lower()
    return mime or None


def _infer_mime_type_from_url(url: str) -> str | None:
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type:
        return mime_type
    return None


def extract_usage(usage_metadata: types.GenerateContentResponseUsageMetadata | None) -> LlmUsage | None:
    if usage_metadata is None:
        return None
    return LlmUsage(
        prompt_tokens=usage_metadata.prompt_token_count,
        prompt_cached_tokens=usage_metadata.cached_content_token_count,
        completion_tokens=usage_metadata.candidates_token_count,
        total_tokens=usage_metadata.total_token_count,
        reasoning_tokens=usage_metadata.thoughts_token_count,
    )


def collect_search_queries(response: Any) -> list[str]:
    queries: list[str] = []
    for candidate in response.candidates or ():
        metadata = candidate.grounding_metadata
        if metadata is None or not metadata.web_search_queries:
            continue
        for entry in metadata.web_search_queries:
            if entry.strip():
                queries.append(entry.strip())
    return queries


def attach_search_metadata(
    source: Any,
    usage: LlmUsage,
) -> tuple[dict[str, Any] | None, LlmUsage]:
    queries = source if isinstance(source, list) else collect_search_queries(source)
    calls = len(queries)
    usage = usage + LlmUsage(web_search_calls=calls)
    if calls:
        return {
            "web_search_calls": calls,
            "web_search_queries": tuple(queries),
        }, usage
    return None, usage

__all__ = [
    "normalize_messages",
    "serialize_tools",
    "serialize_provider_native_tools",
    "resolve_tool_config",
    "supports_thinking_config",
    "resolve_thinking_config",
    "build_choices",
    "extract_usage",
    "attach_search_metadata",
    "_VertexMaasChatRequest",
    "_VertexMaasChatResponse",
    "json_schema_from_model",
]
