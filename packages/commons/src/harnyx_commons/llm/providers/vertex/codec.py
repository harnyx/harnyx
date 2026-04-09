"""Encoding/decoding helpers for Vertex provider."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from google.genai import types
from pydantic import BaseModel

from harnyx_commons.llm.provider_types import normalize_reasoning_effort
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


class _VertexMaasRequestPayload(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    reasoning_effort: str | None = None
    include: list[str] | None = None
    response_format: dict[str, Any] | None = None


class _VertexMaasToolFunctionPayload(BaseModel):
    name: str
    arguments: object | None = None


class _VertexMaasToolCallPayload(BaseModel):
    id: str | None = None
    function: _VertexMaasToolFunctionPayload


class _VertexMaasChoiceMessagePayload(BaseModel):
    content: str | list[dict[str, Any]] | None = None
    reasoning_content: str | list[dict[str, Any]] | None = None
    reasoning: str | None = None
    tool_calls: list[_VertexMaasToolCallPayload] | None = None


class _VertexMaasChoicePayload(BaseModel):
    message: _VertexMaasChoiceMessagePayload
    finish_reason: str | None = None


class _VertexMaasUsagePayload(BaseModel):
    prompt_tokens: int | None = None
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class _VertexMaasResponsePayload(BaseModel):
    id: str | None = None
    choices: list[_VertexMaasChoicePayload]
    usage: _VertexMaasUsagePayload | None = None


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


def build_vertex_maas_chat_payload(request: AbstractLlmRequest) -> dict[str, Any]:
    response_format: dict[str, Any] | None = None
    match request.output_mode:
        case "text":
            response_format = None
        case "json_object":
            response_format = {"type": "json_object"}
        case "structured":
            schema_type = request.output_schema
            if schema_type is None:
                raise ValueError("structured output requires output_schema")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_type.__name__,
                    "schema": json_schema_from_model(schema_type),
                },
            }
        case _:
            raise ValueError(f"unsupported Vertex MaaS output_mode: {request.output_mode!r}")

    payload = _VertexMaasRequestPayload(
        model=vertex_maas_openai_chat_model_name(request.model),
        messages=[_serialize_vertex_maas_chat_message(message) for message in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_output_tokens,
        tools=serialize_vertex_maas_openai_tools(request.tools),
        tool_choice=request.tool_choice,
        reasoning_effort=request.reasoning_effort,
        include=list(request.include) if request.include else None,
        response_format=response_format,
    ).model_dump(exclude_none=True)
    if request.extra:
        payload.update(dict(request.extra))
    return payload


def serialize_vertex_maas_openai_tools(tools: Sequence[LlmTool] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    serialized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.type == "function" and tool.function is not None:
            serialized.append(
                {
                    "type": "function",
                    "function": dict(tool.function),
                }
            )
            continue

        tool_payload: dict[str, Any] = {"type": tool.type}
        if tool.config:
            tool_payload.update(dict(tool.config))
        serialized.append(tool_payload)
    return serialized


def parse_vertex_maas_chat_response(payload: Mapping[str, Any]) -> LlmResponse:
    parsed = _VertexMaasResponsePayload.model_validate(payload)
    choices = tuple(
        LlmChoice(
            index=index,
            message=LlmChoiceMessage(
                role="assistant",
                content=(
                    (_text_part(content_text),)
                    if (
                        content_text := _extract_vertex_maas_text(
                            choice.message.content,
                            require_text_type=True,
                        )
                    )
                    else ()
                ),
                tool_calls=_vertex_maas_tool_calls(choice.message.tool_calls),
                reasoning=(
                    _extract_vertex_maas_text(choice.message.reasoning_content)
                    or _extract_vertex_maas_text(choice.message.reasoning)
                ),
            ),
            finish_reason=choice.finish_reason or "stop",
        )
        for index, choice in enumerate(parsed.choices)
    )
    usage_payload = parsed.usage
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
    response_id = parsed.id or ""
    finish_reason = choices[0].finish_reason if choices else None
    return LlmResponse(
        id=response_id,
        choices=choices,
        usage=usage,
        finish_reason=finish_reason,
    )


def _serialize_vertex_maas_chat_message(message: LlmMessage) -> dict[str, Any]:
    fragments: list[str] = []
    tool_results: list[LlmInputToolResultPart] = []
    for part in message.content:
        match part:
            case LlmInputTextPart(text=text):
                fragments.append(text)
            case LlmInputToolResultPart() as tool_result:
                tool_results.append(tool_result)
            case LlmInputImagePart():
                raise ValueError("vertex-maas GPT OSS requests do not support image content parts")
            case _:
                raise ValueError(f"unsupported Vertex MaaS request content part type: {part!r}")

    if tool_results:
        if fragments:
            raise ValueError("vertex-maas tool messages cannot mix text and input_tool_result parts")
        if len(tool_results) != 1:
            raise ValueError("vertex-maas tool messages must include exactly one input_tool_result part")
        tool_result = tool_results[0]
        return {
            "role": "tool",
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "content": tool_result.output_json,
        }

    return {
        "role": message.role,
        "content": "\n".join(fragments),
    }


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
    value: str | list[dict[str, Any]] | None,
    *,
    require_text_type: bool = False,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    text_fragments = [
        text.strip()
        for item in value
        if not require_text_type or item.get("type") == "text"
        if isinstance((text := item.get("text")), str)
        if text.strip()
    ]
    return "\n\n".join(text_fragments) or None


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


def json_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    if not isinstance(schema, dict):  # pragma: no cover - pydantic guarantees dict
        raise TypeError("output_schema must produce a JSON object")
    return dict(schema)


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
    "json_schema_from_model",
]
