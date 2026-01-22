"""Encoding/decoding helpers for Vertex provider."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Sequence
from typing import Any

import httpx
from google.genai import types
from pydantic import BaseModel

from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmInputContentPart,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmTool,
    LlmUsage,
)

_IMAGE_FETCH_TIMEOUT_SECONDS = 20.0


def normalize_messages(messages: Sequence[LlmMessage]) -> tuple[str | None, list[Any]]:
    system_instruction: str | None = None
    converted: list[Any] = []
    for message in messages:
        if message.role == "system":
            system_instruction = _join_text_parts(message.content, label="system")
            continue
        parts = _serialize_vertex_parts(message.content)
        content = types.Content(role=message.role, parts=parts)
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


def resolve_thinking_config(
    *, model: str, reasoning_effort: str | None,
) -> types.ThinkingConfig | None:
    if reasoning_effort is None:
        return None

    effort_raw = reasoning_effort.strip()

    try:
        return types.ThinkingConfig(thinking_budget=int(effort_raw))
    except ValueError:
        return types.ThinkingConfig(thinking_level=types.ThinkingLevel[effort_raw.upper()])


def build_choices(response: types.GenerateContentResponse) -> tuple[LlmChoice, ...]:
    return tuple(
        _choice_from_candidate(idx, candidate)
        for idx, candidate in enumerate(response.candidates or ())
    )


def _choice_from_candidate(index: int, candidate: Any) -> LlmChoice:
    parts, tool_calls = _candidate_parts_and_calls(candidate)
    finish_reason = candidate.finish_reason.value.lower() if candidate.finish_reason is not None else "stop"
    return LlmChoice(
        index=index,
        message=LlmChoiceMessage(
            role="assistant",
            content=tuple(parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
        ),
        finish_reason=finish_reason,
    )


def _candidate_parts_and_calls(candidate: Any) -> tuple[list[LlmMessageContentPart], list[LlmMessageToolCall]]:
    candidate_content = candidate.content
    if candidate_content is None or not candidate_content.parts:
        return [], []

    parts: list[LlmMessageContentPart] = []
    tool_calls: list[LlmMessageToolCall] = []
    for part in candidate_content.parts:
        if part.function_call is not None:
            _append_tool_call(tool_calls, parts, part)
            continue
        parts.append(_text_part(str(part.text or "")))
    return parts, tool_calls


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
            case _:
                raise ValueError(f"unsupported request content part type: {part!r}")
    return converted


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
    "resolve_tool_config",
    "resolve_thinking_config",
    "build_choices",
    "extract_usage",
    "attach_search_metadata",
    "json_schema_from_model",
]
