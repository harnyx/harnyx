"""Shared OpenAI-compatible chat request encoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer

from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmTool,
)


class OpenAiChatToolPayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    type: str
    function: dict[str, Any] | None = None

    @classmethod
    def from_tool(cls, tool: LlmTool) -> OpenAiChatToolPayload:
        if tool.type == "function" and tool.function is not None:
            return cls(type="function", function=dict(tool.function))

        payload = cls(type=tool.type)
        if tool.config:
            payload = payload.model_copy(update=dict(tool.config))
        return payload


class OpenAiChatResponseFormatPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: str
    json_schema: dict[str, Any] | None = None

    @classmethod
    def from_request(
        cls,
        request: AbstractLlmRequest,
        *,
        provider_name: str,
    ) -> OpenAiChatResponseFormatPayload | None:
        match request.output_mode:
            case "text":
                return None
            case "json_object":
                return cls(type="json_object")
            case "structured":
                schema_type = request.output_schema
                if schema_type is None:
                    raise ValueError("structured output requires output_schema")
                return cls(
                    type="json_schema",
                    json_schema={
                        "name": schema_type.__name__,
                        "schema": json_schema_from_model(schema_type),
                    },
                )
            case _:
                raise ValueError(f"unsupported {provider_name} output_mode: {request.output_mode!r}")


class OpenAiChatMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: str
    content: str | None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_details: list[dict[str, Any]] | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        if self.reasoning_details is not None:
            payload["reasoning_details"] = self.reasoning_details
        return payload

    @classmethod
    def from_message(
        cls,
        message: LlmMessage,
        *,
        image_error_message: str,
        tool_mix_error_message: str,
        tool_count_error_message: str,
    ) -> OpenAiChatMessagePayload:
        text_parts: list[str] = []
        tool_results: list[LlmInputToolResultPart] = []
        for part in message.content:
            match part:
                case LlmInputTextPart(text=text):
                    text_parts.append(text)
                case LlmInputToolResultPart() as tool_result:
                    tool_results.append(tool_result)
                case LlmInputImagePart():
                    raise ValueError(image_error_message)
                case _:
                    raise ValueError(f"unsupported request content part type: {part!r}")

        if message.tool_calls:
            if message.role != "assistant" or tool_results:
                raise ValueError("assistant tool_calls cannot be mixed with tool result parts")
            return cls(
                role="assistant",
                content="\n".join(text_parts) or None,
                tool_calls=[
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in message.tool_calls
                ],
                reasoning_details=(
                    [dict(detail) for detail in message.reasoning_details]
                    if message.reasoning_details is not None
                    else None
                ),
            )

        if tool_results:
            if text_parts:
                raise ValueError(tool_mix_error_message)
            if len(tool_results) != 1:
                raise ValueError(tool_count_error_message)
            tool_result = tool_results[0]
            return cls(
                role="tool",
                tool_call_id=tool_result.tool_call_id,
                name=tool_result.name,
                content=tool_result.output_json,
            )

        return cls(
            role=message.role,
            content="\n".join(text_parts),
            reasoning_details=(
                [dict(detail) for detail in message.reasoning_details]
                if message.role == "assistant" and message.reasoning_details is not None
                else None
            ),
        )


class OpenAiChatRequestParts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    messages: list[OpenAiChatMessagePayload]
    tools: list[OpenAiChatToolPayload] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    include: list[str] | None = None
    response_format: OpenAiChatResponseFormatPayload | None = None

    @classmethod
    def from_request(
        cls,
        request: AbstractLlmRequest,
        *,
        provider_name: str,
        image_error_message: str,
        tool_mix_error_message: str,
        tool_count_error_message: str,
    ) -> OpenAiChatRequestParts:
        return cls(
            messages=[
                OpenAiChatMessagePayload.from_message(
                    message,
                    image_error_message=image_error_message,
                    tool_mix_error_message=tool_mix_error_message,
                    tool_count_error_message=tool_count_error_message,
                )
                for message in request.messages
            ],
            tools=[OpenAiChatToolPayload.from_tool(tool) for tool in request.tools] if request.tools else None,
            tool_choice=(
                dict(request.tool_choice) if isinstance(request.tool_choice, Mapping) else request.tool_choice
            ),
            parallel_tool_calls=request.parallel_tool_calls,
            include=list(request.include) if request.include else None,
            response_format=OpenAiChatResponseFormatPayload.from_request(
                request,
                provider_name=provider_name,
            ),
        )


def json_schema_from_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    if not isinstance(schema, dict):  # pragma: no cover - pydantic guarantees dict
        raise TypeError("output_schema must produce a JSON object")
    return dict(schema)
