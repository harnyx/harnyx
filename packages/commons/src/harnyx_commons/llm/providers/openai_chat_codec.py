"""Shared OpenAI-compatible chat request encoding."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

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
    content: str
    tool_call_id: str | None = None
    name: str | None = None

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

        return cls(role=message.role, content="\n".join(text_parts))


class OpenAiChatRequestParts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    messages: list[OpenAiChatMessagePayload]
    tools: list[OpenAiChatToolPayload] | None = None
    tool_choice: str | None = None
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
            tool_choice=request.tool_choice,
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

