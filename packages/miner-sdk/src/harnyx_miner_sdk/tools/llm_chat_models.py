"""Strict miner-facing request models for the hosted ``llm_chat`` tool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

from harnyx_miner_sdk.json_types import JsonObject
from harnyx_miner_sdk.llm import (
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageToolCall,
    LlmThinkingConfig,
    LlmTool,
    ToolLlmRequest,
)
from harnyx_miner_sdk.tools.llm_provider_extra import (
    AiGatewayExtra,
    OpenRouterExtra,
    validate_provider_extra,
)
from harnyx_miner_sdk.tools.types import ToolInvocationTimeout

LlmChatProviderName = Literal["chutes", "openrouter", "ai_gateway"]
LlmChatToolChoiceName = Literal["none", "auto", "required"]


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not supported: {value}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LlmChatFunctionDefinition(_StrictModel):
    name: str = Field(min_length=1)
    description: str | None = Field(default=None, min_length=1)
    parameters: JsonObject | None = None
    strict: StrictBool | None = None

    @field_validator("name", "description")
    @classmethod
    def _validate_non_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("function name and description must be non-empty")
        return value


class LlmChatFunctionTool(_StrictModel):
    type: Literal["function"]
    function: LlmChatFunctionDefinition


class LlmChatNamedFunction(_StrictModel):
    name: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("named tool choice requires a non-empty function name")
        return value


class LlmChatNamedToolChoice(_StrictModel):
    type: Literal["function"]
    function: LlmChatNamedFunction


class LlmChatToolCall(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function"]
    name: str = Field(min_length=1)
    arguments: str = Field(min_length=1)

    @field_validator("id", "name")
    @classmethod
    def _validate_non_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool call ID and name must be non-empty")
        return value

    @field_validator("arguments")
    @classmethod
    def _validate_arguments(cls, value: str) -> str:
        try:
            parsed = json.loads(value, parse_constant=_reject_non_finite_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("tool call arguments must encode a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("tool call arguments must encode a JSON object")
        return value

    def to_message_tool_call(self) -> LlmMessageToolCall:
        return LlmMessageToolCall(
            id=self.id,
            type=self.type,
            name=self.name,
            arguments=self.arguments,
        )


class LlmChatSystemMessage(_StrictModel):
    role: Literal["system"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("system message content must be non-empty")
        return value

    def to_message(self) -> LlmMessage:
        return LlmMessage(role=self.role, content=(LlmInputTextPart(text=self.content),))


class LlmChatUserMessage(_StrictModel):
    role: Literal["user"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user message content must be non-empty")
        return value

    def to_message(self) -> LlmMessage:
        return LlmMessage(role=self.role, content=(LlmInputTextPart(text=self.content),))


class LlmChatAssistantMessage(_StrictModel):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: tuple[LlmChatToolCall, ...] | None = None
    reasoning_details: tuple[JsonObject, ...] | None = None

    @model_validator(mode="after")
    def _validate_output(self) -> LlmChatAssistantMessage:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant messages require content or tool_calls")
        if self.content is not None and not self.content.strip() and not self.tool_calls:
            raise ValueError("assistant message content must be non-empty when no tool_calls are present")
        return self

    @model_serializer
    def _serialize(self) -> dict[str, object]:
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            payload["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in self.tool_calls]
        if self.reasoning_details is not None:
            payload["reasoning_details"] = list(self.reasoning_details)
        return payload

    def to_message(self) -> LlmMessage:
        content = (LlmInputTextPart(text=self.content),) if self.content is not None else ()
        return LlmMessage(
            role=self.role,
            content=content,
            tool_calls=(
                tuple(tool_call.to_message_tool_call() for tool_call in self.tool_calls)
                if self.tool_calls is not None
                else None
            ),
            reasoning_details=self.reasoning_details,
        )


class LlmChatToolResultMessage(_StrictModel):
    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)

    @field_validator("tool_call_id", "content", "name")
    @classmethod
    def _validate_non_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("tool result ID, content, and name must be non-empty")
        return value

    def to_message(self) -> LlmMessage:
        return LlmMessage(
            role=self.role,
            content=(
                LlmInputToolResultPart(
                    tool_call_id=self.tool_call_id,
                    name=self.name,
                    output_json=self.content,
                ),
            ),
        )


LlmChatMessage: TypeAlias = Annotated[
    LlmChatSystemMessage | LlmChatUserMessage | LlmChatAssistantMessage | LlmChatToolResultMessage,
    Field(discriminator="role"),
]
LlmChatToolChoice: TypeAlias = LlmChatToolChoiceName | LlmChatNamedToolChoice


class LlmChatThinking(_StrictModel):
    enabled: StrictBool
    budget: StrictInt | None = None
    effort: Literal["low", "medium", "high"] | None = None

    @field_validator("budget")
    @classmethod
    def _validate_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("thinking.budget must be positive")
        return value

    @model_validator(mode="after")
    def _validate_single_tuning_knob(self) -> LlmChatThinking:
        if self.budget is not None and self.effort is not None:
            raise ValueError("thinking.budget and thinking.effort are mutually exclusive")
        return self

    def to_thinking_config(self) -> LlmThinkingConfig:
        return LlmThinkingConfig(enabled=self.enabled, budget=self.budget, effort=self.effort)


class LlmChatRequest(_StrictModel):
    """Canonical SDK-owned JSON request boundary for ``llm_chat``."""

    provider: LlmChatProviderName
    model: str = Field(min_length=1)
    messages: tuple[LlmChatMessage, ...] = Field(min_length=1)
    timeout: ToolInvocationTimeout | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    tools: tuple[LlmChatFunctionTool, ...] | None = None
    tool_choice: LlmChatToolChoice | None = None
    parallel_tool_calls: StrictBool | None = None
    thinking: LlmChatThinking | None = None
    provider_extra: OpenRouterExtra | AiGatewayExtra | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_max_tokens_alias(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = cast(Mapping[str, object], value)
        if "max_tokens" not in data:
            return value
        if data.get("max_output_tokens") is not None and data.get("max_tokens") is not None:
            raise ValueError("max_tokens and max_output_tokens are mutually exclusive")
        normalized = dict(data)
        max_tokens = normalized.pop("max_tokens")
        if normalized.get("max_output_tokens") is None:
            normalized["max_output_tokens"] = max_tokens
        return normalized

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must be non-empty")
        return value

    @model_validator(mode="before")
    @classmethod
    def _validate_provider_extra(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = cast(Mapping[str, object], value)
        if "provider_extra" not in data:
            return value
        provider = data.get("provider")
        if not isinstance(provider, str):
            return value
        parsed = validate_provider_extra(provider=provider, provider_extra=data.get("provider_extra"))
        normalized = dict(data)
        normalized["provider_extra"] = parsed
        return normalized

    @model_validator(mode="after")
    def _validate_tools_and_transcript(self) -> LlmChatRequest:
        declared_names = [tool.function.name for tool in self.tools or ()]
        if len(declared_names) != len(set(declared_names)):
            raise ValueError("function tool names must be unique")
        if isinstance(self.tool_choice, LlmChatNamedToolChoice):
            if self.tool_choice.function.name not in declared_names:
                raise ValueError("named tool_choice must target a declared function")
        self._validate_transcript()
        return self

    def _validate_transcript(self) -> None:
        pending: set[str] = set()
        for message in self.messages:
            if pending:
                if not isinstance(message, LlmChatToolResultMessage):
                    raise ValueError("pending tool calls must be resolved before the next non-tool message")
                if message.tool_call_id not in pending:
                    raise ValueError("tool result must reference one pending tool call exactly once")
                pending.remove(message.tool_call_id)
                continue

            if isinstance(message, LlmChatToolResultMessage):
                raise ValueError("tool result must reference a pending tool call")
            if not isinstance(message, LlmChatAssistantMessage) or not message.tool_calls:
                continue
            call_ids = [tool_call.id for tool_call in message.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool call IDs must be unique within each pending block")
            pending = set(call_ids)

        if pending:
            raise ValueError("all pending tool calls require one contiguous tool result")

    def to_tool_request(self) -> ToolLlmRequest:
        choice: LlmChatToolChoiceName | JsonObject | None
        if isinstance(self.tool_choice, LlmChatNamedToolChoice):
            choice = cast(JsonObject, self.tool_choice.model_dump(mode="json"))
        else:
            choice = self.tool_choice
        return ToolLlmRequest(
            provider=self.provider,
            model=self.model,
            messages=tuple(message.to_message() for message in self.messages),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            tools=(
                tuple(
                    LlmTool(
                        type=tool.type,
                        function=tool.function.model_dump(mode="json", exclude_none=True),
                    )
                    for tool in self.tools
                )
                if self.tools is not None
                else None
            ),
            tool_choice=choice,
            parallel_tool_calls=self.parallel_tool_calls,
            thinking=self.thinking.to_thinking_config() if self.thinking is not None else None,
        )

    def provider_extra_payload(self) -> JsonObject | None:
        if self.provider_extra is None:
            return None
        return cast(JsonObject, self.provider_extra.to_request_extra())


__all__ = [
    "LlmChatAssistantMessage",
    "LlmChatFunctionDefinition",
    "LlmChatFunctionTool",
    "LlmChatMessage",
    "LlmChatNamedToolChoice",
    "LlmChatProviderName",
    "LlmChatRequest",
    "LlmChatSystemMessage",
    "LlmChatThinking",
    "LlmChatToolCall",
    "LlmChatToolChoice",
    "LlmChatToolResultMessage",
    "LlmChatUserMessage",
]
