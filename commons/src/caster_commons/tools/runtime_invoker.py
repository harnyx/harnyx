"""Tool invocation dispatch shared by platform and validator."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict
from pydantic import JsonValue as PydanticJsonValue

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.json_types import JsonObject, JsonValue
from caster_commons.llm.pricing import (
    ALLOWED_TOOL_MODELS,
    ToolModelName,
    parse_tool_model,
)
from caster_commons.llm.provider import LlmProviderPort
from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmTool
from caster_commons.tools.executor import ToolInvoker
from caster_commons.tools.normalize import normalize_response
from caster_commons.tools.ports import DeSearchPort
from caster_commons.tools.search_models import SearchWebSearchRequest, SearchXSearchRequest
from caster_commons.tools.types import SearchToolName, ToolName, is_search_tool
from caster_commons.tools.usage_tracker import ToolCallUsage  # noqa: F401 - compatibility


class LlmToolMessage(BaseModel):
    """Message format for LLM tool invocations."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LlmToolInvocation(BaseModel):
    """Request payload for llm_chat tool calls."""

    model: str
    messages: tuple[LlmToolMessage, ...]
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    response_format: str = "text"
    tools: tuple[dict[str, PydanticJsonValue], ...] | None = None
    tool_choice: Literal["auto", "required"] | None = None
    include: tuple[str, ...] | None = None
    reasoning_effort: str | None = None

    model_config = ConfigDict(extra="allow")


class RuntimeToolInvoker(ToolInvoker):
    """Dispatches sandbox tool invocations."""

    def __init__(
        self,
        receipt_log: ReceiptLogPort,
        *,
        search_client: DeSearchPort | None = None,
        llm_provider: LlmProviderPort | None = None,
        llm_provider_name: str | None = None,
        allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
    ) -> None:
        self._receipts = receipt_log
        self._logger = logging.getLogger("caster_tools.invoker")
        self._search = search_client
        self._llm_provider = llm_provider
        self._llm_provider_name = llm_provider_name or "llm"
        self._allowed_models: set[ToolModelName] = set(allowed_models)

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if tool_name == "test_tool":
            return self._invoke_test_tool(args, kwargs)
        if is_search_tool(tool_name):
            return await self._dispatch_search(tool_name, args, kwargs)
        if tool_name == "llm_chat":
            return await self._dispatch_llm(args, kwargs)
        self._log_unhandled(tool_name, args, kwargs)
        raise LookupError(f"tool {tool_name!r} is not registered")

    def _invoke_test_tool(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        message: str = ""
        if args:
            message = str(args[0])
        if "message" in kwargs:
            message = str(kwargs["message"])

        self._logger.info("test_tool message: %s", message)
        return {
            "status": "ok",
            "echo": message,
        }

    def _log_unhandled(
        self,
        tool_name: ToolName | str,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> None:
        self._logger.info(
            "unhandled tool requested",
            extra={
                "tool": tool_name,
                "args": tuple(args),
                "kwargs": dict(kwargs),
            },
        )

    @normalize_response
    async def _dispatch_search(
        self,
        tool_name: SearchToolName,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if self._search is None:
            raise LookupError("search client is not configured")
        payload = self._payload_from_args_kwargs(args, kwargs)
        if "query" not in payload and "prompt" in payload:
            payload = dict(payload)
            payload["query"] = payload.pop("prompt")
        if tool_name == "search_web":
            request_model_web = SearchWebSearchRequest.model_validate(payload)
            response_web = await self._search.search_links_web(request_model_web)
            as_mapping = response_web.model_dump(exclude_none=True, mode="json")
            return cast(JsonObject, as_mapping)
        elif tool_name == "search_x":
            request_model_x = SearchXSearchRequest.model_validate(payload)
            response_x = await self._search.search_links_twitter(request_model_x)
            as_mapping = response_x.model_dump(exclude_none=True, mode="json")
            return cast(JsonObject, as_mapping)
        raise LookupError(f"search tool '{tool_name}' is not supported")

    async def _dispatch_llm(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if self._llm_provider is None:
            raise LookupError("llm provider is not configured")

        invocation = self._parse_invocation(args, kwargs)
        messages = self._normalize_messages(invocation)
        tools = self._normalize_tools(invocation)
        max_output_tokens = invocation.max_output_tokens or invocation.max_tokens

        request = self._build_llm_request(
            invocation,
            messages,
            tools,
            max_output_tokens,
        )

        llm_response = await self._llm_provider.invoke(request)
        return cast(JsonObject, llm_response.to_payload())

    def _parse_invocation(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> LlmToolInvocation:
        payload = dict(self._payload_from_args_kwargs(args, kwargs))
        invocation = LlmToolInvocation.model_validate(payload)
        self._assert_allowed_model(invocation.model)
        return invocation

    def _assert_allowed_model(self, model: str | None) -> None:
        parsed = parse_tool_model(model)
        if parsed not in self._allowed_models:
            raise ValueError(f"model {parsed!r} is not allowed for validator tools")

    @staticmethod
    def _normalize_messages(invocation: LlmToolInvocation) -> tuple[LlmMessage, ...]:
        return tuple(
            LlmMessage(
                role=message.role,
                content=(LlmMessageContentPart.input_text(message.content),),
            )
            for message in invocation.messages
        )

    @staticmethod
    def _normalize_tools(invocation: LlmToolInvocation) -> tuple[LlmTool, ...] | None:
        if not invocation.tools:
            return None
        return tuple(
            LlmTool(
                type=str(tool_spec.get("type", "")),
                function=_optional_mapping(tool_spec.get("function"), label="function"),
                config=_optional_mapping(tool_spec.get("config"), label="config"),
            )
            for tool_spec in invocation.tools
        )

    def _build_llm_request(
        self,
        invocation: LlmToolInvocation,
        messages: tuple[LlmMessage, ...],
        tools: tuple[LlmTool, ...] | None,
        max_output_tokens: int | None,
    ) -> LlmRequest:
        return LlmRequest(
            provider=self._llm_provider_name,
            model=invocation.model,
            messages=messages,
            temperature=invocation.temperature,
            max_output_tokens=int(max_output_tokens) if max_output_tokens is not None else None,
            output_mode="text",
            tools=tools,
            tool_choice=invocation.tool_choice,
            include=invocation.include,
            reasoning_effort=invocation.reasoning_effort,
            extra=dict(invocation.model_extra) if invocation.model_extra else None,
        )

    @staticmethod
    def _payload_from_args_kwargs(
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        if kwargs:
            return dict(kwargs)
        if args:
            first = args[0]
            if isinstance(first, dict):
                for key in first:
                    if not isinstance(key, str):
                        raise TypeError("expected JSON object with string keys")
                return dict(first)
            raise TypeError("expected JSON object payload as first positional argument")
        return {}


def _optional_mapping(value: object | None, *, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"tool spec {label} must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"tool spec {label} must have string keys")
    return cast(Mapping[str, object], value)


__all__ = [
    "ALLOWED_TOOL_MODELS",
    "LlmToolInvocation",
    "LlmToolMessage",
    "RuntimeToolInvoker",
]
