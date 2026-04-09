"""Tool invocation dispatch shared by platform and validator."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict
from pydantic import JsonValue as PydanticJsonValue

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.errors import ToolProviderError
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.pricing import (
    ALLOWED_TOOL_MODELS,
    MODEL_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    ToolModelName,
    parse_tool_model,
)
from harnyx_commons.llm.provider import LlmProviderPort, LlmRetryExhaustedError
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmTool,
)
from harnyx_commons.tools.executor import ToolInvoker
from harnyx_commons.tools.normalize import normalize_response
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    SearchAiSearchRequest,
    SearchWebSearchRequest,
)
from harnyx_commons.tools.types import TOOL_NAMES, SearchToolName, ToolName, is_search_tool
from harnyx_commons.tools.usage_tracker import ToolCallUsage  # noqa: F401 - compatibility

MINER_SANDBOX_TOOL_NAMES: tuple[ToolName, ...] = tuple(sorted(TOOL_NAMES))


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


def build_miner_sandbox_tool_invoker(
    receipt_log: ReceiptLogPort,
    *,
    web_search_client: WebSearchProviderPort | None = None,
    llm_provider: LlmProviderPort | None = None,
    llm_provider_name: str | None = None,
    allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
) -> RuntimeToolInvoker:
    return RuntimeToolInvoker(
        receipt_log,
        web_search_client=web_search_client,
        llm_provider=llm_provider,
        llm_provider_name=llm_provider_name,
        advertised_tool_names=MINER_SANDBOX_TOOL_NAMES,
        allowed_models=allowed_models,
    )


class RuntimeToolInvoker(ToolInvoker):
    """Dispatches sandbox tool invocations."""

    def __init__(
        self,
        receipt_log: ReceiptLogPort,
        *,
        web_search_client: WebSearchProviderPort | None = None,
        llm_provider: LlmProviderPort | None = None,
        llm_provider_name: str | None = None,
        advertised_tool_names: tuple[ToolName, ...] | None = None,
        allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
    ) -> None:
        self._receipts = receipt_log
        self._logger = logging.getLogger("harnyx_commons.tools.runtime_invoker")
        self._web_search = web_search_client
        self._llm_provider = llm_provider
        self._llm_provider_name = llm_provider_name or "llm"
        self._advertised_tool_names = tuple(sorted(advertised_tool_names or TOOL_NAMES))
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
        if tool_name == "tooling_info":
            return self._invoke_tooling_info(args, kwargs)
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

    def _invoke_tooling_info(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if args:
            raise ValueError("tooling_info does not accept positional arguments")
        if kwargs:
            raise ValueError("tooling_info does not accept keyword arguments")

        visible_tool_names = set(self._advertised_tool_names)
        pricing: dict[str, JsonValue] = {}

        if "test_tool" in visible_tool_names:
            pricing["test_tool"] = {"kind": "free"}
        if "tooling_info" in visible_tool_names:
            pricing["tooling_info"] = {"kind": "free"}

        for tool_name, usd_per_referenceable_result in SEARCH_PRICING_PER_REFERENCEABLE_RESULT.items():
            if tool_name not in visible_tool_names:
                continue
            pricing[tool_name] = {
                "kind": "per_referenceable_result",
                "usd_per_referenceable_result": usd_per_referenceable_result,
            }

        if "llm_chat" in visible_tool_names:
            pricing["llm_chat"] = {
                "kind": "per_million_tokens",
                "models": {
                    model: {
                        "input_per_million": rates.input_per_million,
                        "output_per_million": rates.output_per_million,
                        "reasoning_per_million": rates.reasoning_per_million,
                    }
                    for model, rates in MODEL_PRICING.items()
                },
            }

        tool_names: list[JsonValue] = [str(name) for name in self._advertised_tool_names]
        allowed_models: list[JsonValue] = [str(model) for model in ALLOWED_TOOL_MODELS]
        return {
            "tool_names": tool_names,
            "allowed_tool_models": allowed_models,
            "pricing": pricing,
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
                "tool_args": tuple(args),
                "tool_kwargs": dict(kwargs),
            },
        )

    @normalize_response
    async def _dispatch_search(
        self,
        tool_name: SearchToolName,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if self._web_search is None:
            raise LookupError("search client is not configured")
        payload = self._payload_from_args_kwargs(args, kwargs)
        if tool_name == "search_web":
            request_model_web = SearchWebSearchRequest.model_validate(payload)
            response_web = await self._web_search.search_web(request_model_web)
            as_mapping = response_web.model_dump(exclude_none=True, mode="json")
            return cast(JsonObject, as_mapping)
        elif tool_name == "search_ai":
            request_ai = SearchAiSearchRequest.model_validate(payload)
            response = await self._web_search.search_ai(request_ai)
            as_mapping = response.model_dump(exclude_none=True, mode="json")
            return cast(JsonObject, as_mapping)
        elif tool_name == "fetch_page":
            request_page = FetchPageRequest.model_validate(payload)
            response_page = await self._web_search.fetch_page(request_page)
            as_mapping = response_page.model_dump(exclude_none=True, mode="json")
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

        try:
            llm_response = await self._llm_provider.invoke(request)
        except LlmRetryExhaustedError as exc:
            raise ToolProviderError("tool provider failed") from exc
        return _public_llm_response_payload(llm_response)

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
            internal_metadata={"use_case": "tool_runtime_invoker"},
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


def _public_llm_response_payload(response: LlmResponse) -> JsonObject:
    payload: JsonObject = {
        "id": response.id,
        "choices": [_public_llm_choice_payload(choice) for choice in response.choices],
        "usage": cast(JsonObject, asdict(response.usage)),
    }
    if response.finish_reason is not None:
        payload["finish_reason"] = response.finish_reason
    return payload


def _public_llm_choice_payload(choice: LlmChoice) -> JsonObject:
    payload: JsonObject = {
        "index": choice.index,
        "message": _public_llm_message_payload(choice.message),
    }
    if choice.finish_reason is not None:
        payload["finish_reason"] = choice.finish_reason
    return payload


def _public_llm_message_payload(message: LlmChoiceMessage) -> JsonObject:
    payload: JsonObject = {
        "role": message.role,
        "content": [_public_llm_content_part_payload(part) for part in message.content],
    }
    if message.tool_calls:
        payload["tool_calls"] = [_public_llm_tool_call_payload(call) for call in message.tool_calls]
    return payload


def _public_llm_content_part_payload(part: LlmMessageContentPart) -> JsonObject:
    payload: JsonObject = {"type": part.type}
    if part.text is not None:
        payload["text"] = part.text
    if part.data is not None:
        payload["data"] = cast(JsonObject, dict(part.data))
    return payload


def _public_llm_tool_call_payload(call: LlmMessageToolCall) -> JsonObject:
    return {
        "id": call.id,
        "type": call.type,
        "name": call.name,
        "arguments": call.arguments,
    }

__all__ = [
    "ALLOWED_TOOL_MODELS",
    "LlmToolInvocation",
    "LlmToolMessage",
    "RuntimeToolInvoker",
    "MINER_SANDBOX_TOOL_NAMES",
    "build_miner_sandbox_tool_invoker",
]
