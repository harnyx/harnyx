"""Tool invocation dispatch shared by platform and validator."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, field_validator, model_validator
from pydantic import JsonValue as PydanticJsonValue

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.errors import ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.pricing import (
    MODEL_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    price_parallel_extract,
    price_parallel_search,
)
from harnyx_commons.llm.provider import LlmProviderPort, LlmRetryExhaustedError
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmThinkingConfig,
    LlmTool,
)
from harnyx_commons.llm.tool_models import ALLOWED_TOOL_MODELS, ToolModelName, parse_tool_model
from harnyx_commons.tools.dto import tool_payload_from_args_kwargs
from harnyx_commons.tools.executor import ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    SearchAiSearchRequest,
    SearchWebSearchRequest,
)
from harnyx_commons.tools.types import TOOL_NAMES, SearchToolName, ToolInvocationTimeout, ToolName, is_search_tool
from harnyx_commons.tools.usage_tracker import ToolCallUsage  # noqa: F401 - compatibility

MINER_SANDBOX_TOOL_NAMES: tuple[ToolName, ...] = tuple(sorted(TOOL_NAMES))
DEFAULT_TOOL_LLM_TIMEOUT_SECONDS = 120.0
TInvocationResult = TypeVar("TInvocationResult")


class _ToolingInfoInvocation(BaseModel):
    """Request payload for tooling_info tool calls."""

    model_config = ConfigDict(extra="forbid")

    timeout: ToolInvocationTimeout | None = None


class _TestToolInvocation(BaseModel):
    """Request payload for test_tool calls."""

    model_config = ConfigDict(extra="forbid")

    message: str = ""
    timeout: ToolInvocationTimeout | None = None


class LlmToolMessage(BaseModel):
    """Message format for LLM tool invocations."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LlmThinkingConfigPayload(BaseModel):
    """Typed public thinking config for miner llm_chat tool calls."""

    model_config = ConfigDict(extra="forbid")

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
    def _validate_single_tuning_knob(self) -> LlmThinkingConfigPayload:
        if self.budget is not None and self.effort is not None:
            raise ValueError("thinking.budget and thinking.effort are mutually exclusive")
        return self

    def to_schema(self) -> LlmThinkingConfig:
        return LlmThinkingConfig(
            enabled=self.enabled,
            budget=self.budget,
            effort=self.effort,
        )


class LlmToolInvocation(BaseModel):
    """Request payload for llm_chat tool calls."""

    model: str
    messages: tuple[LlmToolMessage, ...]
    timeout: ToolInvocationTimeout | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    response_format: str = "text"
    tools: tuple[dict[str, PydanticJsonValue], ...] | None = None
    tool_choice: Literal["auto", "required"] | None = None
    include: tuple[str, ...] | None = None
    thinking: LlmThinkingConfigPayload | None = None

    model_config = ConfigDict(extra="forbid")


def build_miner_sandbox_tool_invoker(
    receipt_log: ReceiptLogPort,
    *,
    web_search_client: WebSearchProviderPort | None = None,
    web_search_provider_name: str | None = None,
    llm_provider: LlmProviderPort | None = None,
    llm_provider_name: str | None = None,
    allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
) -> RuntimeToolInvoker:
    return RuntimeToolInvoker(
        receipt_log,
        web_search_client=web_search_client,
        web_search_provider_name=web_search_provider_name,
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
        web_search_provider_name: str | None = None,
        llm_provider: LlmProviderPort | None = None,
        llm_provider_name: str | None = None,
        advertised_tool_names: tuple[ToolName, ...] | None = None,
        allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
    ) -> None:
        self._receipts = receipt_log
        self._logger = logging.getLogger("harnyx_commons.tools.runtime_invoker")
        self._web_search = web_search_client
        self._web_search_provider_name = web_search_provider_name
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
    ) -> JsonObject | ToolInvocationOutput:
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
        payload = dict(kwargs)
        if "message" in kwargs:
            message = str(payload.pop("message"))

        invocation = _TestToolInvocation.model_validate({"message": message, **payload})

        self._logger.info("test_tool message: %s", invocation.message)
        return {
            "status": "ok",
            "echo": invocation.message,
        }

    def _invoke_tooling_info(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> JsonObject:
        if args:
            raise ValueError("tooling_info does not accept positional arguments")
        _ToolingInfoInvocation.model_validate(dict(kwargs))

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
                        "reasoning_per_million": rates.billable_reasoning_per_million,
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

    async def _dispatch_search(
        self,
        tool_name: SearchToolName,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> ToolInvocationOutput:
        web_search = self._web_search
        if web_search is None:
            raise LookupError("search client is not configured")
        payload = tool_payload_from_args_kwargs(args, kwargs)
        if tool_name == "search_web":
            request_model_web = SearchWebSearchRequest.model_validate(payload)
            response_web = await _invoke_with_optional_timeout(
                "search_web",
                request_model_web.timeout,
                lambda: web_search.search_web(request_model_web),
            )
            as_mapping = response_web.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                provider_name=self._web_search_provider_name,
                tool_name=tool_name,
                request=request_model_web,
            )
        elif tool_name == "search_ai":
            request_ai = SearchAiSearchRequest.model_validate(payload)
            response = await _invoke_with_optional_timeout(
                "search_ai",
                request_ai.timeout,
                lambda: web_search.search_ai(request_ai),
            )
            as_mapping = response.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                provider_name=self._web_search_provider_name,
                tool_name=tool_name,
                request=request_ai,
            )
        elif tool_name == "fetch_page":
            request_page = FetchPageRequest.model_validate(payload)
            response_page = await _invoke_with_optional_timeout(
                "fetch_page",
                request_page.timeout,
                lambda: web_search.fetch_page(request_page),
            )
            as_mapping = response_page.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                provider_name=self._web_search_provider_name,
                tool_name=tool_name,
                request=request_page,
            )
        raise LookupError(f"search tool '{tool_name}' is not supported")

    async def _dispatch_llm(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> ToolInvocationOutput:
        llm_provider = self._llm_provider
        if llm_provider is None:
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
            started_at = time.perf_counter()
            llm_response = await _invoke_with_optional_timeout(
                "llm_chat",
                invocation.timeout,
                lambda: llm_provider.invoke(request),
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        except LlmRetryExhaustedError as exc:
            raise ToolProviderError("tool provider failed") from exc
        return ToolInvocationOutput(
            public_payload=_public_llm_response_payload(llm_response),
            execution=ToolExecutionFacts(elapsed_ms=elapsed_ms),
            actual_cost_usd=_openrouter_actual_cost_usd(llm_response),
            actual_cost_provider=_openrouter_actual_cost_provider(llm_response),
        )

    def _parse_invocation(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> LlmToolInvocation:
        payload = tool_payload_from_args_kwargs(args, kwargs)
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
            timeout_seconds=DEFAULT_TOOL_LLM_TIMEOUT_SECONDS,
            thinking=invocation.thinking.to_schema() if invocation.thinking is not None else None,
            use_case="tool_runtime_invoker",
        )


async def _invoke_with_optional_timeout(
    tool_name: str,
    timeout: float | None,
    operation: Callable[[], Awaitable[TInvocationResult]],
) -> TInvocationResult:
    if timeout is None:
        return await _invoke_provider_operation(operation)

    task = asyncio.ensure_future(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        await _cancel_provider_task(task)
        raise
    if task in done:
        return _task_result(task)

    await _cancel_provider_task(task)
    raise ToolInvocationTimeoutError(f"{tool_name} timed out after {timeout:g} seconds")


async def _invoke_provider_operation(
    operation: Callable[[], Awaitable[TInvocationResult]],
) -> TInvocationResult:
    try:
        return await operation()
    except TimeoutError as exc:
        raise ToolProviderError("tool provider failed") from exc


def _task_result(task: asyncio.Task[TInvocationResult]) -> TInvocationResult:
    try:
        return task.result()
    except TimeoutError as exc:
        raise ToolProviderError("tool provider failed") from exc


async def _cancel_provider_task(task: asyncio.Task[TInvocationResult]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


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
    if message.reasoning is not None:
        payload["reasoning"] = message.reasoning
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


def _search_invocation_output(
    public_payload: JsonObject,
    *,
    provider_name: str | None,
    tool_name: SearchToolName,
    request: SearchWebSearchRequest | SearchAiSearchRequest | FetchPageRequest,
) -> ToolInvocationOutput:
    actual_cost_usd = _parallel_actual_search_cost_usd(
        provider_name=provider_name,
        tool_name=tool_name,
        request=request,
    )
    return ToolInvocationOutput(
        public_payload=public_payload,
        actual_cost_usd=actual_cost_usd,
        actual_cost_provider="parallel" if actual_cost_usd is not None else None,
    )


def _parallel_actual_search_cost_usd(
    *,
    provider_name: str | None,
    tool_name: SearchToolName,
    request: SearchWebSearchRequest | SearchAiSearchRequest | FetchPageRequest,
) -> float | None:
    if provider_name != "parallel":
        return None
    if tool_name == "search_web":
        if not isinstance(request, SearchWebSearchRequest):
            raise TypeError("search_web actual pricing requires SearchWebSearchRequest")
        return price_parallel_search(requested_results=request.num)
    if tool_name == "search_ai":
        if not isinstance(request, SearchAiSearchRequest):
            raise TypeError("search_ai actual pricing requires SearchAiSearchRequest")
        return price_parallel_search(requested_results=request.count)
    if tool_name == "fetch_page":
        if not isinstance(request, FetchPageRequest):
            raise TypeError("fetch_page actual pricing requires FetchPageRequest")
        return price_parallel_extract(url_count=1)
    raise LookupError(f"search tool '{tool_name}' is not supported")


def _openrouter_actual_cost_usd(response: LlmResponse) -> float | None:
    if _openrouter_actual_cost_provider(response) != OPENROUTER_PROVIDER:
        return None
    raw_response = (response.metadata or {}).get("raw_response")
    if not isinstance(raw_response, Mapping):
        return None
    raw_response_mapping = cast(Mapping[str, object], raw_response)
    usage = raw_response_mapping.get("usage")
    if not isinstance(usage, Mapping):
        return None
    usage_mapping = cast(Mapping[str, object], usage)
    cost = usage_mapping.get("cost")
    if cost is None:
        return None
    if not isinstance(cost, (int, float)):
        raise ValueError("OpenRouter usage.cost must be numeric when supplied")
    if cost < 0.0:
        raise ValueError("OpenRouter usage.cost must be non-negative")
    return float(cost)


def _openrouter_actual_cost_provider(response: LlmResponse) -> str | None:
    metadata = response.metadata or {}
    if metadata.get("effective_provider") == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    if metadata.get("selected_provider") == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    return None

__all__ = [
    "ALLOWED_TOOL_MODELS",
    "DEFAULT_TOOL_LLM_TIMEOUT_SECONDS",
    "LlmToolInvocation",
    "LlmToolMessage",
    "LlmThinkingConfigPayload",
    "RuntimeToolInvoker",
    "MINER_SANDBOX_TOOL_NAMES",
    "build_miner_sandbox_tool_invoker",
]
