"""Tool invocation dispatch shared by platform and validator."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, ValidationError, field_validator, model_validator
from pydantic import JsonValue as PydanticJsonValue

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.errors import ToolInvocationTimeoutError, ToolProviderError
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.llm.pricing import (
    MINER_TOOL_LLM_PRICING,
    SEARCH_PRICING_PER_REFERENCEABLE_RESULT,
    price_miner_llm,
    price_search,
)
from harnyx_commons.llm.provider import LlmProviderError, LlmProviderPort, LlmRetryExhaustedError
from harnyx_commons.llm.provider_types import CHUTES_PROVIDER, OPENROUTER_PROVIDER
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
from harnyx_commons.llm.tool_models import (
    ALLOWED_TOOL_MODELS,
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    ToolModelName,
    parse_miner_selected_llm_provider_model,
)
from harnyx_commons.platform_tool_proxy import (
    PLATFORM_TOOL_PROXY_LLM_CHAT_DEFAULT_TIMEOUT_SECONDS,
    PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS,
    platform_tool_proxy_provider_timeout_seconds,
)
from harnyx_commons.tools.dto import tool_payload_from_args_kwargs
from harnyx_commons.tools.executor import ToolInvocationContext, ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.provider_billing import (
    ProviderBillingMetadata,
    SearchProviderResult,
    billing_evidence_payload,
)
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchProviderName,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_commons.tools.types import TOOL_NAMES, SearchToolName, ToolInvocationTimeout, ToolName, is_search_tool
from harnyx_commons.tools.usage_tracker import ToolCallUsage  # noqa: F401 - compatibility

MINER_SANDBOX_TOOL_NAMES: tuple[ToolName, ...] = tuple(sorted(TOOL_NAMES))
DEFAULT_TOOL_LLM_TIMEOUT_SECONDS = PLATFORM_TOOL_PROXY_LLM_CHAT_DEFAULT_TIMEOUT_SECONDS
DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS = PLATFORM_TOOL_PROXY_SEARCH_TOOL_DEFAULT_TIMEOUT_SECONDS
TInvocationResult = TypeVar("TInvocationResult")
SearchProviderResolver = Callable[
    [SearchProviderName, ToolInvocationContext | None],
    WebSearchProviderPort | Awaitable[WebSearchProviderPort],
]
LlmProviderResolver = Callable[
    [str, ToolInvocationContext | None],
    LlmProviderPort | Awaitable[LlmProviderPort],
]


@dataclass(frozen=True, slots=True)
class _ActualCost:
    cost_usd: float | None
    provider: str | None
    evidence: JsonObject | None = None


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

    provider: Literal["chutes", "openrouter"]
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
    web_search_provider_resolver: SearchProviderResolver | None = None,
    llm_provider: LlmProviderPort | None = None,
    llm_provider_name: str | None = None,
    llm_provider_resolver: LlmProviderResolver | None = None,
    allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
) -> RuntimeToolInvoker:
    return RuntimeToolInvoker(
        receipt_log,
        web_search_client=web_search_client,
        web_search_provider_name=web_search_provider_name,
        web_search_provider_resolver=web_search_provider_resolver,
        llm_provider=llm_provider,
        llm_provider_name=llm_provider_name,
        llm_provider_resolver=llm_provider_resolver,
        advertised_tool_names=MINER_SANDBOX_TOOL_NAMES,
        allowed_models=allowed_models,
    )


def effective_tool_timeout_seconds(
    tool_name: ToolName,
    *,
    args: Sequence[JsonValue],
    kwargs: Mapping[str, JsonValue],
) -> float:
    payload = tool_payload_from_args_kwargs(args, kwargs)
    if tool_name == "llm_chat":
        return _effective_timeout_from_payload(payload, default=DEFAULT_TOOL_LLM_TIMEOUT_SECONDS)
    if tool_name in {"search_web", "search_ai", "fetch_page"}:
        return _effective_timeout_from_payload(payload, default=DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS)
    raise LookupError(f"tool {tool_name!r} does not have a provider timeout")


def _effective_timeout_from_payload(payload: JsonObject, *, default: float) -> float:
    raw_timeout = payload.get("timeout")
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, int | float):
        return default
    timeout = float(raw_timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        return default
    return timeout


def _provider_request_timeout_seconds(*, default: float, effective_timeout: float | None) -> float:
    if effective_timeout is None:
        return default
    return max(default, platform_tool_proxy_provider_timeout_seconds(effective_timeout))


async def _resolve_maybe_awaitable(value: TInvocationResult | Awaitable[TInvocationResult]) -> TInvocationResult:
    if inspect.isawaitable(value):
        return await value
    return value


class RuntimeToolInvoker(ToolInvoker):
    """Dispatches sandbox tool invocations."""

    def __init__(
        self,
        receipt_log: ReceiptLogPort,
        *,
        web_search_client: WebSearchProviderPort | None = None,
        web_search_provider_name: str | None = None,
        web_search_provider_resolver: SearchProviderResolver | None = None,
        llm_provider: LlmProviderPort | None = None,
        llm_provider_name: str | None = None,
        llm_provider_resolver: LlmProviderResolver | None = None,
        advertised_tool_names: tuple[ToolName, ...] | None = None,
        allowed_models: tuple[ToolModelName, ...] = ALLOWED_TOOL_MODELS,
    ) -> None:
        self._receipts = receipt_log
        self._logger = logging.getLogger("harnyx_commons.tools.runtime_invoker")
        self._web_search = web_search_client
        self._web_search_provider_name = web_search_provider_name
        self._web_search_provider_resolver = web_search_provider_resolver
        self._llm_provider = llm_provider
        self._llm_provider_name = llm_provider_name
        self._llm_provider_resolver = llm_provider_resolver
        self._advertised_tool_names = tuple(sorted(advertised_tool_names or TOOL_NAMES))
        _ = allowed_models

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        context: ToolInvocationContext | None = None,
    ) -> JsonObject | ToolInvocationOutput:
        if tool_name == "test_tool":
            return self._invoke_test_tool(args, kwargs)
        if tool_name == "tooling_info":
            return self._invoke_tooling_info(args, kwargs)
        if is_search_tool(tool_name):
            return await self._dispatch_search(tool_name, args, kwargs, context=context)
        if tool_name == "llm_chat":
            return await self._dispatch_llm(args, kwargs, context=context)
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

        # Search tools keep a generic static price table here. Provider-returned
        # settlement can differ, for example Parallel search has a base price.
        for tool_name, usd_per_referenceable_result in SEARCH_PRICING_PER_REFERENCEABLE_RESULT.items():
            if tool_name not in visible_tool_names:
                continue
            pricing[tool_name] = {
                "kind": "per_referenceable_result",
                "settlement_order": ["provider_returned", "static_pricing"],
                "usd_per_referenceable_result": usd_per_referenceable_result,
            }

        if "llm_chat" in visible_tool_names:
            pricing["llm_chat"] = {
                "kind": "per_million_tokens",
                "settlement_order": [
                    "provider_returned",
                    "cached_provider_pricing",
                    "static_pricing",
                ],
                "provider_models": {
                    provider: {
                        model: {
                            "input_per_million": rates.input_per_million,
                            "output_per_million": rates.output_per_million,
                            "reasoning_per_million": rates.billable_reasoning_per_million,
                        }
                        for model, rates in model_pricing.items()
                    }
                    for provider, model_pricing in MINER_TOOL_LLM_PRICING.items()
                },
            }

        tool_names: list[JsonValue] = [str(name) for name in self._advertised_tool_names]
        allowed_provider_models: dict[str, JsonValue] = {
            provider: [str(model) for model in models]
            for provider, models in MINER_SELECTED_LLM_PROVIDER_MODELS.items()
        }
        return {
            "tool_names": tool_names,
            "allowed_llm_provider_models": allowed_provider_models,
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
        *,
        context: ToolInvocationContext | None,
    ) -> ToolInvocationOutput:
        if self._web_search is None and self._web_search_provider_resolver is None:
            raise LookupError("search client is not configured")
        payload = tool_payload_from_args_kwargs(args, kwargs)
        if tool_name == "search_web":
            request_model_web = SearchWebSearchRequest.model_validate(payload)
            web_search, provider_name = await self._resolve_search_provider(request_model_web.provider, context)
            response_web = await _invoke_with_optional_timeout(
                "search_web",
                request_model_web.timeout,
                lambda: _invoke_search_provider(
                    web_search,
                    request_model_web,
                    tool_name=tool_name,
                ),
            )
            as_mapping = response_web.response.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                tool_name=tool_name,
                billing=response_web.billing,
                request_provider=provider_name,
            )
        elif tool_name == "search_ai":
            request_ai = SearchAiSearchRequest.model_validate(payload)
            web_search, provider_name = await self._resolve_search_provider(request_ai.provider, context)
            response = await _invoke_with_optional_timeout(
                "search_ai",
                request_ai.timeout,
                lambda: _invoke_search_provider(
                    web_search,
                    request_ai,
                    tool_name=tool_name,
                ),
            )
            as_mapping = response.response.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                tool_name=tool_name,
                billing=response.billing,
                request_provider=provider_name,
            )
        elif tool_name == "fetch_page":
            request_page = FetchPageRequest.model_validate(payload)
            web_search, provider_name = await self._resolve_search_provider(request_page.provider, context)
            response_page = await _invoke_with_optional_timeout(
                "fetch_page",
                request_page.timeout,
                lambda: _invoke_search_provider(
                    web_search,
                    request_page,
                    tool_name=tool_name,
                ),
            )
            as_mapping = response_page.response.model_dump(exclude_none=True, mode="json")
            return _search_invocation_output(
                cast(JsonObject, as_mapping),
                tool_name=tool_name,
                billing=response_page.billing,
                request_provider=provider_name,
            )
        raise LookupError(f"search tool '{tool_name}' is not supported")

    async def _dispatch_llm(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        *,
        context: ToolInvocationContext | None,
    ) -> ToolInvocationOutput:
        if self._llm_provider is None and self._llm_provider_resolver is None:
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

        llm_provider = await self._resolve_llm_provider(invocation.provider, context)

        try:
            started_at = time.perf_counter()
            llm_response = await _invoke_with_optional_timeout(
                "llm_chat",
                invocation.timeout,
                lambda: llm_provider.invoke(request),
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        except (ToolProviderError, ToolInvocationTimeoutError):
            raise
        except (LlmProviderError, LlmRetryExhaustedError) as exc:
            raise ToolProviderError("tool provider failed") from exc
        try:
            actual_cost = _settle_llm_cost(
                llm_response,
                provider=invocation.provider,
                model=invocation.model,
            )
            _require_actual_cost(actual_cost, tool_name="llm_chat")
        except ValueError as exc:
            raise ToolProviderError("tool provider failed") from exc
        return ToolInvocationOutput(
            public_payload=_public_llm_response_payload(llm_response),
            execution=ToolExecutionFacts(elapsed_ms=elapsed_ms, ttft_ms=_response_ttft_ms(llm_response)),
            actual_cost_usd=actual_cost.cost_usd,
            actual_cost_provider=actual_cost.provider,
            actual_cost_evidence=actual_cost.evidence,
        )

    def _parse_invocation(
        self,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
    ) -> LlmToolInvocation:
        payload = tool_payload_from_args_kwargs(args, kwargs)
        invocation = LlmToolInvocation.model_validate(payload)
        selection = parse_miner_selected_llm_provider_model(
            provider=invocation.provider,
            model=invocation.model,
        )
        return invocation.model_copy(
            update={
                "provider": selection.provider,
                "model": selection.model,
            }
        )

    def _require_search_provider(self, requested_provider: SearchProviderName) -> SearchProviderName:
        configured_provider = self._web_search_provider_name
        if configured_provider is None:
            raise LookupError("search provider name is not configured")
        if requested_provider != configured_provider:
            raise ValueError(
                f"requested search provider {requested_provider!r} does not match configured provider "
                f"{configured_provider!r}"
            )
        return requested_provider

    async def _resolve_search_provider(
        self,
        requested_provider: SearchProviderName,
        context: ToolInvocationContext | None,
    ) -> tuple[WebSearchProviderPort, SearchProviderName]:
        resolver = self._web_search_provider_resolver
        if resolver is not None:
            return await _resolve_maybe_awaitable(resolver(requested_provider, context)), requested_provider
        web_search = self._web_search
        if web_search is None:
            raise LookupError("search client is not configured")
        return web_search, self._require_search_provider(requested_provider)

    def _require_llm_provider(self, requested_provider: str) -> str:
        configured_provider = self._llm_provider_name
        if configured_provider is None:
            raise LookupError("llm provider name is not configured")
        if requested_provider != configured_provider:
            raise ValueError(
                f"requested llm provider {requested_provider!r} does not match configured provider "
                f"{configured_provider!r}"
            )
        return requested_provider

    async def _resolve_llm_provider(
        self,
        requested_provider: str,
        context: ToolInvocationContext | None,
    ) -> LlmProviderPort:
        resolver = self._llm_provider_resolver
        if resolver is not None:
            return await _resolve_maybe_awaitable(resolver(requested_provider, context))
        llm_provider = self._llm_provider
        if llm_provider is None:
            raise LookupError("llm provider is not configured")
        self._require_llm_provider(requested_provider)
        return llm_provider

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
            provider=invocation.provider,
            model=invocation.model,
            messages=messages,
            temperature=invocation.temperature,
            max_output_tokens=int(max_output_tokens) if max_output_tokens is not None else None,
            output_mode="text",
            tools=tools,
            tool_choice=invocation.tool_choice,
            include=invocation.include,
            timeout_seconds=_provider_request_timeout_seconds(
                default=DEFAULT_TOOL_LLM_TIMEOUT_SECONDS,
                effective_timeout=invocation.timeout,
            ),
            thinking=invocation.thinking.to_schema() if invocation.thinking is not None else None,
            use_case="tool_runtime_invoker",
        )


async def _invoke_search_provider(
    web_search: WebSearchProviderPort,
    request: SearchWebSearchRequest | SearchAiSearchRequest | FetchPageRequest,
    *,
    tool_name: SearchToolName,
) -> SearchProviderResult[SearchWebSearchResponse | SearchAiSearchResponse | FetchPageResponse]:
    if tool_name == "search_web":
        if not isinstance(request, SearchWebSearchRequest):
            raise TypeError("search_web requires SearchWebSearchRequest")
        return await web_search.search_web(request)
    if tool_name == "search_ai":
        if not isinstance(request, SearchAiSearchRequest):
            raise TypeError("search_ai requires SearchAiSearchRequest")
        return await web_search.search_ai(request)
    if tool_name == "fetch_page":
        if not isinstance(request, FetchPageRequest):
            raise TypeError("fetch_page requires FetchPageRequest")
        return await web_search.fetch_page(request)
    raise LookupError(f"search tool '{tool_name}' is not supported")


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
    except (TimeoutError, ValidationError) as exc:
        raise ToolProviderError("tool provider failed") from exc


def _task_result(task: asyncio.Task[TInvocationResult]) -> TInvocationResult:
    try:
        return task.result()
    except (TimeoutError, ValidationError) as exc:
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


def _response_ttft_ms(response: LlmResponse) -> float | None:
    raw_ttft_ms = (response.metadata or {}).get("ttft_ms")
    if isinstance(raw_ttft_ms, bool) or not isinstance(raw_ttft_ms, (int, float)):
        return None
    ttft_ms = float(raw_ttft_ms)
    if ttft_ms <= 0:
        return None
    return ttft_ms


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
    tool_name: SearchToolName,
    billing: ProviderBillingMetadata | None,
    request_provider: SearchProviderName,
) -> ToolInvocationOutput:
    try:
        actual_cost = _settle_search_cost(
            tool_name=tool_name,
            public_payload=public_payload,
            billing=billing,
            request_provider=request_provider,
        )
        _require_actual_cost(actual_cost, tool_name=tool_name)
    except ValueError as exc:
        raise ToolProviderError("tool provider failed") from exc
    return ToolInvocationOutput(
        public_payload=public_payload,
        actual_cost_usd=actual_cost.cost_usd,
        actual_cost_provider=actual_cost.provider,
        actual_cost_evidence=actual_cost.evidence,
    )


def _settle_search_cost(
    *,
    tool_name: SearchToolName,
    public_payload: JsonObject,
    billing: ProviderBillingMetadata | None,
    request_provider: SearchProviderName,
) -> _ActualCost:
    if billing is not None and billing.actual_cost_usd is not None:
        if billing.actual_cost_provider not in {"desearch", "parallel"}:
            raise ValueError(f"{tool_name} provider-backed success missing supported provider cost evidence")
        return _ActualCost(
            billing.actual_cost_usd,
            billing.actual_cost_provider,
            {
                "settlement_source": "provider_returned",
                "provider_billing": cast(JsonObject, billing_evidence_payload(billing) or {}),
            },
        )

    provider = billing.actual_cost_provider if billing is not None else str(request_provider)
    if provider not in {"desearch", "parallel"}:
        raise ValueError(f"{tool_name} provider-backed success missing supported provider cost evidence")
    referenceable_results = _referenceable_result_count(tool_name, public_payload)
    if referenceable_results is None:
        raise ValueError(f"{tool_name} provider-backed success missing settled cost")
    return _ActualCost(
        price_search(tool_name, referenceable_results=referenceable_results),
        provider,
        {
            "settlement_source": "static_pricing",
            "provider": provider,
            "referenceable_results": referenceable_results,
        },
    )


def _referenceable_result_count(tool_name: SearchToolName, public_payload: JsonObject) -> int | None:
    data = public_payload.get("data")
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        return None
    url_key = "link" if tool_name == "search_web" else "url"
    count = 0
    for item in data:
        if not isinstance(item, Mapping):
            continue
        result_item = cast(Mapping[str, object], item)
        url = result_item.get(url_key)
        if isinstance(url, str) and url.strip():
            count += 1
    return count


def _require_actual_cost(actual_cost: _ActualCost, *, tool_name: str) -> None:
    if actual_cost.cost_usd is None:
        raise ValueError(f"{tool_name} provider-backed success missing actual_cost_usd")
    if isinstance(actual_cost.cost_usd, bool) or not isinstance(actual_cost.cost_usd, int | float):
        raise ValueError(f"{tool_name} provider-backed success actual_cost_usd must be numeric")
    if not math.isfinite(actual_cost.cost_usd):
        raise ValueError(f"{tool_name} provider-backed success actual_cost_usd must be finite")
    if actual_cost.cost_usd < 0.0:
        raise ValueError(f"{tool_name} provider-backed success actual_cost_usd must be non-negative")
    if actual_cost.provider is None:
        raise ValueError(f"{tool_name} provider-backed success missing actual_cost_provider")


def _settle_llm_cost(response: LlmResponse, *, provider: str, model: str) -> _ActualCost:
    provider_cost = _provider_returned_llm_cost(response, provider=provider)
    if provider_cost.cost_usd is not None:
        return provider_cost
    return _ActualCost(
        price_miner_llm(provider, model, response.usage),
        provider,
        {
            "settlement_source": "static_pricing",
            "provider": provider,
            "model": model,
        },
    )


def _provider_returned_llm_cost(response: LlmResponse, *, provider: str) -> _ActualCost:
    metadata = response.metadata or {}
    if metadata.get("actual_cost_provider") == CHUTES_PROVIDER:
        cost = metadata.get("actual_cost_usd")
        if not isinstance(cost, int | float) or isinstance(cost, bool):
            raise ValueError("Chutes actual_cost_usd must be numeric when supplied")
        if cost < 0:
            raise ValueError("Chutes actual_cost_usd must be non-negative")
        evidence = metadata.get("actual_cost_evidence")
        return _ActualCost(
            float(cost),
            CHUTES_PROVIDER,
            cast(JsonObject, evidence) if isinstance(evidence, Mapping) else None,
        )
    return _ActualCost(
        _openrouter_actual_cost_usd(response, provider=provider),
        _openrouter_actual_cost_provider(response, provider=provider),
        {"settlement_source": "provider_returned", "pricing_origin": "openrouter_usage_cost"},
    )


def _openrouter_actual_cost_usd(response: LlmResponse, *, provider: str) -> float | None:
    if _openrouter_actual_cost_provider(response, provider=provider) != OPENROUTER_PROVIDER:
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
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        raise ValueError("OpenRouter usage.cost must be numeric when supplied")
    if cost < 0.0:
        raise ValueError("OpenRouter usage.cost must be non-negative")
    return float(cost)


def _openrouter_actual_cost_provider(response: LlmResponse, *, provider: str) -> str | None:
    metadata = response.metadata or {}
    if provider == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    if metadata.get("effective_provider") == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    if metadata.get("selected_provider") == OPENROUTER_PROVIDER:
        return OPENROUTER_PROVIDER
    return None


__all__ = [
    "ALLOWED_TOOL_MODELS",
    "DEFAULT_SEARCH_TOOL_TIMEOUT_SECONDS",
    "DEFAULT_TOOL_LLM_TIMEOUT_SECONDS",
    "LlmToolInvocation",
    "LlmToolMessage",
    "LlmThinkingConfigPayload",
    "RuntimeToolInvoker",
    "MINER_SANDBOX_TOOL_NAMES",
    "build_miner_sandbox_tool_invoker",
    "effective_tool_timeout_seconds",
]
