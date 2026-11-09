"""OpenRouter Responses gateway for internal platform workloads."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Annotated, Any, Literal, cast

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolParam,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from harnyx_commons.llm.cost_settlement import settled_openrouter_reported_cost
from harnyx_commons.llm.provider import BaseLlmProvider, LlmProviderConfigurationError, record_observed_llm_response
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmUsage,
    SearchQueryArguments,
)


class _OpenRouterModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class _SearchSource(_OpenRouterModel):
    type: Literal["url"]
    url: str


class _SearchAction(_OpenRouterModel):
    type: Literal["search"]
    query: str
    sources: list[_SearchSource] = Field(default_factory=list)


class OpenRouterWebSearchOutput(_OpenRouterModel):
    type: Literal["openrouter:web_search"]
    status: Literal["in_progress", "searching", "completed", "failed"]
    id: str | None = None
    action: _SearchAction | None = None
    error: str | None = None


class _InputTokenDetails(_OpenRouterModel):
    cached_tokens: int = 0


class _OutputTokenDetails(_OpenRouterModel):
    reasoning_tokens: int = 0


class _ServerToolUsage(_OpenRouterModel):
    web_search_requests: int | None = Field(default=None, ge=0, strict=True)


class _OpenRouterUsage(_OpenRouterModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: _InputTokenDetails | None = None
    output_tokens_details: _OutputTokenDetails | None = None
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    server_tool_use_details: _ServerToolUsage | None = None


class OpenRouterResponse(_OpenRouterModel):
    """OpenRouter's Responses wire contract, including its server-tool variant."""

    id: str
    status: str
    output: list[
        Annotated[
            ResponseOutputMessage
            | ResponseFunctionToolCall
            | ResponseReasoningItem
            | ResponseFunctionWebSearch
            | OpenRouterWebSearchOutput,
            Field(discriminator="type"),
        ]
    ]
    usage: _OpenRouterUsage | None = None


async def create_openrouter_response(client: AsyncOpenAI, **kwargs: Any) -> OpenRouterResponse:
    # The OpenAI SDK is the transport; its output union excludes OpenRouter server tools.
    response = await client.responses.with_raw_response.create(**kwargs)
    return OpenRouterResponse.model_validate(response.http_response.json())


class OpenRouterResponsesLlmProvider(BaseLlmProvider):
    """Keep SDK objects, native search and retrieval callbacks at the boundary."""

    def __init__(
        self,
        *,
        api_key: SecretStr | None = None,
        client: AsyncOpenAI | None = None,
        retrieval_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(provider_label="openrouter", max_concurrent=100)
        self._key = api_key if api_key is not None else SecretStr(os.environ.get("OPENROUTER_API_KEY", ""))
        self._client = client
        self._retrieval = retrieval_client or httpx.AsyncClient(timeout=300)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
        await self._retrieval.aclose()

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        return await self._call_with_retry(
            request,
            call_coro=self._request,
            verifier=self._verify,
            classify_exception=self._classify,
            policy=request.retry_policy,
        )

    @staticmethod
    def _classify(exc: Exception) -> tuple[bool, str]:
        if isinstance(exc, APIStatusError):
            return exc.status_code in {408, 409, 429, 500, 502, 503, 504}, f"http_{exc.status_code}"
        return isinstance(exc, (APIConnectionError, httpx.TransportError)), type(exc).__name__

    @staticmethod
    def _verify(response: LlmResponse) -> tuple[bool, bool, str | None]:
        if response.finish_reason not in {"stop", "tool_calls"}:
            return False, False, response.finish_reason or "incomplete_response"
        return bool(response.raw_text or response.tool_calls), False, "empty_response"

    async def _request(self, request: AbstractLlmRequest) -> LlmResponse:
        if self._client is None:
            key = self._key.get_secret_value().strip()
            if not key:
                raise LlmProviderConfigurationError("OPENROUTER_API_KEY is required for OpenRouter calls")
            self._client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0)
        history = _input(request)
        tools: list[dict[str, Any]] = [{"type": "openrouter:web_search"}] if request.grounded else []
        retrieval: dict[str, Any] | None = None
        for tool in request.tools or ():
            if tool.type == "provider_native":
                if retrieval is not None:
                    raise ValueError("Only one native contextual retrieval endpoint is supported")
                retrieval = dict(tool.config or {})["retrieval"]["external_api"]
                if retrieval["api_spec"] != "SIMPLE_SEARCH":
                    raise ValueError("Native retrieval requires SIMPLE_SEARCH")
                tools.append(
                    {
                        "type": "function",
                        "name": "search_context",
                        "strict": True,
                        "description": "Search the supplied feed or repository context for relevant evidence.",
                        "parameters": SearchQueryArguments.model_json_schema(),
                    }
                )
            elif tool.type == "function" and tool.function is not None:
                tools.append({"type": "function", **dict(tool.function)})
            else:
                raise ValueError(f"Unsupported internal OpenRouter tool: {tool.type}")
        kwargs: dict[str, Any] = {
            "model": request.model,
            "store": False,
            "service_tier": "default",
            "reasoning": {"effort": request.reasoning_effort or "none", "summary": "auto"},
            "include": ["reasoning.encrypted_content", "web_search_call.action.sources"],
        }
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls
        if request.output_mode == "structured" and request.output_schema is not None:
            from openai.lib._pydantic import to_strict_json_schema

            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema.__name__,
                    "strict": True,
                    "schema": to_strict_json_schema(request.output_schema),
                }
            }
        elif request.output_mode == "json_object":
            kwargs["text"] = {"format": {"type": "json_object"}}
        total_usage = LlmUsage()
        total_cost = 0.0
        cost_receipts: list[dict[str, Any]] = []
        complete_cost = True
        native_response_count = 0
        while True:
            if request.on_provider_dispatch is not None:
                request.on_provider_dispatch()
            raw = await create_openrouter_response(
                self._client, input=history, tools=cast(list[ToolParam], tools), **kwargs
            )
            result = normalize_response(raw)
            native_response_count += 1
            total_usage += result.usage
            cost = settled_openrouter_reported_cost(
                raw_response=raw.model_dump(mode="json", exclude_none=True), model=request.model
            )
            if cost is None:
                complete_cost = False
            else:
                total_cost += cost.cost_usd
                cost_receipts.append(dict(cost.evidence))
            metadata = dict(result.metadata or {})
            metadata["native_response_count"] = native_response_count
            if not complete_cost:
                metadata["cost_incomplete_reason"] = "missing_openrouter_billing_receipt"
            if complete_cost:
                metadata["actual_cost_usd"] = total_cost
                metadata["actual_cost_provider"] = "openrouter"
                metadata["actual_cost_evidence"] = {
                    "settlement_source": "provider_returned",
                    "includes_search": True,
                    "response_count": native_response_count,
                    "receipts": cost_receipts.copy(),
                }
            result = replace(result, usage=total_usage, metadata=metadata)
            record_observed_llm_response(result)
            if result.finish_reason not in {"stop", "tool_calls"} or retrieval is None or not result.tool_calls:
                return result
            history.extend(item.model_dump(mode="json", exclude_none=True) for item in raw.output)
            for call in raw.output:
                if not isinstance(call, ResponseFunctionToolCall):
                    continue
                if call.name != "search_context":
                    raise ValueError("Unexpected function call during contextual review")
                try:
                    arguments = SearchQueryArguments.model_validate_json(call.arguments)
                except ValidationError:
                    raise ValueError("Context search requires exactly one string query") from None
                auth = retrieval["auth_config"]["api_key_config"]
                response = await self._retrieval.post(
                    retrieval["endpoint"],
                    json=arguments.model_dump(),
                    headers={auth["name"]: auth["api_key_string"]},
                )
                response.raise_for_status()
                hits = response.json()
                if not isinstance(hits, list):
                    raise ValueError("SIMPLE_SEARCH response must be a list")
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(hits, ensure_ascii=False),
                    }
                )


def _input(request: AbstractLlmRequest) -> list[Any]:
    items: list[Any] = []
    for message in request.messages:
        parts: list[dict[str, Any]] = []
        for part in message.content:
            if isinstance(part, LlmInputToolResultPart):
                items.append({"type": "function_call_output", "call_id": part.tool_call_id, "output": part.output_json})
            elif isinstance(part, LlmInputTextPart):
                parts.append({"type": "input_text", "text": part.text})
            elif isinstance(part, LlmInputImagePart):
                parts.append({"type": "input_image", "image_url": part.data.url})
        if parts:
            items.append({"role": message.role, "content": parts})
        for call in message.tool_calls or ():
            items.append({"type": "function_call", "call_id": call.id, "name": call.name, "arguments": call.arguments})
    return items


def normalize_response(raw: OpenRouterResponse) -> LlmResponse:
    """Translate only public output and usage; opaque reasoning stays in native history."""
    text: list[str] = []
    summaries: list[str] = []
    calls: list[LlmMessageToolCall] = []
    sources: list[dict[str, str]] = []
    refused = False
    for item in raw.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputRefusal):
                    refused = True
                elif isinstance(part, ResponseOutputText):
                    text.append(part.text)
                    sources.extend(
                        {"url": a.url, "title": a.title}
                        for a in part.annotations
                        if isinstance(a, AnnotationURLCitation)
                    )
        elif isinstance(item, ResponseReasoningItem):
            summaries.extend(part.text for part in item.summary)
        elif isinstance(item, (ResponseFunctionWebSearch, OpenRouterWebSearchOutput)):
            for source in getattr(item.action, "sources", None) or ():
                sources.append({"url": source.url, "title": getattr(source, "title", None) or source.url})
        elif isinstance(item, ResponseFunctionToolCall):
            calls.append(LlmMessageToolCall(id=item.call_id, type="function", name=item.name, arguments=item.arguments))
    usage = raw.usage
    normalized_usage = LlmUsage(
        prompt_tokens=usage.input_tokens if usage else None,
        completion_tokens=usage.output_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        prompt_cached_tokens=usage.input_tokens_details.cached_tokens if usage and usage.input_tokens_details else None,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        if usage and usage.output_tokens_details
        else None,
        prompt_cache_write_tokens=getattr(usage.input_tokens_details, "cache_write_tokens", None) if usage else None,
        web_search_calls=usage.server_tool_use_details.web_search_requests
        if usage and usage.server_tool_use_details
        else None,
    )
    finish = "refusal" if refused else ("tool_calls" if calls else "stop")
    if raw.status != "completed":
        finish = "length" if raw.status == "incomplete" else "provider_failure"
    return LlmResponse(
        id=raw.id,
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text="".join(text)),),
                    tool_calls=tuple(calls),
                ),
            ),
        ),
        usage=normalized_usage,
        finish_reason=finish,
        metadata={
            "response_status": raw.status,
            "reasoning_summary": "".join(summaries),
            "sources": sources,
            "raw_response": raw.model_dump(mode="json", exclude_none=True),
        },
    )
