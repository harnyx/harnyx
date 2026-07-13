from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Generic, Literal, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, field_validator

from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
from harnyx_miner_sdk.llm import (
    LlmInputTextPart,
    LlmInputToolResultPart,
    LlmMessage,
    LlmResponse,
    LlmThinkingConfig,
)
from harnyx_miner_sdk.tools.embedding_models import (
    EmbeddingInputType,
    EmbeddingProviderName,
    EmbedTextRequest,
    EmbedTextResponse,
)
from harnyx_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)
from harnyx_miner_sdk.tools.llm_chat_models import (
    LlmChatFunctionTool,
    LlmChatMessage,
    LlmChatRequest,
    LlmChatThinking,
    LlmChatToolChoice,
)
from harnyx_miner_sdk.tools.llm_provider_extra import (
    AiGatewayExtra,
    OpenRouterExtra,
    ProviderExtra,
)
from harnyx_miner_sdk.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchProviderName,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_miner_sdk.tools.types import ToolInvocationTimeout

TResponse = TypeVar("TResponse")


@dataclass(frozen=True)
class ToolCallResponse(Generic[TResponse]):
    """Typed envelope returned by all hosted tool calls."""

    receipt_id: str
    response: TResponse
    results: tuple[ToolResultDTO, ...]
    result_policy: str
    cost_usd: float | None
    usage: ToolUsageDTO | None
    budget: ToolBudgetDTO


@dataclass(frozen=True)
class LlmChatResult(ToolCallResponse[LlmResponse]):
    """Typed payload returned by the llm_chat tool."""

    @property
    def llm(self) -> LlmResponse:
        return self.response


class TestToolResponse(BaseModel):
    status: str = ""
    echo: str = ""

    @field_validator("status", "echo", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return "" if value is None else str(value)


class _ToolingInfoInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: ToolInvocationTimeout | None = None


class _TestToolInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    timeout: ToolInvocationTimeout | None = None


def _parse_execute_response(raw_response: object) -> ToolExecuteResponseDTO:
    return ToolExecuteResponseDTO.model_validate(raw_response)


def _require_response_mapping(response_payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(response_payload, Mapping):
        raise RuntimeError(label)
    return cast(Mapping[str, Any], response_payload)


async def test_tool(
    message: str,
    *,
    timeout: float | None = None,
) -> ToolCallResponse[TestToolResponse]:
    """Invoke the validator-hosted test tool."""

    kwargs: dict[str, Any] = {}
    if timeout is not None:
        payload = _TestToolInvocationPayload.model_validate({"message": message, "timeout": timeout})
        message = payload.message
        kwargs["timeout"] = payload.timeout
    raw_response = await _current_tool_invoker().invoke("test_tool", args=(message,), kwargs=kwargs)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="test_tool response payload must be a mapping")
    response = TestToolResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def tooling_info(
    *,
    timeout: float | None = None,
) -> ToolCallResponse[dict[str, Any]]:
    """Fetch tool pricing and current session budget metadata."""

    payload = _ToolingInfoInvocationPayload.model_validate({"timeout": timeout}).model_dump(
        exclude_none=True,
        mode="json",
    )
    raw_response = await _current_tool_invoker().invoke("tooling_info", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="tooling_info response payload must be a mapping")
    response = dict(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_web(
    search_queries: str | Sequence[str],
    /,
    *,
    provider: SearchProviderName,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[SearchWebSearchResponse]:
    """Execute the validator-hosted search tool and return its response payload."""

    raw_payload = {"provider": provider, "search_queries": search_queries, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = SearchWebSearchRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("search_web", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_web response payload must be a mapping")
    response = SearchWebSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_ai(
    prompt: str,
    /,
    *,
    provider: SearchProviderName,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[SearchAiSearchResponse]:
    """Execute the validator-hosted AI search tool and return its response payload."""

    raw_payload = {"provider": provider, "prompt": prompt, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = SearchAiSearchRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("search_ai", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_ai response payload must be a mapping")
    response = SearchAiSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def fetch_page(
    url: str,
    /,
    *,
    provider: SearchProviderName,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[FetchPageResponse]:
    """Execute the validator-hosted page fetch tool and return its response payload."""

    raw_payload = {"provider": provider, "url": url, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = FetchPageRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("fetch_page", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="fetch_page response payload must be a mapping")
    response = FetchPageResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


@overload
async def embed_text(
    texts: str | Sequence[str],
    /,
    *,
    input_type: EmbeddingInputType,
    provider: Literal["openrouter"],
    model: str,
    instruction: str | None = None,
    dimensions: int | None = None,
    provider_extra: Mapping[str, Any] | OpenRouterExtra | None = None,
    timeout: float | None = None,
) -> ToolCallResponse[EmbedTextResponse]: ...


@overload
async def embed_text(
    texts: str | Sequence[str],
    /,
    *,
    input_type: EmbeddingInputType,
    provider: Literal["chutes"],
    model: str,
    instruction: str | None = None,
    dimensions: int | None = None,
    provider_extra: None = None,
    timeout: float | None = None,
) -> ToolCallResponse[EmbedTextResponse]: ...


async def embed_text(
    texts: str | Sequence[str],
    /,
    *,
    input_type: EmbeddingInputType,
    provider: EmbeddingProviderName,
    model: str,
    instruction: str | None = None,
    dimensions: int | None = None,
    provider_extra: Mapping[str, Any] | OpenRouterExtra | None = None,
    timeout: float | None = None,
) -> ToolCallResponse[EmbedTextResponse]:
    """Embed query or document text with the validator-hosted embedding tool."""

    payload_raw: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "texts": texts,
        "input_type": input_type,
        "instruction": instruction,
        "dimensions": dimensions,
        "timeout": timeout,
    }
    if provider_extra is not None:
        payload_raw["provider_extra"] = (
            provider_extra.to_request_extra() if isinstance(provider_extra, OpenRouterExtra) else provider_extra
        )
    payload = EmbedTextRequest.model_validate(payload_raw).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("embed_text", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="embed_text response payload must be a mapping")
    response = EmbedTextResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


@overload
async def llm_chat(
    *,
    provider: Literal["openrouter"],
    messages: Sequence[Mapping[str, Any] | LlmChatMessage | LlmMessage],
    model: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any] | LlmChatFunctionTool] | None = None,
    tool_choice: LlmChatToolChoice | Mapping[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    thinking: Mapping[str, Any] | LlmChatThinking | LlmThinkingConfig | None = None,
    provider_extra: Mapping[str, Any] | OpenRouterExtra | None = None,
    timeout: float | None = None,
) -> LlmChatResult: ...


@overload
async def llm_chat(
    *,
    provider: Literal["ai_gateway"],
    messages: Sequence[Mapping[str, Any] | LlmChatMessage | LlmMessage],
    model: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any] | LlmChatFunctionTool] | None = None,
    tool_choice: LlmChatToolChoice | Mapping[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    thinking: Mapping[str, Any] | LlmChatThinking | LlmThinkingConfig | None = None,
    provider_extra: Mapping[str, Any] | AiGatewayExtra | None = None,
    timeout: float | None = None,
) -> LlmChatResult: ...


@overload
async def llm_chat(
    *,
    provider: Literal["chutes"],
    messages: Sequence[Mapping[str, Any] | LlmChatMessage | LlmMessage],
    model: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any] | LlmChatFunctionTool] | None = None,
    tool_choice: LlmChatToolChoice | Mapping[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    thinking: Mapping[str, Any] | LlmChatThinking | LlmThinkingConfig | None = None,
    provider_extra: None = None,
    timeout: float | None = None,
) -> LlmChatResult: ...


async def llm_chat(
    *,
    provider: Literal["chutes", "openrouter", "ai_gateway"],
    messages: Sequence[Mapping[str, Any] | LlmChatMessage | LlmMessage],
    model: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any] | LlmChatFunctionTool] | None = None,
    tool_choice: LlmChatToolChoice | Mapping[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    thinking: Mapping[str, Any] | LlmChatThinking | LlmThinkingConfig | None = None,
    provider_extra: Mapping[str, Any] | ProviderExtra | None = None,
    timeout: float | None = None,
    **params: Any,
) -> LlmChatResult:
    """Invoke the validator-hosted LLM chat tool and return its response payload."""

    payload_raw: dict[str, object] = {
        "provider": provider,
        "model": model,
        "messages": [_llm_chat_message_input(message) for message in messages],
        "temperature": temperature,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
    }
    if max_output_tokens is not None:
        payload_raw["max_output_tokens"] = max_output_tokens
    if max_tokens is not None:
        payload_raw["max_tokens"] = max_tokens
    if thinking is not None:
        payload_raw["thinking"] = asdict(thinking) if isinstance(thinking, LlmThinkingConfig) else thinking
    if provider_extra is not None:
        if isinstance(provider_extra, OpenRouterExtra | AiGatewayExtra):
            payload_raw["provider_extra"] = provider_extra.to_request_extra()
        else:
            payload_raw["provider_extra"] = provider_extra
    if timeout is not None:
        payload_raw["timeout"] = timeout
    if params:
        payload_raw.update(params)
    payload = LlmChatRequest.model_validate(payload_raw).model_dump(
        exclude_none=True,
        mode="json",
        by_alias=True,
    )
    raw_response = await _current_tool_invoker().invoke(
        "llm_chat",
        args=(),
        kwargs=payload,
    )
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="llm_chat response missing 'response' payload")
    llm = LlmResponse.from_payload(response_payload)
    return LlmChatResult(
        receipt_id=dto.receipt_id,
        response=llm,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


def _llm_chat_message_input(message: Mapping[str, Any] | LlmChatMessage | LlmMessage) -> object:
    if isinstance(message, Mapping):
        return dict(message)
    if not isinstance(message, LlmMessage):
        return message

    text_parts = [part.text for part in message.content if isinstance(part, LlmInputTextPart)]
    if message.role == "assistant":
        if len(text_parts) != len(message.content):
            raise ValueError("assistant messages can contain only input_text parts")
        return {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
            "tool_calls": [asdict(tool_call) for tool_call in message.tool_calls or ()] or None,
            "reasoning_details": list(message.reasoning_details) if message.reasoning_details is not None else None,
        }
    tool_results = [part for part in message.content if isinstance(part, LlmInputToolResultPart)]
    if message.role == "tool":
        if len(message.content) != 1 or len(tool_results) != 1:
            raise ValueError("tool messages require exactly one input_tool_result part")
        result = tool_results[0]
        return {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.output_json,
            "name": result.name,
        }
    if len(text_parts) != len(message.content):
        raise ValueError("system and user messages can contain only input_text parts")
    return {"role": message.role, "content": "\n".join(text_parts)}


__all__ = [
    "embed_text",
    "fetch_page",
    "llm_chat",
    "search_web",
    "search_ai",
    "test_tool",
    "tooling_info",
    "ToolCallResponse",
    "EmbedTextResponse",
    "LlmChatResult",
    "TestToolResponse",
]
