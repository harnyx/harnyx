from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from caster_miner_sdk._internal.tool_invoker import _current_tool_invoker
from caster_miner_sdk.llm import LlmResponse
from caster_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)
from caster_miner_sdk.tools.search_models import (
    SearchAiSearchResponse,
    SearchWebSearchResponse,
    SearchXSearchResponse,
)

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


@dataclass(frozen=True)
class TestToolResponse:
    status: str
    echo: str


def _parse_execute_response(raw_response: object) -> ToolExecuteResponseDTO:
    return ToolExecuteResponseDTO.model_validate(raw_response)


async def test_tool(message: str) -> ToolCallResponse[TestToolResponse]:
    """Invoke the validator-hosted test tool."""

    raw_response = await _current_tool_invoker().invoke("test_tool", args=(message,), kwargs={})
    dto = _parse_execute_response(raw_response)
    if not isinstance(dto.response, Mapping):
        raise RuntimeError("test_tool response payload must be a mapping")
    response = TestToolResponse(
        status=str(dto.response.get("status", "")),
        echo=str(dto.response.get("echo", "")),
    )
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_web(query: str, /, **kwargs: Any) -> ToolCallResponse[SearchWebSearchResponse]:
    """Execute the validator-hosted search tool and return its response payload."""

    payload = {"query": query}
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("search_web", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    if not isinstance(dto.response, Mapping):
        raise RuntimeError("search_web response payload must be a mapping")
    response = SearchWebSearchResponse.model_validate(dto.response)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_x(query: str, /, **kwargs: Any) -> ToolCallResponse[SearchXSearchResponse]:
    """Execute the validator-hosted X search tool and return its response payload."""

    payload = {"query": query}
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("search_x", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    if not isinstance(dto.response, Mapping):
        raise RuntimeError("search_x response payload must be a mapping")
    response = SearchXSearchResponse.model_validate(dto.response)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_ai(prompt: str, /, **kwargs: Any) -> ToolCallResponse[SearchAiSearchResponse]:
    """Execute the validator-hosted AI search tool and return its response payload."""

    payload = {"prompt": prompt}
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("search_ai", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    if not isinstance(dto.response, Mapping):
        raise RuntimeError("search_ai response payload must be a mapping")
    response = SearchAiSearchResponse.model_validate(dto.response)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def llm_chat(
    *,
    messages: Sequence[Mapping[str, Any]],
    model: str,
    **params: Any,
) -> LlmChatResult:
    """Invoke the validator-hosted LLM chat tool and return its response payload."""

    payload = {"model": model, "messages": [dict(message) for message in messages]}
    if "provider" in params:
        params = {k: v for k, v in params.items() if k != "provider"}
    if params:
        payload.update(params)
    raw_response = await _current_tool_invoker().invoke(
        "llm_chat",
        args=(),
        kwargs=payload,
    )
    dto = _parse_execute_response(raw_response)
    if not isinstance(dto.response, Mapping):
        raise RuntimeError("llm_chat response missing 'response' payload")
    llm = LlmResponse.from_payload(dto.response)
    return LlmChatResult(
        receipt_id=dto.receipt_id,
        response=llm,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


__all__ = [
    "llm_chat",
    "search_x",
    "search_web",
    "search_ai",
    "test_tool",
    "ToolCallResponse",
    "LlmChatResult",
    "TestToolResponse",
]
