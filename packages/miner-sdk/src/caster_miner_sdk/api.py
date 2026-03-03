from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast
from uuid import UUID

from caster_miner_sdk._internal.tool_invoker import _current_tool_invoker
from caster_miner_sdk.llm import LlmResponse
from caster_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)
from caster_miner_sdk.tools.search_models import (
    FeedSearchResponse,
    GetRepoFileResponse,
    SearchAiSearchResponse,
    SearchRepoSearchResponse,
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


def _require_response_mapping(response_payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(response_payload, Mapping):
        raise RuntimeError(label)
    return cast(Mapping[str, Any], response_payload)


async def test_tool(message: str) -> ToolCallResponse[TestToolResponse]:
    """Invoke the validator-hosted test tool."""

    raw_response = await _current_tool_invoker().invoke("test_tool", args=(message,), kwargs={})
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="test_tool response payload must be a mapping")
    response = TestToolResponse(
        status=str(response_payload.get("status", "")),
        echo=str(response_payload.get("echo", "")),
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

async def tooling_info() -> ToolCallResponse[dict[str, Any]]:
    """Fetch tool pricing and current session budget metadata."""

    raw_response = await _current_tool_invoker().invoke("tooling_info", args=(), kwargs={})
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


async def search_web(query: str, /, **kwargs: Any) -> ToolCallResponse[SearchWebSearchResponse]:
    """Execute the validator-hosted search tool and return its response payload."""

    payload = {"query": query}
    payload.update(kwargs)
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


async def search_x(query: str, /, **kwargs: Any) -> ToolCallResponse[SearchXSearchResponse]:
    """Execute the validator-hosted X search tool and return its response payload."""

    payload = {"query": query}
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("search_x", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_x response payload must be a mapping")
    response = SearchXSearchResponse.model_validate(response_payload)
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


async def search_repo(
    *,
    repo_url: str,
    commit_sha: str,
    query: str,
    **kwargs: Any,
) -> ToolCallResponse[SearchRepoSearchResponse]:
    """Execute the validator-hosted repository search tool."""

    payload = {
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "query": query,
    }
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("search_repo", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_repo response payload must be a mapping")
    response = SearchRepoSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def get_repo_file(
    *,
    repo_url: str,
    commit_sha: str,
    path: str,
    **kwargs: Any,
) -> ToolCallResponse[GetRepoFileResponse]:
    """Execute the validator-hosted repository file tool."""

    payload = {
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "path": path,
    }
    payload.update(kwargs)
    raw_response = await _current_tool_invoker().invoke("get_repo_file", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="get_repo_file response payload must be a mapping")
    response = GetRepoFileResponse.model_validate(response_payload)
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


async def search_items(
    *,
    feed_id: UUID | str,
    enqueue_seq: int,
    search_queries: Sequence[str],
    num_hit: int,
) -> ToolCallResponse[FeedSearchResponse]:
    """Query prior similar items in a feed via the host-provided tool."""

    payload = {
        "feed_id": str(feed_id),
        "enqueue_seq": int(enqueue_seq),
        "search_queries": list(search_queries),
        "num_hit": int(num_hit),
    }
    raw_response = await _current_tool_invoker().invoke("search_items", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_items response payload must be a mapping")
    response = FeedSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
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
    "search_repo",
    "get_repo_file",
    "search_items",
    "test_tool",
    "tooling_info",
    "ToolCallResponse",
    "LlmChatResult",
    "TestToolResponse",
]
