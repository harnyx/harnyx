"""Shared HTTP serialization helpers for tool execution responses."""

from __future__ import annotations

from caster_commons.domain.tool_call import SearchToolResult, ToolResult
from caster_commons.tools.dto import ToolInvocationResult
from caster_commons.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)


def serialize_tool_execute_response(result: ToolInvocationResult) -> ToolExecuteResponseDTO:
    receipt = result.receipt
    results = tuple(_serialize_tool_result(r) for r in receipt.metadata.results)
    usage = _serialize_usage(result)
    budget = ToolBudgetDTO(
        session_budget_usd=result.budget.session_budget_usd,
        session_used_budget_usd=result.budget.session_used_budget_usd,
        session_remaining_budget_usd=result.budget.session_remaining_budget_usd,
    )
    return ToolExecuteResponseDTO(
        receipt_id=receipt.receipt_id,
        response=result.response_payload,
        results=results,
        result_policy=receipt.metadata.result_policy.value,
        cost_usd=receipt.metadata.cost_usd,
        usage=usage,
        budget=budget,
    )


def _serialize_usage(result: ToolInvocationResult) -> ToolUsageDTO | None:
    usage = result.usage
    if usage is None:
        return None
    if (
        usage.prompt_tokens is None
        and usage.completion_tokens is None
        and usage.total_tokens is None
    ):
        return None
    return ToolUsageDTO(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _serialize_tool_result(tool_result: ToolResult) -> ToolResultDTO:
    if isinstance(tool_result, SearchToolResult):
        return ToolResultDTO(
            index=tool_result.index,
            result_id=tool_result.result_id,
            url=tool_result.url,
            note=tool_result.note,
            title=tool_result.title,
            raw=None,
        )
    return ToolResultDTO(
        index=tool_result.index,
        result_id=tool_result.result_id,
        url=None,
        note=None,
        title=None,
        raw=tool_result.raw,
    )


__all__ = ["serialize_tool_execute_response"]

