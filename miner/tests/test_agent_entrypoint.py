from __future__ import annotations

from typing import Any

import pytest

from caster_miner_sdk.api import LlmChatResult, ToolCallResponse
from caster_miner_sdk.llm import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)
from caster_miner_sdk.tools.http_models import ToolBudgetDTO, ToolResultDTO
from caster_miner_sdk.tools.search_models import SearchWebSearchResponse
from miner.tests import docker_sandbox_entrypoint as agent

pytestmark = pytest.mark.anyio("asyncio")


def _fake_budget() -> ToolBudgetDTO:
    return ToolBudgetDTO(
        session_budget_usd=1.0,
        session_used_budget_usd=0.0,
        session_remaining_budget_usd=1.0,
    )


def _fake_search_response(results: list[dict[str, Any]]) -> ToolCallResponse[SearchWebSearchResponse]:
    parsed_results = tuple(ToolResultDTO.model_validate(result) for result in results)
    return ToolCallResponse(
        receipt_id="receipt-123",
        response=SearchWebSearchResponse(data=[]),
        results=parsed_results,
        result_policy="referenceable",
        cost_usd=0.003,
        usage=None,
        budget=_fake_budget(),
    )


def _fake_chutes_response(content: str) -> LlmChatResult:
    response = LlmResponse(
        id="resp-1",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=content),),
                ),
            ),
        ),
        usage=LlmUsage(),
    )
    return LlmChatResult(
        receipt_id="chat-1",
        response=response,
        results=(),
        result_policy="log_only",
        cost_usd=0.01,
        usage=None,
        budget=_fake_budget(),
    )


async def test_evaluate_criterion_builds_supported_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        {
            "index": 0,
            "result_id": "r-1",
            "url": "https://example.com/a",
            "note": "Alpha evidence",
            "title": "Alpha",
        },
        {
            "index": 1,
            "result_id": "r-2",
            "url": "https://example.com/b",
            "note": "Beta evidence",
            "title": "Beta",
        },
    ]

    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _fake_search_response(fake_results)

    async def fake_llm_chat(**__: object) -> LlmChatResult:
        return _fake_chutes_response('{"verdict": 1, "justification": "Evidence 1 confirms."}')

    monkeypatch.setattr(
        agent,
        "search_web",
        fake_search_web,
    )
    monkeypatch.setattr(
        agent,
        "llm_chat",
        fake_llm_chat,
    )

    result = await agent.evaluate_criterion(
        {
            "claim_text": "Caster Subnet validators manage sandboxed miners.",
            "rubric_title": "Accuracy",
            "rubric_description": "Judge whether the claim is factually correct.",
            "verdict_options": [
                {"value": -1, "description": "Fail"},
                {"value": 1, "description": "Pass"},
            ],
        },
    )

    assert result["verdict"] == 1
    assert "Evidence 1" in result["justification"]
    assert len(result["citations"]) == 2
    assert {c["result_id"] for c in result["citations"]} == {"r-1", "r-2"}
    assert all(c["receipt_id"] == "receipt-123" for c in result["citations"])


async def test_evaluate_criterion_raises_when_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search_web(*_: object, **__: object) -> ToolCallResponse[SearchWebSearchResponse]:
        return _fake_search_response([])

    monkeypatch.setattr(agent, "search_web", fake_search_web)

    with pytest.raises(RuntimeError):
        await agent.evaluate_criterion(
            {
                "claim_text": "Test",
                "rubric_title": "Accuracy",
                "rubric_description": "Judge",
                "verdict_options": [
                    {"value": -1, "description": "Fail"},
                    {"value": 1, "description": "Pass"},
                ],
            },
        )
