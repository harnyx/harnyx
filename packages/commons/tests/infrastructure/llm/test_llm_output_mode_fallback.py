from __future__ import annotations

import pytest
from pydantic import BaseModel

from caster_commons.llm.adapter import LlmProviderAdapter
from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")


class GradeSchema(BaseModel):
    rationale: str
    support_ok: bool


class StubProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:  # pragma: no cover - simple stub
        self.requests.append(request)
        return LlmResponse(
            id="stub",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=None,
                    ),
                ),
            ),
            usage=LlmUsage(),
        )


async def test_chutes_structured_request_is_rewritten_to_text() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="chutes", delegate=delegate)

    request = LlmRequest(
        provider="chutes",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hi"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        output_mode="structured",
        output_schema=GradeSchema,
    )

    await provider.invoke(request)

    adapted = delegate.requests[0]
    assert adapted.output_mode == "text"
    assert adapted.output_schema is None
    assert adapted.messages[0].role == "system"
    assert "Reply with JSON only" in adapted.messages[0].content[0].text
    assert "rationale" in adapted.messages[0].content[0].text
    assert "support_ok" in adapted.messages[0].content[0].text
    assert adapted.messages[1] == request.messages[0]
