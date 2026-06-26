from __future__ import annotations

import pytest

from harnyx_commons.domain.verdict import VerdictOption, VerdictOptions
from harnyx_commons.llm.grading import JustificationGrader, JustificationGraderConfig
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmResponse, LlmUsage

pytestmark = pytest.mark.anyio("asyncio")

_VERDICT_OPTIONS = VerdictOptions(
    options=(
        VerdictOption(value=-1, description="Fail"),
        VerdictOption(value=1, description="Pass"),
    )
)


class RecordingProvider:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def invoke(self, request: object) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="grade-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(role="assistant", content=(), reasoning=None),
                ),
            ),
            usage=LlmUsage(),
            postprocessed={"rationale": "same support", "support_ok": True},
        )

    async def aclose(self) -> None:
        return None


async def test_justification_grader_forwards_reasoning_effort_without_typed_thinking() -> None:
    provider = RecordingProvider()
    grader = JustificationGrader(
        provider,
        JustificationGraderConfig(
            provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            reasoning_effort="high",
        ),
    )

    result = await grader.grade(
        claim_text="The reference is supported.",
        reference_verdict=1,
        reference_justification="The cited text supports it.",
        miner_verdict=1,
        miner_justification="The cited text supports it.",
        verdict_options=_VERDICT_OPTIONS,
    )

    assert result.support_ok is True
    request = provider.requests[0]
    assert request.reasoning_effort == "high"
    assert request.thinking is None
