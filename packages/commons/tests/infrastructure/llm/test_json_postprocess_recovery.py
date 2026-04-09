from __future__ import annotations

import json

from pydantic import BaseModel

from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage


class _ExampleAnswer(BaseModel):
    verdict: int
    justification: str


def _response(text: str) -> LlmResponse:
    return LlmResponse(
        id="response-id",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=text),),
                ),
            ),
        ),
        usage=LlmUsage(),
    )


def test_pydantic_postprocessor_emits_recovery_context_for_json_decode_failure() -> None:
    postprocessor = pydantic_postprocessor(_ExampleAnswer)

    result = postprocessor(_response("not valid json"))

    assert result.ok is False
    assert result.retryable is True
    assert result.recovery is not None
    assert result.recovery.kind == "retry_with_feedback"
    assert result.recovery.failure_reason.startswith("json decode error:")


def test_pydantic_postprocessor_emits_recovery_context_for_validation_failure() -> None:
    postprocessor = pydantic_postprocessor(_ExampleAnswer)

    result = postprocessor(_response(json.dumps({"verdict": 1})))

    assert result.ok is False
    assert result.retryable is True
    assert result.recovery is not None
    assert result.recovery.kind == "retry_with_feedback"
    assert "justification" in result.recovery.failure_reason
