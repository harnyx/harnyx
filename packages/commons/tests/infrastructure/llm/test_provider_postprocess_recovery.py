from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
    PostprocessResult,
)

pytestmark = pytest.mark.anyio("asyncio")


class _ExpectedAnswer(BaseModel):
    verdict: int
    justification: str


def _response(text: str, *, role: str = "assistant", response_id: str = "response-id") -> LlmResponse:
    return LlmResponse(
        id=response_id,
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role=role,
                    content=(LlmMessageContentPart(type="text", text=text),),
                ),
            ),
        ),
        usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class _SequencedProvider(BaseLlmProvider):
    def __init__(
        self,
        *,
        responses: list[LlmResponse],
        postprocessor: Callable[[LlmResponse], PostprocessResult],
        provider_label: str = "openai",
        max_concurrent: int | None = None,
    ) -> None:
        super().__init__(provider_label=provider_label, max_concurrent=max_concurrent)
        self._retry_policy = RetryPolicy(attempts=len(responses), initial_ms=0, max_ms=0, jitter=0.0)
        self._responses = list(responses)
        self.requests: list[AbstractLlmRequest] = []
        self._postprocessor = postprocessor

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        async def _call(current_request: AbstractLlmRequest) -> LlmResponse:
            self.requests.append(current_request)
            if not self._responses:
                raise AssertionError("test exhausted response sequence")
            return self._responses.pop(0)

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=lambda _: (True, False, None),
        )


def _request(
    *,
    provider: str = "openai",
    model: str = "gpt-5-mini",
    postprocessor: Callable[[LlmResponse], PostprocessResult] | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider=provider,
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        postprocessor=postprocessor or pydantic_postprocessor(_ExpectedAnswer),
    )


async def test_provider_uses_feedback_retry_for_json_decode_failure() -> None:
    provider = _SequencedProvider(
        responses=[
            _response("not valid json", response_id="resp-1"),
            _response(json.dumps({"verdict": 1, "justification": "repaired"}), response_id="resp-2"),
        ],
        postprocessor=pydantic_postprocessor(_ExpectedAnswer),
    )

    result = await provider.invoke(_request())

    assert len(provider.requests) == 2
    retry_request = provider.requests[1]
    assert retry_request is not provider.requests[0]
    assert retry_request.messages[0].content[0].text == "hello"
    assert retry_request.messages[1].role == "assistant"
    assert retry_request.messages[1].content[0].text == "not valid json"
    assert retry_request.messages[2].role == "user"
    assert "json decode error:" in retry_request.messages[2].content[0].text
    assert "original instructions" in retry_request.messages[2].content[0].text
    assert result.postprocessed == _ExpectedAnswer(verdict=1, justification="repaired")
    assert result.metadata is not None
    assert result.metadata["postprocess_recoveries"] == (
        {
            "kind": "retry_with_feedback",
            "response_id": "resp-1",
            "feedback_role": "user",
        },
    )


async def test_provider_uses_feedback_retry_for_validation_failure() -> None:
    provider = _SequencedProvider(
        responses=[
            _response(json.dumps({"verdict": 1}), response_id="resp-1"),
            _response(json.dumps({"verdict": 1, "justification": "completed"}), response_id="resp-2"),
        ],
        postprocessor=pydantic_postprocessor(_ExpectedAnswer),
    )

    result = await provider.invoke(_request())

    assert len(provider.requests) == 2
    retry_request = provider.requests[1]
    assert retry_request.messages[1].content[0].text == json.dumps({"verdict": 1})
    assert "justification" in retry_request.messages[2].content[0].text
    assert result.postprocessed == _ExpectedAnswer(verdict=1, justification="completed")


async def test_provider_feedback_retry_rebuilds_from_original_messages_each_attempt() -> None:
    provider = _SequencedProvider(
        responses=[
            _response("first invalid", response_id="resp-1"),
            _response("second invalid", response_id="resp-2"),
            _response("third invalid", response_id="resp-3"),
            _response(json.dumps({"verdict": 1, "justification": "fixed"}), response_id="resp-4"),
        ],
        postprocessor=pydantic_postprocessor(_ExpectedAnswer),
    )

    result = await provider.invoke(_request())

    assert len(provider.requests) == 4
    second_attempt = provider.requests[1]
    third_attempt = provider.requests[2]
    fourth_attempt = provider.requests[3]
    assert len(second_attempt.messages) == 3
    assert len(third_attempt.messages) == 3
    assert len(fourth_attempt.messages) == 3
    assert third_attempt.messages[0] == provider.requests[0].messages[0]
    assert third_attempt.messages[1].content[0].text == "second invalid"
    assert third_attempt.messages[1].content[0].text != "first invalid"
    assert fourth_attempt.messages[0] == provider.requests[0].messages[0]
    assert fourth_attempt.messages[1].content[0].text == "third invalid"
    assert fourth_attempt.messages[1].content[0].text != "second invalid"
    assert result.postprocessed == _ExpectedAnswer(verdict=1, justification="fixed")


async def test_provider_feedback_retry_does_not_deadlock_under_semaphore_limit() -> None:
    provider = _SequencedProvider(
        responses=[
            _response("not valid json", response_id="resp-1"),
            _response(json.dumps({"verdict": 1, "justification": "repaired"}), response_id="resp-2"),
        ],
        postprocessor=pydantic_postprocessor(_ExpectedAnswer),
        max_concurrent=1,
    )

    result = await asyncio.wait_for(provider.invoke(_request()), timeout=0.5)

    assert len(provider.requests) == 2
    assert result.postprocessed == _ExpectedAnswer(verdict=1, justification="repaired")


@pytest.mark.parametrize(
    ("provider_name", "model", "expected_role"),
    [
        ("vertex", "gemini-2.5-flash", "tool"),
        ("openai", "gpt-5-mini", "user"),
    ],
)
async def test_provider_feedback_retry_preserves_or_falls_back_tool_role(
    provider_name: str,
    model: str,
    expected_role: str,
) -> None:
    provider = _SequencedProvider(
        responses=[
            _response("broken tool payload", role="tool", response_id="resp-1"),
            _response(json.dumps({"verdict": 1, "justification": "repaired"}), response_id="resp-2"),
        ],
        postprocessor=pydantic_postprocessor(_ExpectedAnswer),
        provider_label=provider_name,
    )

    await provider.invoke(_request(provider=provider_name, model=model))

    retry_request = provider.requests[1]
    assert retry_request.messages[2].role == expected_role


async def test_provider_retryable_postprocess_failure_without_recovery_keeps_original_history() -> None:
    attempts = 0

    def _semantic_postprocessor(response: LlmResponse) -> PostprocessResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return PostprocessResult(
                ok=False,
                retryable=True,
                reason="generated unique task count below minimum_task_total",
                processed=None,
                recovery=None,
            )
        return PostprocessResult(ok=True, retryable=False, reason=None, processed="ok")

    provider = _SequencedProvider(
        responses=[_response("first"), _response("second")],
        postprocessor=_semantic_postprocessor,
    )

    result = await provider.invoke(_request(postprocessor=_semantic_postprocessor))

    assert len(provider.requests) == 2
    assert provider.requests[1].messages == provider.requests[0].messages
    assert result.postprocessed == "ok"
