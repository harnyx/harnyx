from __future__ import annotations

import logging

import pytest

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
)

pytestmark = pytest.mark.anyio("asyncio")


class _RetryOnceExceptionProvider(BaseLlmProvider):
    def __init__(self) -> None:
        super().__init__(provider_label="openai")
        self._attempt = 0
        self._retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        del request
        self._attempt += 1
        if self._attempt == 1:
            try:
                raise ValueError("dns lookup failed")
            except ValueError as exc:
                raise RuntimeError("provider transport failed") from exc
        return _response()

    async def invoke_with_retry(self, request: AbstractLlmRequest) -> LlmResponse:
        async def _call(current_request: AbstractLlmRequest) -> LlmResponse:
            return await self._invoke(current_request)

        def _classify(exc: Exception) -> tuple[bool, str]:
            return True, f"transport_error: {exc}"

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=lambda _: (True, False, None),
            classify_exception=_classify,
        )


def _request() -> LlmRequest:
    return LlmRequest(
        provider="openai",
        model="gpt-5-mini",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        reasoning_effort=None,
        output_mode="text",
    )


def _response() -> LlmResponse:
    return LlmResponse(
        id="response-id",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text="ok"),),
                    tool_calls=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            ),
        ),
        usage=LlmUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        metadata=None,
        finish_reason="stop",
    )


async def test_retry_exception_log_includes_exception_details(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_commons.llm.calls")
    provider = _RetryOnceExceptionProvider()

    result = await provider.invoke_with_retry(_request())

    assert result.choices[0].message.content[0].text == "ok"
    retry_records = [record for record in caplog.records if record.name == "harnyx_commons.llm.calls"]
    assert retry_records
    retry_record = retry_records[0]
    assert retry_record.message.startswith("llm.retry.exception: RuntimeError: provider transport failed")
    assert retry_record.__dict__["data"]["reason"] == "transport_error: provider transport failed"
    assert retry_record.__dict__["data"]["exception_type"] == "RuntimeError"
    assert retry_record.__dict__["data"]["exception_message"] == "provider transport failed"
    assert retry_record.__dict__["data"]["exception_repr"] == "RuntimeError('provider transport failed')"
    assert retry_record.__dict__["data"]["cause_chain"] == ("ValueError: dns lookup failed",)
