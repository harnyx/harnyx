from __future__ import annotations

import logging

import pytest

import harnyx_commons.llm.provider as provider_module
from harnyx_commons.llm.provider import BaseLlmProvider, LlmProviderError, LlmRetryExhaustedError
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
            policy=request.retry_policy,
        )


class _RetryExhaustingExceptionProvider(BaseLlmProvider):
    def __init__(self) -> None:
        super().__init__(provider_label="openai")
        self._retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        del request
        raise RuntimeError("provider timeout")

    async def invoke_with_retry(self, request: AbstractLlmRequest) -> LlmResponse:
        async def _call(current_request: AbstractLlmRequest) -> LlmResponse:
            return await self._invoke(current_request)

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=lambda _: (True, False, None),
            classify_exception=lambda _: (True, "transport_error"),
        )


class _NonRetryableExceptionProvider(BaseLlmProvider):
    def __init__(self) -> None:
        super().__init__(provider_label="openai")
        self._retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        del request
        raise ValueError("bad request")

    async def invoke_with_retry(self, request: AbstractLlmRequest) -> LlmResponse:
        async def _call(current_request: AbstractLlmRequest) -> LlmResponse:
            return await self._invoke(current_request)

        return await self._call_with_retry(
            request,
            call_coro=_call,
            verifier=lambda _: (True, False, None),
            classify_exception=lambda exc: (False, f"invalid_request: {exc}"),
        )


def _request(*, use_case: str | None = None) -> LlmRequest:
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
        use_case=use_case,
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


async def test_retryable_exception_still_raises_retry_exhausted_after_attempts() -> None:
    provider = _RetryExhaustingExceptionProvider()

    with pytest.raises(LlmRetryExhaustedError, match="transport_error"):
        await provider.invoke_with_retry(_request())


async def test_retry_success_exposes_safe_retry_metadata() -> None:
    provider = _RetryOnceExceptionProvider()

    result = await provider.invoke_with_retry(_request())

    assert result.metadata is not None
    assert result.metadata["attempts"] == 2
    assert result.metadata["retry_reasons"] == ("transport_error: provider transport failed",)
    assert result.metadata["latency_ms_total"] >= 0
    assert "prompt_tokens" not in result.metadata
    assert "actual_cost_usd" not in result.metadata


async def test_retry_exhaustion_without_response_carries_retry_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_counter_values = iter((0.0, 0.125, 0.125, 0.5))
    monkeypatch.setattr(provider_module.time, "perf_counter", lambda: next(perf_counter_values))
    provider = _RetryExhaustingExceptionProvider()

    with pytest.raises(LlmRetryExhaustedError, match="transport_error") as raised:
        await provider.invoke_with_retry(_request())

    assert raised.value.response is None
    assert raised.value.attempts == 2
    assert raised.value.retry_reasons == ("transport_error",)
    assert raised.value.latency_ms_total == 500.0


async def test_non_retryable_exception_raises_provider_error_without_retry_exhaustion() -> None:
    provider = _NonRetryableExceptionProvider()

    with pytest.raises(LlmProviderError, match="invalid_request: bad request") as exc_info:
        await provider.invoke_with_retry(_request())

    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_retry_exception_log_includes_request_use_case(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_commons.llm.calls")
    provider = _RetryOnceExceptionProvider()

    await provider.invoke_with_retry(_request(use_case="miner_task_pairwise_judge"))

    retry_records = [record for record in caplog.records if record.name == "harnyx_commons.llm.calls"]
    assert retry_records[0].__dict__["data"]["use_case"] == "miner_task_pairwise_judge"


async def test_request_retry_policy_overrides_provider_default() -> None:
    provider = _RetryOnceExceptionProvider()
    provider._retry_policy = RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0)
    request = _request()
    request = LlmRequest(
        provider=request.provider,
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        reasoning_effort=request.reasoning_effort,
        output_mode=request.output_mode,
        retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0),
    )

    result = await provider.invoke_with_retry(request)

    assert result.choices[0].message.content[0].text == "ok"
