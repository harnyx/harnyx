from __future__ import annotations

import logging
from dataclasses import replace

import pytest

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
from harnyx_commons.llm.similarity_observability import (
    SimilarityLlmAttemptObservation,
    record_similarity_stream_event,
    record_similarity_stream_headers_received,
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
        record_similarity_stream_headers_received()
        record_similarity_stream_event(saw_output=True)
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


async def test_similarity_retry_logs_each_stream_attempt_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Future failure: a slow similarity fallback cannot be separated from a stalled retry."""
    provider = _RetryOnceExceptionProvider()
    request = replace(
        _request(use_case="miner_task_similarity_judge"),
        internal_metadata={
            "batch_id": "batch-1",
            "candidate_artifact_id": "candidate-1",
            "reference_artifact_id": "reference-1",
            "candidate_model": "configured-model",
            "candidate_position": 1,
            "candidate_count": 2,
            "private_sentinel": "must-not-be-logged",
        },
        include_payloads_in_observability=False,
    )
    caplog.set_level(logging.INFO, logger="harnyx_commons.llm.calls")

    await provider.invoke_with_retry(request)

    records = [record for record in caplog.records if record.message.startswith("similarity_judge.llm_attempt.")]
    assert [record.message for record in records] == [
        "similarity_judge.llm_attempt.started",
        "similarity_judge.llm_attempt.headers_received",
        "similarity_judge.llm_attempt.progress",
        "similarity_judge.llm_attempt.finished",
        "similarity_judge.llm_attempt.started",
        "similarity_judge.llm_attempt.headers_received",
        "similarity_judge.llm_attempt.progress",
        "similarity_judge.llm_attempt.finished",
    ]
    data = [record.data for record in records]
    assert {item["attempt_number"] for item in data} == {1, 2}
    assert len({item["attempt_id"] for item in data}) == 2
    assert {item["attempt_limit"] for item in data} == {2}
    assert all(item["batch_id"] == "batch-1" for item in data)
    assert data[3]["outcome"] == "failed"
    assert data[3]["exception_type"] == "RuntimeError"
    assert data[3]["retryable"] is True
    assert data[7]["outcome"] == "response_received"
    assert data[7]["finish_reason"] == "stop"
    assert data[7]["completion_tokens"] == 7
    rendered_data = repr(data)
    assert "must-not-be-logged" not in rendered_data
    assert not any(value in {"hello", "ok"} for item in data for value in item.values())


async def test_similarity_stream_reports_first_output_and_periodic_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Future failure: metadata-only events hide when productive model output actually starts."""
    clock_values = iter((0.0, 0.1, 1.0, 2.0, 61.0, 62.0, 70.0))
    caplog.set_level(logging.INFO, logger="harnyx_commons.llm.calls")
    observation = SimilarityLlmAttemptObservation(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        attempt_number=1,
        attempt_limit=1,
        identity={},
        clock=lambda: next(clock_values),
    )

    observation.headers_received()
    observation.stream_event(saw_output=False)
    observation.stream_event(saw_output=True)
    observation.stream_event(saw_output=True)
    observation.stream_event(saw_output=True)
    observation.finish_response(_response())

    progress = [record.data for record in caplog.records if record.message.endswith(".progress")]
    assert [item["stream_event_count"] for item in progress] == [1, 2, 4]
    assert progress[0]["first_output_ms"] is None
    assert progress[1]["first_output_ms"] == 2000.0
    finished = next(record.data for record in caplog.records if record.message.endswith(".finished"))
    assert finished["max_stream_event_gap_ms"] == 59000.0
    assert finished["time_since_last_stream_event_ms"] == 8000.0


async def test_similarity_attempt_logging_failure_does_not_change_provider_result() -> None:
    """Future failure: an observability handler outage breaks an otherwise valid LLM call."""

    class _ExplodingLogger(logging.Logger):
        def info(self, msg: object, *args: object, **kwargs: object) -> None:
            del msg, args, kwargs
            raise RuntimeError("log sink unavailable")

    observation = SimilarityLlmAttemptObservation(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
        attempt_number=1,
        attempt_limit=1,
        identity={},
        logger=_ExplodingLogger("exploding"),
    )

    observation.headers_received()
    observation.stream_event(saw_output=True)
    observation.finish_response(_response())


async def test_non_retryable_exception_raises_provider_error_without_retry_exhaustion() -> None:
    provider = _NonRetryableExceptionProvider()

    with pytest.raises(LlmProviderError, match="invalid_request: bad request") as exc_info:
        await provider.invoke_with_retry(_request())

    assert isinstance(exc_info.value.__cause__, ValueError)


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


async def test_request_retry_policy_attempts_one_disables_retry() -> None:
    provider = _RetryOnceExceptionProvider()
    provider._retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)
    request = replace(
        _request(),
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0),
    )

    with pytest.raises(LlmRetryExhaustedError) as raised:
        await provider.invoke_with_retry(request)

    assert raised.value.attempts == 1
