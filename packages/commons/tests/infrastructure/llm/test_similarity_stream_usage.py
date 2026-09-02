from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx
import pytest

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = pytest.mark.anyio("asyncio")


class _InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, *, cancel: bool, include_usage: bool) -> None:
        self.cancel = cancel
        self.include_usage = include_usage
        self.consumed = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for completion, reasoning in ((1, 1), (6, 4)):
            event = {
                "id": "gen-stream",
                "provider": "Makora",
                "choices": [{"index": 0, "delta": {"reasoning_content": "private reasoning"}}],
            }
            if self.include_usage:
                event["usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": completion,
                    "reasoning_tokens": reasoning,
                    "total_tokens": 100 + completion,
                }
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        if self.include_usage:
            # A malformed update must not erase the last usable counters or replace the transport failure.
            yield b'data: {"usage": {"completion_tokens": "invalid"}}\n\n'
        yield b'data: {"choices": [{"delta": {"content": "private output"}}]}\n\n'
        self.consumed.set()
        if self.cancel:
            await asyncio.Event().wait()
        raise httpx.ReadTimeout("stream stalled")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("provider_name", ("chutes", "openai-compatible"))
@pytest.mark.parametrize("cancel", (False, True), ids=("timeout", "cancelled"))
@pytest.mark.parametrize("include_usage", (False, True), ids=("missing-usage", "partial-usage"))
async def test_interrupted_similarity_stream_retains_available_usage_and_closes(
    provider_name: str, cancel: bool, include_usage: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Future failure: interrupted calls lose counters, invent zero usage, or return a partial verdict."""
    stream = _InterruptedStream(cancel=cancel, include_usage=include_usage)
    client = httpx.AsyncClient(
        base_url="https://example.com",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=stream, headers={"x-generation-id": "gen-header"})
        ),
    )
    if provider_name == "chutes":
        provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key", client=client)
    else:
        provider = OpenAiCompatibleLlmProvider(
            endpoint=OpenAiCompatibleEndpointConfig.model_validate(
                {"id": "openrouter", "base_url": "https://example.com", "auth": {"type": "none"}}
            ),
            client=client,
        )
    request = LlmRequest(
        provider="chutes" if provider_name == "chutes" else "custom-openai-compatible:openrouter",
        model="zai-org/GLM-5.2-TEE" if provider_name == "chutes" else "deepseek/deepseek-v4-flash-0731",
        messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("private prompt"),)),),
        temperature=0.0,
        max_output_tokens=None,
        use_case="miner_task_similarity_judge",
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0),
        include_payloads_in_observability=False,
    )
    caplog.set_level(logging.INFO, logger="harnyx_commons.llm.calls")
    invocation = asyncio.create_task(provider.invoke(request))
    try:
        if cancel:
            await asyncio.wait_for(stream.consumed.wait(), timeout=5)
            invocation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await invocation
        else:
            with pytest.raises(LlmRetryExhaustedError):
                await invocation
    finally:
        if not invocation.done():
            invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)
        await provider.aclose()
        await client.aclose()

    finished = [r.data for r in caplog.records if r.message == "similarity_judge.llm_attempt.finished"]
    assert len(finished) == 1
    result = finished[0]
    assert result["outcome"] == ("cancelled" if cancel else "failed")
    assert result["response_id"] == "gen-stream"
    assert result["upstream_provider"] == ("Makora" if provider_name == "openai-compatible" else None)
    assert stream.closed
    if include_usage:
        assert result["usage_status"] == "partial"
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 2
        assert result["reasoning_tokens"] == 4
        assert result["total_tokens"] == 106
        assert 0 <= result["last_usage_ms"] <= result["last_stream_event_ms"] <= result["elapsed_ms"]
    else:
        assert result["usage_status"] == "unavailable"
        assert result["prompt_tokens"] is None
        assert result["completion_tokens"] is None
        assert result["reasoning_tokens"] is None
        assert result["total_tokens"] is None
        assert result["last_usage_ms"] is None
    assert "private" not in repr(finished)
