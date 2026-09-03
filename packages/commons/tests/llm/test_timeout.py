from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.ai_gateway import AiGatewayLlmProvider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter import OpenRouterLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_commons.llm.timeout import (
    LlmAttemptTimeoutError,
    enforce_attempt_deadlines,
    record_output_progress,
    resolve_timeout,
    streaming_transport_timeout,
)
from harnyx_miner_sdk.llm import Timeout

pytestmark = pytest.mark.anyio("asyncio")


def test_timeout_inherits_only_omitted_phases() -> None:
    defaults = Timeout(900, prefill=300, inactivity=60)
    assert resolve_timeout(120, defaults) == Timeout(120, prefill=300, inactivity=60)
    assert resolve_timeout(Timeout(120, inactivity=10), defaults) == Timeout(120, prefill=300, inactivity=10)
    assert defaults == Timeout(900, prefill=300, inactivity=60)
    assert resolve_timeout(120) == Timeout(120)


async def test_slow_prefill_does_not_use_inactivity_deadline() -> None:
    async with enforce_attempt_deadlines(Timeout(1, prefill=0.5, inactivity=0.01)):
        await asyncio.sleep(0.03)
        record_output_progress()


async def test_parent_cancellation_is_not_an_owned_timeout() -> None:
    started = asyncio.Event()

    async def invoke() -> None:
        async with enforce_attempt_deadlines(Timeout(10, prefill=5)):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(invoke())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record_output_progress()  # the cancelled task cannot leak its controller


async def test_expired_attempt_cannot_be_revived_by_late_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from harnyx_commons.llm import timeout as module

    async with enforce_attempt_deadlines(Timeout(10, prefill=5)):
        attempt = module._current_attempt.get()
        assert attempt is not None
        monkeypatch.setattr(attempt, "now", lambda: attempt.started_at + 6)
        with pytest.raises(LlmAttemptTimeoutError) as exc:
            record_output_progress()
        assert exc.value.phase == "prefill"
        monkeypatch.undo()


class _Stream(httpx.AsyncByteStream):
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        # Headers, role, empty reasoning, and usage cannot establish output.
        yield b': heartbeat\n\ndata: {"choices":[{"delta":{"role":"assistant","reasoning_details":[]}}]}\n\n'
        if self.scenario == "prefill":
            while True:
                yield b'data: {"choices":[],"usage":{"prompt_tokens":1}}\n\n'
                await asyncio.sleep(0.005)
        if self.scenario == "tool_inactivity":
            yield _event(
                {"tool_calls": [{"index": 0, "id": "call", "type": "function", "function": {"name": "lookup"}}]}
            )
            while True:
                yield _event({"tool_calls": [{"index": 0, "function": {}}]})
                await asyncio.sleep(0.005)
        if self.scenario in {"encrypted_total", "encrypted_inactivity"}:
            yield _event({"reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque"}]})
            while True:
                detail = (
                    {"type": "reasoning.encrypted", "data": "opaque"}
                    if self.scenario == "encrypted_total"
                    else {"type": "reasoning.encrypted", "data": "", "id": "reasoning-1"}
                )
                yield _event({"reasoning_details": [detail]})
                await asyncio.sleep(0.005)
        yield _event({"reasoning": "thinking"})
        while True:
            yield _event({"content": "x"}) if self.scenario == "total" else b": heartbeat\n\n"
            await asyncio.sleep(0.005)

    async def aclose(self) -> None:
        self.closed = True


def _event(delta: dict[str, object]) -> bytes:
    return ("data: " + json.dumps({"id": "test", "choices": [{"index": 0, "delta": delta}]}) + "\n\n").encode()


@pytest.mark.parametrize("provider_name", ["chutes", "custom", "openrouter", "ai_gateway"])
@pytest.mark.parametrize("scenario", ["prefill", "inactivity", "total", "tool_inactivity"])
async def test_stream_deadlines_and_cleanup_are_consistent(provider_name: str, scenario: str) -> None:
    await _assert_stream_deadline(provider_name, scenario, "inactivity" if scenario == "tool_inactivity" else scenario)


@pytest.mark.parametrize("provider_name", ["custom", "openrouter", "ai_gateway"])
@pytest.mark.parametrize("phase", ["total", "inactivity"])
async def test_encrypted_reasoning_counts_as_output_but_empty_data_does_not(provider_name: str, phase: str) -> None:
    await _assert_stream_deadline(provider_name, f"encrypted_{phase}", phase)


async def _assert_stream_deadline(provider_name: str, scenario: str, expected_phase: str) -> None:
    stream = _Stream(scenario)
    client = httpx.AsyncClient(
        base_url="https://example.com",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
    )
    if provider_name == "chutes":
        provider = ChutesLlmProvider(base_url="https://example.com", api_key="test-key", client=client)
    elif provider_name == "ai_gateway":
        provider = AiGatewayLlmProvider(ai_gateway_api_key=SecretStr("test-key"), client=client)
    else:
        endpoint = OpenAiCompatibleEndpointConfig.model_validate(
            {"id": "test", "base_url": "https://example.com/v1", "auth": {"type": "none"}}
        )
        provider = OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client)
        if provider_name == "openrouter":
            delegate = provider
            provider = OpenRouterLlmProvider(
                openrouter_api_key=SecretStr("test-key"),
                openrouter_chat_provider_factory=lambda model: (delegate, client),
            )
    request = LlmRequest(
        provider=provider_name,
        model="openai/gpt-oss-20b",
        messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
        temperature=None,
        max_output_tokens=None,
        timeout=Timeout(0.15, prefill=0.05, inactivity=0.05),
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
        use_case="miner_task_similarity_judge",
    )
    try:
        with pytest.raises(LlmRetryExhaustedError) as exc:
            await provider.invoke(request)
        assert isinstance(exc.value.__cause__, LlmAttemptTimeoutError)
        assert exc.value.__cause__.phase == expected_phase
        assert stream.closed
    finally:
        await provider.aclose()
        await client.aclose()


def test_transport_setup_keeps_request_budget_without_read_deadline() -> None:
    for policy in (300, Timeout(300, prefill=120, inactivity=60)):
        transport = streaming_transport_timeout(policy)
        assert transport.connect == transport.write == transport.pool == 300
        assert transport.read is None


def test_unchanged_tool_call_snapshots_do_not_count_as_output() -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    state = OpenAiStreamState()
    event = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
            }
        ]
    }
    assert state.merge_event(_OpenAiStreamEvent.model_validate(event), reasoning_keys=())
    assert not state.merge_event(_OpenAiStreamEvent.model_validate(event), reasoning_keys=())


async def test_concurrent_attempts_keep_independent_progress() -> None:
    async def stalled() -> str:
        with pytest.raises(LlmAttemptTimeoutError) as exc:
            async with enforce_attempt_deadlines(Timeout(1, prefill=0.04)):
                await asyncio.sleep(0.15)
        return exc.value.phase

    async def active() -> None:
        async with enforce_attempt_deadlines(Timeout(1, prefill=0.04, inactivity=0.04)):
            for _ in range(12):
                record_output_progress()
                await asyncio.sleep(0.01)

    phase, _ = await asyncio.gather(stalled(), active())
    assert phase == "prefill"


@pytest.mark.parametrize(
    "tool_call",
    [
        {"index": 0, "id": "", "type": "function"},
        {"index": 0, "function": {"name": ""}},
    ],
)
def test_empty_tool_call_metadata_does_not_establish_output(tool_call: dict[str, object]) -> None:
    from harnyx_commons.llm.providers.openai_stream import OpenAiStreamState, _OpenAiStreamEvent

    event = _OpenAiStreamEvent.model_validate({"choices": [{"delta": {"tool_calls": [tool_call]}}]})
    assert not OpenAiStreamState().merge_event(event, reasoning_keys=())
