"""Retry provider requests without replaying completed agent turns or peer work."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from time import monotonic

import httpx
import pytest
from google.auth import exceptions as google_auth_errors

from harnyx_commons.miner_task_generation.agent_runner import CallCapture, retry_after, retry_request
from harnyx_commons.miner_task_generation.contracts import CandidateSession

pytestmark = pytest.mark.anyio("asyncio")


def openai_sse(status, *, code=None, answer="complete answer", summary=None, no_output=False):
    """Pinned Responses SDK fixtures; every physical attempt carries its own receipt."""
    response = {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0.0,
        "model": "gpt-6-luna",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "tool_usage": {"web_search": {"num_requests": 0}},
        "error": {"code": code, "message": "offline failure"} if code else None,
        "output": []
        if status != "completed" or no_output
        else [
            {
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }
        ],
    }
    events = []
    if summary is not None:
        events.append(
            {
                "type": "response.reasoning_summary_text.done",
                "sequence_number": 0,
                "item_id": "rs_offline",
                "output_index": 0,
                "summary_index": 0,
                "text": summary,
            }
        )
    events.append(
        {"type": "error", "sequence_number": len(events), "code": code, "message": "offline failure", "param": None}
        if status == "error"
        else {"type": f"response.{status}", "sequence_number": len(events), "response": response}
    )
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.parametrize("first_output", ["invalid_json", "blank", "no_final"])
@pytest.mark.parametrize("exhaust", [False, True])
async def test_model_output_failure_replays_once_without_committing_failed_output(monkeypatch, first_output, exhaust):
    """Malformed or unfinished output must recover without losing receipts or poisoning agent history."""
    import httpx2
    from agents.exceptions import ModelBehaviorError
    from openai import AsyncOpenAI

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner

    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    requests = []
    role = "reviewer" if first_output == "invalid_json" else "analyst"
    valid = json.dumps({"pass": True, "feedback": ""}) if role == "reviewer" else "complete answer"

    def respond(request):
        requests.append(json.loads(request.content))
        failed = len(requests) == 1 or exhaust
        body = openai_sse(
            "completed",
            answer=("{" if first_output == "invalid_json" else " ") if failed else valid,
            no_output=failed and first_output == "no_final",
        )
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        client = AsyncOpenAI(api_key="offline-test", max_retries=0, http_client=http)
        runner = GenerationAgentRunner(project_id=None, judge=None, openai_client=client)
        session = capture().session
        if exhaust:
            with pytest.raises(ModelBehaviorError):
                await runner._invoke_openai(role, "exact input", session, monotonic() + 5)
        else:
            result = await runner._invoke_openai(role, "exact input", session, monotonic() + 5)
            if role == "reviewer":
                assert json.loads(result.answer) == json.loads(valid)
            else:
                assert result.answer == valid
            assert [item["attempt_status"] for item in result.provider_responses] == ["failed", "completed"]
    assert len(requests) == 2
    inputs = [{key: value for key, value in request.items() if key != "prompt_cache_key"} for request in requests]
    assert inputs[0] == inputs[1]
    assert session.attempted_calls == session.tool_usage.llm.call_count == 2
    assert session.tool_usage.llm.prompt_tokens == 20
    assert session.missing_usage_calls == 0 and session.author_history == []


@pytest.mark.parametrize(
    "code,exhaust,expected_calls,event_type",
    [
        ("server_error", False, 2, "failed"),
        ("rate_limit_exceeded", False, 2, "failed"),
        ("server_error", True, 2, "failed"),
        ("invalid_prompt", False, 1, "failed"),
        (None, False, 1, "incomplete"),
        ("server_error", False, 2, "error"),
        ("invalid_prompt", False, 1, "error"),
    ],
)
async def test_real_openai_stream_retries_only_transient_codes_and_retains_all_receipts(
    monkeypatch, code, exhaust, expected_calls, event_type
):
    """Provider SSE failures must take the same bounded retries as transient HTTP errors."""
    import httpx2
    from openai import AsyncOpenAI

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner, OpenAIStreamError

    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    requests = []
    sdk_runs = []
    from agents import Runner

    original_run = Runner.run_streamed

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        sdk_runs.append(result)
        return result

    monkeypatch.setattr(Runner, "run_streamed", run)

    def respond(request):
        assert all(result.run_loop_task.done() for result in sdk_runs[:-1])
        requests.append(json.loads(request.content))
        content = openai_sse(event_type, code=code) if len(requests) == 1 or exhaust else openai_sse("completed")
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        client = AsyncOpenAI(api_key="offline-test", max_retries=0, http_client=http)
        runner = GenerationAgentRunner(project_id=None, judge=None, openai_client=client)
        session = capture().session
        if expected_calls == 2 and not exhaust:
            result = await runner._invoke_openai("analyst", "exact input", session, monotonic() + 5)
            assert result.answer == "complete answer"
            assert [item["attempt_status"] for item in result.provider_responses] == ["failed", "completed"]
            assert result.evidence_responses()[0]["response"] == {}
        else:
            with pytest.raises(OpenAIStreamError):
                await runner._invoke_openai("analyst", "exact input", session, monotonic() + 5)
    assert len(requests) == expected_calls
    # The SDK allocates a cache key per run; question, history, tools and settings must be identical.
    inputs = [{key: value for key, value in request.items() if key != "prompt_cache_key"} for request in requests]
    assert inputs == [inputs[0]] * expected_calls
    receipts = expected_calls - (1 if event_type == "error" else 0)
    assert session.attempted_calls == expected_calls and session.tool_usage.llm.call_count == receipts
    assert session.tool_usage.llm.prompt_tokens == receipts * 10
    assert session.missing_usage_calls == expected_calls - receipts and session.author_history == []
    assert all(result.run_loop_task.done() for result in sdk_runs)


async def test_retried_author_supplies_only_completed_summaries_to_reviewer_and_verifier(monkeypatch, caplog):
    """Failed research must not be attributed to the author of the selected draft."""
    import httpx2
    from openai import AsyncOpenAI
    from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import FakeRunner, run

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner

    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")
    draft = {
        "question": "Which bird is named in the source?",
        "answer": "kākāpō",
        "answer_components": ["kākāpō"],
        "scope": "The named source edition",
        "source_support": [{"url": "https://example.org/bird", "evidence": "kākāpō"}],
        "revision_note": "Current supported selection",
    }
    calls = []
    raw_results = []

    def respond(request):
        calls.append(json.loads(request.content))
        content = (
            openai_sse("failed", code="server_error", summary="abandoned-author-research")
            if len(calls) == 1
            else openai_sse("completed", answer=json.dumps(draft), summary="completed-author-research")
        )
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        client = AsyncOpenAI(api_key="offline-test", max_retries=0, http_client=http)
        actual = GenerationAgentRunner(project_id=None, judge=None, openai_client=client)

        class ResearchRunner(FakeRunner):
            async def invoke(self, role, message, session, deadline):
                if role == "author":
                    answer = await actual.invoke(role, message, session, deadline)
                    raw_results.append(answer)
                    assert "abandoned-author-research" not in str(session.author_history)
                    return answer
                if role in {"reviewer", "verifier"}:
                    packet = json.loads(message)
                    assert packet["author_trace"][0]["thought_summaries"] == ["completed-author-research"]
                    assert "abandoned-author-research" not in message
                return await super().invoke(role, message, session, deadline)

        result = await run(ResearchRunner())
    assert result.status == "finalized" and len(calls) == 2
    assert "abandoned-author-research" in str(raw_results[0].provider_responses)
    assert any("abandoned-author-research" in str(getattr(record, "data", {})) for record in caplog.records)
    assert result.tool_usage.llm.call_count == 2 and result.tool_usage.llm.prompt_tokens == 20


async def test_real_openai_cancellation_drains_stream_and_marks_receipts_without_retry(monkeypatch):
    """A cancelled stream must release its provider work before the owner releases the slot."""
    import httpx2
    from agents import Runner
    from openai import AsyncOpenAI

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner

    observed = []
    sdk_runs = []
    receipt_seen = asyncio.Event()
    closed = asyncio.Event()
    requests = []
    original_event = CallCapture.event
    original_run = Runner.run_streamed

    def event(capture, payload):
        observed.append(capture)
        index = original_event(capture, payload)
        if payload["type"] == "response.completed":
            receipt_seen.set()
        return index

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        sdk_runs.append(result)
        return result

    monkeypatch.setattr(CallCapture, "event", event)
    monkeypatch.setattr(Runner, "run_streamed", run)

    class ActiveStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield openai_sse("completed")
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    def respond(request):
        requests.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=ActiveStream())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        runner = GenerationAgentRunner(
            project_id=None,
            judge=None,
            openai_client=AsyncOpenAI(api_key="offline-test", max_retries=0, http_client=http),
        )
        session = capture().session
        invocation = asyncio.create_task(runner.invoke("analyst", "exact input", session, monotonic() + 5))
        await asyncio.wait_for(receipt_seen.wait(), 2)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, 2)
    assert len(requests) == 1 and closed.is_set()
    assert all(result.run_loop_task.done() for result in sdk_runs)
    assert all(item["attempt_status"] == "cancelled" for item in observed[0].responses)
    assert session.attempted_calls == session.tool_usage.llm.call_count == 1
    assert session.tool_usage.llm.prompt_tokens == 10 and session.missing_usage_calls == 0


def capture():
    return CallCapture(
        CandidateSession(task_id="retry", effective_date=datetime.now(UTC).date()), "solver", "test", "vertex"
    )


@pytest.mark.parametrize("error_type", [google_auth_errors.DefaultCredentialsError, google_auth_errors.RefreshError])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_local_google_credentials_fault_is_batch_terminal_without_retry(error_type, wrapped):
    from harnyx_commons.llm.provider import LlmProviderError
    from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError

    original = error_type("private credential failure")
    failure = LlmProviderError("wrapped provider failure") if wrapped else original
    if wrapped:
        failure.__cause__ = original

    async def call():
        raise failure

    observed = capture()
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await retry_request(call, observed, monotonic() + 2, solver=True)
    assert caught.value.__cause__ is failure
    assert caught.value.failure_class == "provider_auth"
    assert observed.attempted == 1
    assert "private credential failure" not in str(caught.value)


@pytest.mark.parametrize(
    "failure",
    [
        google_auth_errors.RefreshError("temporary authentication service failure", retryable=True),
        google_auth_errors.TransportError("temporary authentication transport failure"),
    ],
)
async def test_transient_google_auth_failure_retries_without_becoming_terminal(monkeypatch, failure):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    observed = capture()

    async def call():
        if observed.attempted == 1:
            raise failure
        return "completed"

    assert await retry_request(call, observed, monotonic() + 2, solver=True) == "completed"
    assert observed.attempted == 2


async def test_solver_retries_multiple_transient_failures_with_identical_inputs(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="harnyx_commons.llm.calls")
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    inputs = []

    async def call():
        inputs.append({"question": "unchanged", "completed_turn": 2})
        if len(inputs) < 4:
            raise httpx.ConnectError("private-provider-detail")
        return "completed"

    observed = capture()
    assert await retry_request(call, observed, monotonic() + 2, solver=True) == "completed"
    assert observed.attempted == 4 and len(inputs) == 4
    assert inputs == [inputs[0]] * 4
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3
    assert "private-provider-detail" not in caplog.text


async def test_non_solver_has_only_one_transient_retry_and_permanent_errors_never_retry(monkeypatch):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)

    async def transient():
        raise httpx.ConnectError("transient")

    observed = capture()
    with pytest.raises(httpx.ConnectError):
        await retry_request(transient, observed, monotonic() + 2, solver=False)
    assert observed.attempted == 2

    async def permanent():
        raise ValueError("invalid output")

    observed = capture()
    with pytest.raises(ValueError):
        await retry_request(permanent, observed, monotonic() + 2, solver=True)
    assert observed.attempted == 1


async def test_deadline_bounds_retry_after_delay():
    class Delayed(httpx.ConnectError):
        retry_after_seconds = 60

    async def call():
        raise Delayed("provider unavailable")

    observed = capture()
    with pytest.raises(TimeoutError):
        await retry_request(call, observed, monotonic() + 0.02, solver=True)
    assert observed.attempted == 1


async def test_incremental_payload_is_visible_before_completion_only_at_debug(caplog):
    observed = capture()
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    observed.event({"type": "thought_delta", "text": "exposed provider summary", "encrypted_content": "hidden"})
    assert any("exposed provider summary" in str(r.__dict__.get("data")) for r in caplog.records)
    assert "hidden" not in str([r.__dict__.get("data") for r in caplog.records])
    caplog.clear()
    caplog.set_level(logging.INFO, logger="harnyx_commons.llm.calls")
    observed.event({"text": "private second delta"})
    assert not caplog.records


def test_invalid_retry_after_does_not_replace_original_provider_failure():
    error = httpx.ConnectError("temporary")
    error.response = httpx.Response(429, headers={"retry-after": "malformed"})
    assert retry_after(error) == 0


async def test_author_roles_share_twenty_slots_across_candidates(monkeypatch):
    """Parallel candidates must not serialize behind a small cap or exceed the shared limit."""
    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
    from harnyx_commons.miner_task_generation.contracts import AgentResult

    runner = GenerationAgentRunner(project_id=None, judge=None)
    active = 0
    maximum = 0

    async def invoke(*args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return AgentResult(answer="complete")

    monkeypatch.setattr(runner, "_invoke_openai", invoke)
    await asyncio.gather(
        *(
            runner.invoke(role, "input", capture().session, monotonic() + 2)
            for role in ["author", "reviewer", "analyst", "verifier"] * 6
        )
    )
    assert maximum == 20


async def test_expired_deadline_admits_no_provider_request():
    observed = capture()

    async def immediate():
        raise AssertionError("provider must not start")

    with pytest.raises(TimeoutError):
        await retry_request(immediate, observed, monotonic() - 1, solver=True)
    assert observed.attempted == 0


def test_missing_attempt_usage_keeps_total_unavailable_and_retains_known_component_costs():
    from harnyx_commons.llm.schema import LlmUsage

    observed = capture()
    observed.model = "gemini-3.1-pro-preview"
    observed.attempted = 2
    observed.account(LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, web_search_calls=1))
    observed.finish()
    assert observed.session.tool_usage.actual_total_cost_usd is None
    assert observed.session.missing_usage_calls == 1
    assert observed.session.stage_summaries[0].available_cost_usd > 0


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-6-luna", 0.01035),
        ("vertex", "gemini-3.8-flash", 0.017375),
        ("vertex", "gemini-3.1-pro-preview", 0.0244),
    ],
)
def test_generation_capture_settles_shared_generation_rates(provider, model, expected):
    from harnyx_commons.llm.schema import LlmUsage

    observed = capture()
    observed.provider, observed.model = provider, model
    observed.attempted = 1
    observed.account(LlmUsage(prompt_tokens=1_000, completion_tokens=500, reasoning_tokens=200, web_search_calls=1))
    observed.finish()
    assert observed.session.tool_usage.actual_total_cost_usd == pytest.approx(expected)
    assert observed.session.stage_summaries[0].available_cost_usd == pytest.approx(expected)


def test_missing_openai_search_receipt_keeps_known_tokens_without_claiming_free_search():
    from harnyx_commons.llm.schema import LlmUsage

    observed = capture()
    observed.provider, observed.model = "openai", "gpt-6-luna"
    observed.attempted = 1
    observed.account(LlmUsage(prompt_tokens=1_000, completion_tokens=500))
    observed.finish()
    assert observed.session.tool_usage.actual_total_cost_usd is None
    assert observed.session.tool_usage.search_tool.actual_cost is None
    assert observed.session.stage_summaries[0].available_cost_usd == pytest.approx(0.00035)


@pytest.mark.parametrize("search_requests", [3, None])
@pytest.mark.parametrize("event_type", ["response.completed", "response.incomplete", "response.failed"])
async def test_openai_stream_accounts_provider_receipts_instead_of_tool_actions(
    monkeypatch, search_requests, event_type
):
    from types import SimpleNamespace

    from agents.stream_events import RawResponsesStreamEvent

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner

    payload = {
        "type": event_type,
        "response": {
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 500,
                "total_tokens": 1_500,
                "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300},
                "output_tokens_details": {"reasoning_tokens": 200},
            },
            "output": [{"type": "web_search_call"}, {"type": "web_search_call"}],
            "tool_usage": {"web_search": {"num_requests": search_requests}},
        },
    }

    async def events():
        yield RawResponsesStreamEvent(data=SimpleNamespace(model_dump=lambda **kwargs: payload))

    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.agent_runner.Runner.run_streamed",
        lambda *args, **kwargs: SimpleNamespace(
            stream_events=events, final_output="complete answer", cancel=lambda: None, run_loop_task=None
        ),
    )
    runner = GenerationAgentRunner(project_id=None, judge=None, openai_client=SimpleNamespace())
    session = capture().session
    if event_type == "response.completed":
        await runner._invoke_openai("analyst", "input", session, monotonic() + 2)
    else:
        with pytest.raises(RuntimeError, match="incomplete or failed"):
            await runner._invoke_openai("analyst", "input", session, monotonic() + 2)
    assert session.missing_usage_calls == 0
    assert session.tool_usage.search_tool.call_count == (search_requests or 0)
    if search_requests is None:
        assert session.tool_usage.actual_total_cost_usd is None
        assert session.tool_usage.search_tool.actual_cost is None
    else:
        assert session.tool_usage.actual_total_cost_usd == pytest.approx(0.0303395)


@pytest.mark.parametrize("status", [401, 403, 404])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_shared_provider_fault_is_batch_terminal_without_retry(status, wrapped):
    from openai import APIStatusError

    from harnyx_commons.llm.provider import LlmProviderError
    from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.example.test/responses"))
    error = APIStatusError("private provider details", response=response, body=None)
    if wrapped:
        wrapped_error = LlmProviderError("provider invocation failed")
        wrapped_error.__cause__ = error
        error = wrapped_error

    async def call():
        raise error

    observed = capture()
    observed.provider = "openai"
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await retry_request(call, observed, monotonic() + 2, solver=False)
    assert caught.value.__cause__ is error
    assert caught.value.failure_class == ("provider_configuration" if status == 404 else "provider_auth")
    assert "private provider details" not in str(caught.value)
    assert observed.attempted == 1


async def test_batch_terminal_fault_retains_known_costs_and_missing_receipt_uncertainty():
    from openai import APIStatusError

    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError

    observed = capture()
    observed.provider, observed.model = "openai", "gpt-6-luna"
    observed.attempted = 1
    observed.account(LlmUsage(prompt_tokens=100, completion_tokens=20, web_search_calls=1))
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.example.test/responses"))

    async def call():
        raise APIStatusError("authentication failed", response=response, body=None)

    with pytest.raises(BatchTerminalGenerationError):
        try:
            await retry_request(call, observed, monotonic() + 2, solver=False)
        finally:
            observed.finish()
    assert observed.session.tool_usage.llm.call_count == 1
    assert observed.session.tool_usage.actual_total_cost_usd is None
    assert observed.session.stage_summaries[0].available_cost_usd > 0


async def test_missing_openai_credentials_are_terminal_and_close_initialization_client(monkeypatch):
    import httpx2
    from openai import OpenAIError

    from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
    from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    client = httpx2.AsyncClient()
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.httpx2.AsyncClient", lambda **kwargs: client)
    runner = GenerationAgentRunner(project_id=None, judge=None)
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await runner.generate_seeds(
            2,
            capture().session,
            monotonic() + 1,
            start=datetime.now(UTC).date(),
            end=datetime.now(UTC).date(),
            previous_seeds=[],
        )
    assert caught.value.failure_class == "provider_configuration"
    assert isinstance(caught.value.__cause__, OpenAIError)
    assert client.is_closed and runner._client is None and runner._http is None


async def test_pricing_uses_each_attempt_start_date_even_if_receipt_crosses_midnight(monkeypatch):
    from datetime import date

    from harnyx_commons.llm.cost_settlement import settled_generation_llm_cost
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation import agent_runner
    from harnyx_commons.miner_task_generation.solver_runner import WORKER_MODEL

    dates = [date(2026, 12, 31), date(2027, 1, 1)]
    current = [dates[0]]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.combine(current[0], datetime.min.time(), tzinfo=UTC)

    monkeypatch.setattr(agent_runner, "datetime", Clock)
    monkeypatch.setattr(agent_runner, "backoff_ms", lambda *args: 0)
    observed = capture()
    observed.model = WORKER_MODEL
    observed.session.effective_date = date(2026, 11, 1)
    usage = LlmUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, web_search_calls=0)

    async def call():
        current[0] = dates[1]  # First request's receipt arrives after midnight.
        observed.account(usage)
        if observed.attempted == 1:
            raise httpx.ConnectError("retry")
        return "complete"

    assert await retry_request(call, observed, monotonic() + 2, solver=True) == "complete"
    expected = sum(
        settled_generation_llm_cost(usage=usage, model=WORKER_MODEL, provider="vertex", pricing_date=value).cost_usd
        for value in dates
    )
    observed.finish()
    assert observed.session.tool_usage.actual_total_cost_usd == pytest.approx(expected)
    assert observed.session.attempted_calls == 2 and observed.session.missing_usage_calls == 0
