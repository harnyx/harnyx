"""Commit complete provider attempts to ADK while streaming diagnostics immediately."""

from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
from google.adk.models.llm_response import LlmResponse as AdkResponse
from google.auth import exceptions as google_auth_errors
from google.genai import types

from harnyx_commons.miner_task_generation.agent_runner import CallCapture
from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError, CandidateSession
from harnyx_commons.miner_task_generation.solver_runner import GenerationGemini, RecordedModels, SearchWorkers, solve

pytestmark = pytest.mark.anyio("asyncio")


def capture():
    return CallCapture(
        CandidateSession(task_id="google", effective_date=datetime.now(UTC).date()),
        "solver",
        "gemini-3.8-flash",
        "vertex",
    )


def chunk(text, *, stop=False, searches=None):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP if stop else None,
                grounding_metadata=types.GroundingMetadata(web_search_queries=searches) if searches else None,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        ),
    )


async def test_partial_failed_stream_is_logged_but_never_committed_into_retried_turn(monkeypatch):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    calls = []

    class Models:
        async def generate_content_stream(self, **kwargs):
            calls.append(kwargs)

            async def events():
                if len(calls) == 1:
                    yield chunk("partial discarded")
                    raise httpx.ReadError("disconnect")
                yield chunk("complete", stop=True, searches=["public query"])

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    contents = [types.Content(role="user", parts=[types.Part(text="exact question")])]
    stream = await proxy.generate_content_stream(
        model="gemini-3.8-flash", contents=contents, config=types.GenerateContentConfig()
    )
    committed = [item async for item in stream]
    assert len(committed) == 1 and committed[0].candidates[0].content.parts[0].text == "complete"
    assert calls[0] == calls[1] and observed.attempted == 2
    assert any("partial discarded" in str(item) for item in observed.responses)
    assert observed.tool_usage.search_tool.call_count == 1


@pytest.mark.parametrize("first_response", ["partial", "usage_only", "empty"])
async def test_stream_without_finish_reason_retries_exact_input_and_retains_only_complete_evidence(
    monkeypatch, first_response
):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    calls, closed = [], []

    class Models:
        async def generate_content_stream(self, **kwargs):
            calls.append(kwargs)
            attempt = len(calls)

            async def events():
                try:
                    if attempt == 1:
                        if first_response == "partial":
                            yield chunk("partial discarded", searches=["abandoned query"])
                        elif first_response == "usage_only":
                            yield types.GenerateContentResponse(usage_metadata=chunk("").usage_metadata)
                    else:
                        yield chunk("complete", stop=True, searches=["public query"])
                finally:
                    closed.append(attempt)

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    stream = await proxy.generate_content_stream(
        model="gemini-3.1-pro-preview",
        contents=[types.Content(role="user", parts=[types.Part(text="exact question")])],
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )
    committed = [item async for item in stream]
    assert len(committed) == 1 and committed[0].candidates[0].content.parts[0].text == "complete"
    assert calls[0] == calls[1] and observed.attempted == 2
    assert closed == [1, 2]
    assert observed.tool_usage.llm.prompt_tokens == (10 if first_response == "empty" else 20)
    assert observed.tool_usage.search_tool.call_count == (2 if first_response == "partial" else 1)
    completed_evidence = [item for item in observed.responses if item["attempt_status"] == "completed"]
    assert "partial discarded" not in str(completed_evidence)
    observed.finish()
    assert observed.session.missing_usage_calls == (1 if first_response == "empty" else 0)


async def test_repeated_stream_without_finish_reason_stops_at_deadline(monkeypatch):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 10)
    closed = []

    class Models:
        async def generate_content_stream(self, **kwargs):
            async def events():
                try:
                    yield chunk("incomplete", searches=["paid query"])
                finally:
                    closed.append(True)

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 0.05)
    with pytest.raises(TimeoutError):
        await proxy.generate_content_stream(
            model="gemini-3.1-pro-preview", contents=[], config=types.GenerateContentConfig()
        )
    assert observed.attempted > 1 and len(closed) == observed.attempted
    assert observed.tool_usage.llm.prompt_tokens == 10 * observed.attempted
    assert observed.tool_usage.search_tool.call_count == observed.attempted


@pytest.mark.parametrize(
    "block_reason", [None, types.BlockedReason.BLOCKED_REASON_UNSPECIFIED, types.BlockedReason.SAFETY]
)
async def test_prompt_feedback_is_not_retried_even_without_specific_block_reason(block_reason):
    class Models:
        async def generate_content_stream(self, **kwargs):
            async def events():
                yield types.GenerateContentResponse(
                    prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
                )

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    with pytest.raises(RuntimeError, match="Provider blocked prompt"):
        await proxy.generate_content_stream(
            model="gemini-3.1-pro-preview", contents=[], config=types.GenerateContentConfig()
        )
    assert observed.attempted == 1


async def test_grounded_worker_retains_full_response_and_search_usage():
    response = chunk("worker answer", stop=True, searches=["first query", "second query"])

    class Models:
        async def generate_content(self, **kwargs):
            return response

    observed = capture()
    worker = SearchWorkers(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    result = await worker.search("exact independent query")
    assert result["status"] == "completed" and result["answer"] == "worker answer"
    assert result["response"]["candidates"][0]["grounding_metadata"]["web_search_queries"] == [
        "first query",
        "second query",
    ]
    assert observed.tool_usage.search_tool.call_count == 2
    assert observed.tool_usage.search_tool.actual_cost > 0
    assert observed.requests[0]["contents"] == "exact independent query"


@pytest.mark.parametrize("search_enabled", [False, True])
async def test_stream_without_search_receipt_reports_unknown_cost_only_when_search_enabled(search_enabled):
    class Models:
        async def generate_content_stream(self, **kwargs):
            async def events():
                yield chunk("complete", stop=True)

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())] if search_enabled else None
    )
    stream = await proxy.generate_content_stream(model="gemini-3.8-flash", contents=[], config=config)
    assert len([item async for item in stream]) == 1
    observed.finish()
    assert observed.session.stage_summaries[0].available_cost_usd > 0
    if search_enabled:
        assert observed.session.tool_usage.search_tool.actual_cost is None
        assert observed.session.tool_usage.actual_total_cost_usd is None
        assert observed.session.tool_usage.actual_cost_by_provider == {}
    else:
        assert observed.session.tool_usage.llm.actual_cost > 0
        assert observed.session.tool_usage.search_tool.actual_cost == 0
        assert observed.session.tool_usage.actual_total_cost_usd == observed.session.tool_usage.llm.actual_cost


async def test_missing_worker_search_receipt_stays_unknown_after_later_known_receipt():
    responses = iter([chunk("first answer", stop=True), chunk("second answer", stop=True, searches=["public query"])])

    class Models:
        async def generate_content(self, **kwargs):
            return next(responses)

    observed = capture()
    worker = SearchWorkers(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    assert (await worker.search("first query"))["status"] == "completed"
    assert observed.tool_usage.search_tool.actual_cost is None
    first_available_cost = observed.available_cost_usd
    assert first_available_cost > 0
    assert (await worker.search("second query"))["status"] == "completed"
    observed.finish()
    assert observed.session.tool_usage.search_tool.call_count == 1
    assert observed.session.tool_usage.search_tool.actual_cost is None
    assert observed.session.tool_usage.actual_total_cost_usd is None
    assert observed.session.tool_usage.actual_cost_by_provider == {}
    assert observed.session.tool_usage.llm.prompt_tokens == 20
    assert observed.session.stage_summaries[0].available_cost_usd > first_available_cost + 0.014


@pytest.mark.parametrize("finish_reason", [types.FinishReason.MAX_TOKENS, types.FinishReason.SAFETY])
async def test_non_stop_response_is_explicit_contract_failure_without_retry(finish_reason):
    class Models:
        async def generate_content_stream(self, **kwargs):
            async def events():
                response = chunk("truncated")
                response.candidates[0].finish_reason = finish_reason
                yield response

            return events()

    observed = capture()
    proxy = RecordedModels(SimpleNamespace(aio=SimpleNamespace(models=Models())), observed, monotonic() + 2)
    with pytest.raises(RuntimeError, match=finish_reason.value):
        await proxy.generate_content_stream(model="gemini-3.8-flash", contents=[], config=types.GenerateContentConfig())
    assert observed.attempted == 1


@pytest.mark.parametrize("shared_fault", [False, True])
async def test_search_worker_failure_escapes_actual_adk_runner_without_a_final_solver_answer(monkeypatch, shared_fault):
    """ADK must not turn a failed research worker into answer text or resume model spending."""
    failure = (
        google_auth_errors.DefaultCredentialsError("missing credentials")
        if shared_fault
        else ValueError("bad response")
    )
    model_calls = []
    closed = []

    class Models:
        async def generate_content(self, **kwargs):
            raise failure

    class Aio:
        models = Models()

        async def aclose(self):
            closed.append("async-client")

    class Client:
        aio = Aio()

        def close(self):
            closed.append("client")

    async def request_worker(self, llm_request, stream=False):
        model_calls.append(llm_request)
        yield AdkResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="search", args={"query": "public question"}))],
            )
        )

    monkeypatch.setattr("harnyx_commons.miner_task_generation.solver_runner.genai.Client", lambda **kwargs: Client())
    monkeypatch.setattr(GenerationGemini, "generate_content_async", request_worker)
    observed = capture()
    error_type = BatchTerminalGenerationError if shared_fault else ValueError
    with pytest.raises(error_type) as caught:
        await solve("solver", "Public question?", observed.session, monotonic() + 2, project_id="test-project")
    assert len(model_calls) == 1
    if shared_fault:
        assert caught.value.__cause__ is failure
    else:
        assert caught.value is failure
    assert "client" in closed and "async-client" in closed
    assert observed.session.stage_summaries[0].outcome == "failed"


async def test_solver_initialization_credentials_failure_is_terminal_and_closes_transport(monkeypatch):
    failure = google_auth_errors.DefaultCredentialsError("missing credentials during initialization")
    transports = []

    def client(**kwargs):
        transports.append(kwargs["http_options"].httpx_async_client)
        raise failure

    monkeypatch.setattr("harnyx_commons.miner_task_generation.solver_runner.genai.Client", client)
    observed = capture()
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await solve("solver", "Public question?", observed.session, monotonic() + 2, project_id="test-project")
    assert caught.value.__cause__ is failure
    assert caught.value.failure_class == "provider_auth"
    assert transports[0].is_closed
    assert observed.session.stage_summaries[0].outcome == "failed"
