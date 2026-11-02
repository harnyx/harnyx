"""Retry provider requests without replaying completed agent turns or peer work."""

import json
import logging
from datetime import UTC, datetime
from time import monotonic
from unittest.mock import AsyncMock

import httpx
import pytest
from google.auth import exceptions as google_auth_errors
from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import draft

from harnyx_commons.domain.miner_task import ReferenceAnswer
from harnyx_commons.llm.schema import LlmChoice, LlmChoiceMessage, LlmMessageContentPart, LlmResponse, LlmUsage
from harnyx_commons.miner_task_generation.agent_runner import (
    JUDGE_MODEL,
    CallCapture,
    GenerationAgentRunner,
    retry_after,
    retry_request,
)
from harnyx_commons.miner_task_generation.contracts import CandidateSession, TerminalReference

pytestmark = pytest.mark.anyio("asyncio")


def capture():
    return CallCapture(
        CandidateSession(task_id="retry", effective_date=datetime.now(UTC).date()), "solver", "test", "vertex"
    )


@pytest.mark.parametrize("structured", [False, True])
async def test_report_grader_receives_frozen_key_and_full_report_with_separate_usage(
    structured,
):
    """Report grading must never substitute the report itself for the tested ground truth."""
    requests = []
    assessment = {"expected_components": [{"component_id": "item", "is_correct": True}], "excessive_components": []}

    class Judge:
        async def invoke(self, request):
            requests.append(request)
            return LlmResponse(
                id="report-assessment",
                choices=(
                    LlmChoice(
                        index=0,
                        message=LlmChoiceMessage(
                            role="assistant", content=(LlmMessageContentPart(type="text", text=json.dumps(assessment)),)
                        ),
                    ),
                ),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, web_search_calls=0),
            )

    report = "The archive identifies item [[1]]. Here is the full supporting research."
    rendered = ReferenceAnswer(text='{"item":"item"}' if structured else report, note=report if structured else None)
    session = CandidateSession(task_id="report", effective_date=datetime.now(UTC).date())
    grade = await GenerationAgentRunner(solve=AsyncMock(), judge=Judge()).assess_reference(
        draft(structured), TerminalReference(reference_answer=rendered), session, monotonic() + 5
    )
    assert grade.model_dump(mode="json") == assessment and len(requests) == 1
    request = requests[0]
    payload = json.loads(request.messages[1].content[0].text.removeprefix("Payload:\n"))
    assert payload["reference_answer"] == "item"
    assert payload["miner_answer"] == rendered.text and payload["miner_note"] == rendered.note
    assert request.model == JUDGE_MODEL and request.provider == "vertex" and request.max_output_tokens is None
    assert session.attempted_calls == 1
    assert session.stage_summaries[0].stage == "reference_consistency"
    assert session.stage_summaries[0].available_cost_usd > 0


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


async def test_pricing_uses_each_attempt_start_date_even_if_receipt_crosses_midnight(monkeypatch):
    from datetime import date

    from harnyx_commons.llm.cost_settlement import settled_generation_llm_cost
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation import agent_runner

    dates = [date(2026, 12, 31), date(2027, 1, 1)]
    current = [dates[0]]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.combine(current[0], datetime.min.time(), tzinfo=UTC)

    monkeypatch.setattr(agent_runner, "datetime", Clock)
    monkeypatch.setattr(agent_runner, "backoff_ms", lambda *args: 0)
    observed = capture()
    observed.model = JUDGE_MODEL
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
        settled_generation_llm_cost(usage=usage, model=JUDGE_MODEL, provider="vertex", pricing_date=value).cost_usd
        for value in dates
    )
    observed.finish()
    assert observed.session.tool_usage.actual_total_cost_usd == pytest.approx(expected)
    assert observed.session.attempted_calls == 2 and observed.session.missing_usage_calls == 0


class AssessmentSequence:
    """The provider port receives real feedback work orders; no live model calls."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    async def invoke(self, request):
        self.requests.append(request)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return LlmResponse(
            id=f"assessment-{len(self.requests)}",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(role="assistant", content=(LlmMessageContentPart.input_text(output),)),
                ),
            ),
            usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, web_search_calls=0),
        )


VALID_ASSESSMENT = json.dumps(
    {"expected_components": [{"component_id": "item", "is_correct": True}], "excessive_components": []}
)


async def assess_sequence(judge, session, *, seconds=5):
    return await GenerationAgentRunner(solve=AsyncMock(), judge=judge).assess_reference(
        draft(),
        TerminalReference(reference_answer=ReferenceAnswer(text="item [[1]]")),
        session,
        monotonic() + seconds,
    )


async def test_assessment_repairs_five_strict_parse_failures_and_preserves_context_usage_and_limit(caplog):
    """Feedback must correct only the judge, preserving original input and every billed response."""
    failures = ["not JSON", "{}", f"```json\n{VALID_ASSESSMENT}\n```", "[]", '{"expected_components": 1}']
    judge = AssessmentSequence([*failures, VALID_ASSESSMENT])
    session = capture().session
    caplog.set_level(logging.WARNING, logger="harnyx_commons.llm.calls")
    grade = await assess_sequence(judge, session)
    assert grade.expected_components[0].is_correct
    assert len(judge.requests) == session.attempted_calls == session.tool_usage.llm.call_count == 6
    assert session.tool_usage.llm.total_tokens == 90 and session.missing_usage_calls == 0
    assert len(session.stage_summaries) == 1
    for number, request in enumerate(judge.requests[1:], 1):
        assert request.messages[:2] == judge.requests[0].messages
        assert len(request.messages) == 4
        assert request.messages[-2].content[0].text == failures[number - 1]
        assert "Validation/parsing error:" in request.messages[-1].content[0].text
        assert request.output_schema == judge.requests[0].output_schema
        assert request.retry_policy.attempts == 1
    feedback = [r.data for r in caplog.records if r.msg == "task_generation.output_model_feedback"]
    assert [v["retry"] for v in feedback] == [1, 2, 3, 4, 5]
    assert all(v["feedback_sent"] and not v["exhausted"] for v in feedback)


async def test_assessment_parse_exhaustion_retains_original_error_and_all_usage(caplog):
    from pydantic import ValidationError

    judge = AssessmentSequence(["{}"] * 6)
    session = capture().session
    caplog.set_level(logging.WARNING, logger="harnyx_commons.llm.calls")
    with pytest.raises(ValidationError):
        await assess_sequence(judge, session)
    assert len(judge.requests) == session.attempted_calls == session.tool_usage.llm.call_count == 6
    assert session.stage_summaries[0].outcome == "failed"
    feedback = [r.data for r in caplog.records if r.msg == "task_generation.output_model_feedback"]
    assert len(feedback) == 6 and feedback[-1]["exhausted"] and not feedback[-1]["feedback_sent"]


async def test_assessment_recovery_does_not_refresh_transient_retry_allowance(monkeypatch):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    judge = AssessmentSequence([httpx.ConnectError("first"), "{}", httpx.ConnectError("second"), VALID_ASSESSMENT])
    session = capture().session
    with pytest.raises(httpx.ConnectError, match="second"):
        await assess_sequence(judge, session)
    assert len(judge.requests) == session.attempted_calls == 3
    assert len(judge.requests[-1].messages) == 4
    assert session.tool_usage.llm.call_count == 1 and session.missing_usage_calls == 2


async def test_assessment_transient_fault_does_not_refresh_parse_recovery_allowance(monkeypatch):
    from pydantic import ValidationError

    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    judge = AssessmentSequence([*["{}"] * 3, httpx.ConnectError("transient"), *["{}"] * 3])
    session = capture().session
    with pytest.raises(ValidationError):
        await assess_sequence(judge, session)
    assert len(judge.requests) == 7 and session.tool_usage.llm.call_count == 6


async def test_assessment_recovery_waits_share_original_deadline(monkeypatch):
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 60_000)
    judge = AssessmentSequence(["{}", httpx.ConnectError("temporary"), VALID_ASSESSMENT])
    session = capture().session
    with pytest.raises(TimeoutError):
        await assess_sequence(judge, session, seconds=0.03)
    assert len(judge.requests) == 2
    assert session.tool_usage.llm.call_count == 1 and session.missing_usage_calls == 1


async def test_assessment_cancellation_during_recovery_preserves_completed_receipt():
    import asyncio

    waiting = asyncio.Event()
    judge = AssessmentSequence(["{}"])
    original = judge.invoke

    async def invoke(request):
        if judge.requests:
            waiting.set()
            await asyncio.Event().wait()
        return await original(request)

    judge.invoke = invoke
    session = capture().session
    task = asyncio.create_task(assess_sequence(judge, session))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.attempted_calls == 2 and session.tool_usage.llm.call_count == 1
    assert session.missing_usage_calls == 1


async def test_pipeline_recovers_both_assessments_and_reference_without_replaying_solvers():
    """Each output owner gets its own correction streak while completed research stays frozen."""
    from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import FakeAuthor, run

    from harnyx_commons.miner_task_generation.contracts import AgentResult

    miss = json.dumps(
        {"expected_components": [{"component_id": "item", "is_correct": False}], "excessive_components": []}
    )
    judge = AssessmentSequence(["{}", miss, "{}", miss, *["{}"] * 5, VALID_ASSESSMENT])
    solve = AsyncMock(return_value=AgentResult(answer="solver result"))
    author = FakeAuthor()
    result = await run(author, GenerationAgentRunner(solve=solve, judge=judge))
    assert result.finalized is not None and len(result.cycles) == 1
    assert solve.await_count == 2 and len(author.feedback) == len(author.decisions) == 1
    assert set(result.cycles[0].f1_scores.values()) == {0.0}
    assert len(judge.requests) == 10
    assert len([r for r in judge.requests if r.use_case.endswith("reference_consistency")]) == 6
