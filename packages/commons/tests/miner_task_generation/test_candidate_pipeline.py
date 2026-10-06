"""Protect stopping, solver blindness and terminal reference ownership."""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from time import monotonic

import pytest

from harnyx_commons.domain.miner_task import ReferenceAnswer
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_generation.agent_runner import CallCapture
from harnyx_commons.miner_task_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.miner_task_generation.contracts import (
    AgentResult,
    BatchTerminalGenerationError,
    CandidateResult,
    CandidateSession,
    QuestionDraft,
    ReviewDecision,
    Seed,
    SeedSource,
    SourceSupport,
    StructuredQuestionDraft,
    TerminalReference,
)

pytestmark = pytest.mark.anyio("asyncio")


class FakeRunner:
    def __init__(self, *, success_cycles: set[int] | None = None, reject_verifier: bool = False) -> None:
        self.success_cycles = success_cycles or set()
        self.reject_verifier = reject_verifier
        self.calls: list[tuple[str, int, str]] = []
        self.started: set[str] = set()
        self.both_started = asyncio.Event()
        self.reference_calls = 0
        self.corrected_reference = False
        self.reject_reviews = False
        self.reference_inputs = []

    async def invoke(self, role, message, session, deadline):
        self.calls.append((role, session.cycle, message))
        if role == "author":
            draft = QuestionDraft(
                question="Which bird is named in the source?",
                answer="kākāpō",
                answer_components=["kākāpō"],
                scope="The named source edition",
                source_support=[SourceSupport(url="https://example.org/bird", evidence="kākāpō")],
                revision_note="Current supported selection",
            )
            return AgentResult(answer=draft.model_dump_json())
        if role in {"reviewer", "verifier"}:
            reject = self.reject_reviews if role == "reviewer" else self.reject_verifier
            return AgentResult(
                answer=ReviewDecision(passed=not reject, feedback="repair scope" if reject else "").model_dump_json(
                    by_alias=True
                )
            )
        if role in {"solver", "grounded_solver"}:
            self.started.add(role)
            if len(self.started) == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), 1)
            return AgentResult(answer="kākāpō" if session.cycle in self.success_cycles else "kea")
        return AgentResult(answer="Research report")

    async def assess(self, draft, result, session, deadline):
        self.calls.append(("assessment", session.cycle, result.answer))
        return FastJudgeAssessment.model_validate(
            {
                "expected_components": [{"component_id": "bird", "is_correct": result.answer == "kākāpō"}],
                "excessive_components": [],
            }
        )

    async def derive_reference(self, draft, verification, session, deadline):
        self.reference_calls += 1
        self.calls.append(("reference", session.cycle, draft.question))
        self.reference_inputs.append((draft, verification, list(session.source_support)))
        return TerminalReference(
            reference_answer=ReferenceAnswer(text=draft.answer), factual_correction=self.corrected_reference
        )


class RecordingRunner(FakeRunner):
    """Use the real accounting owner around deterministic provider receipts."""

    def __init__(self, *, analyst_failure=None, **kwargs):
        super().__init__(**kwargs)
        self.captures = []
        self.analyst_failure = analyst_failure
        self.analyst_entered = asyncio.Event()

    @contextmanager
    def capture(self, role, session):
        capture = CallCapture(session, role, "gpt-6-luna", "openai")
        self.captures.append(capture)
        capture.attempted = 1
        capture.account(LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, web_search_calls=0))
        try:
            yield
        finally:
            capture.finish()

    async def invoke(self, role, message, session, deadline):
        with self.capture(role, session):
            if role == "analyst" and self.analyst_failure is not None:
                self.calls.append((role, session.cycle, message))
                self.analyst_entered.set()
                if self.analyst_failure == "exception":
                    raise ValueError("analyst interrupted")
                await asyncio.Event().wait()
            return await super().invoke(role, message, session, deadline)

    async def assess(self, draft, result, session, deadline):
        with self.capture("assessment", session):
            return await super().assess(draft, result, session, deadline)

    async def derive_reference(self, draft, verification, session, deadline):
        with self.capture("reference", session):
            return await super().derive_reference(draft, verification, session, deadline)


def seed() -> Seed:
    return Seed(
        seed_id="bird",
        seed_question="Which bird?",
        reference_answer="kākāpō",
        scope="named edition",
        event_date="2026-10-01",
        supporting_sources=[
            SeedSource(url="https://example.org/bird", evidence="kākāpō", publication_date="2026-10-01")
        ],
    )


async def run(runner: FakeRunner):
    return await CandidatePipeline(runner=runner).run(
        seed(),
        mode="plain_text",
        fast=False,
        session=CandidateSession(task_id="task-1", effective_date=datetime.now(UTC).date()),
        deadline=monotonic() + 10,
    )


async def test_invalid_schema_stops_before_reviewer_and_solver_calls():
    """A forbidden unused reference must not incur downstream research work."""

    class InvalidSchemaRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            result = await super().invoke(role, message, session, deadline)
            if role == "author":
                draft = json.loads(result.answer)
                draft.update(
                    output_schema_json=json.dumps(
                        {"type": "object", "properties": {"unused": {"$ref": "#/$defs/missing"}}}
                    ),
                    structured_answer_json="{}",
                )
                return AgentResult(answer=json.dumps(draft))
            return result

    runner = InvalidSchemaRunner()
    result = await CandidatePipeline(runner=runner).run(
        seed(),
        mode="structured",
        fast=False,
        session=CandidateSession(task_id="invalid-schema", effective_date=datetime.now(UTC).date()),
        deadline=monotonic() + 10,
    )
    assert result.status == "operational_failure" and result.error_type == "ValidationError"
    assert [role for role, _, _ in runner.calls] == ["author"]


async def test_analyst_excludes_abandoned_attempts_and_keeps_workers_once_at_resolvable_indices(monkeypatch, caplog):
    """Attempt failures must stay diagnostic while completed evidence and worker pointers survive."""
    from types import SimpleNamespace

    import httpx
    from google.genai import types

    from harnyx_commons.miner_task_generation.agent_runner import CallCapture
    from harnyx_commons.miner_task_generation.solver_runner import RecordedModels, SearchWorkers

    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *args: 0)
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")
    captures = []

    class Models:
        def __init__(self):
            self.stream_calls = 0

        async def generate_content_stream(self, **kwargs):
            self.stream_calls += 1

            async def chunks():
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part(
                                        text="abandoned-wrong-answer" if self.stream_calls == 1 else "completed-answer"
                                    )
                                ],
                            ),
                            finish_reason=None if self.stream_calls == 1 else types.FinishReason.STOP,
                        )
                    ],
                    usage_metadata=types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=10, candidates_token_count=5, total_token_count=15
                    ),
                )
                if self.stream_calls == 1:
                    raise httpx.ReadError("stream disconnected")

            return chunks()

        async def generate_content(self, **kwargs):
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(text="distinctive-worker-thought", thought=True),
                                types.Part(text="kea"),
                            ],
                        ),
                        finish_reason=types.FinishReason.STOP,
                        grounding_metadata=types.GroundingMetadata(web_search_queries=["bird query"]),
                    )
                ],
                usage_metadata=types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                ),
            )

    class Runner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            if role == "solver":
                self.started.add(role)
                if len(self.started) == 2:
                    self.both_started.set()
                capture = CallCapture(session, role, "gemini-3.8-flash", "vertex")
                captures.append(capture)
                client = SimpleNamespace(aio=SimpleNamespace(models=Models()))
                workers = SearchWorkers(client, capture, deadline)
                await workers.search("first query")
                stream = await RecordedModels(client, capture, deadline).generate_content_stream(
                    model="gemini-3.8-flash", contents=[], config=types.GenerateContentConfig()
                )
                assert [part async for part in stream][0].candidates[0].content.parts[0].text == "completed-answer"
                await workers.search("second query")
                capture.finish()
                return AgentResult(
                    answer="kea",
                    provider_responses=capture.responses,
                    search_worker_responses=capture.workers,
                )
            if role == "analyst":
                packet = json.loads(message)
                assert message.count("distinctive-worker-thought") == 2  # One per physical worker.
                assert "abandoned-wrong-answer" not in message and "completed-answer" in message
                envelopes = packet["solver_results"]["solver"]["provider_responses"]
                assert envelopes[1] == {"provider": "vertex", "attempt_status": "failed", "response": {}}
                records = packet["solver_results"]["solver"]["search_worker_responses"]
                assert [record["provider_response_index"] for record in records] == [0, 3]
                assert [record["query"] for record in records] == ["first query", "second query"]
                assert [record["order"] for record in records] == [1, 2]
                references = [record["response_reference"] for record in records]
                references.extend(location["reference"] for location in packet["trace_locations"])
                for reference in references:
                    value = packet
                    for part in reference.split("#/")[1].split("/"):
                        value = value[int(part)] if isinstance(value, list) else value[part]
                    assert value
                for record in records:
                    response = packet["solver_results"]["solver"]["provider_responses"][
                        record["provider_response_index"]
                    ]["response"]
                    assert response["candidates"][0]["grounding_metadata"]["web_search_queries"] == ["bird query"]
            return await super().invoke(role, message, session, deadline)

    runner = Runner(success_cycles={1})
    result = await run(runner)
    assert result.status == "finalized"
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1]
    assert "abandoned-wrong-answer" in str(captures[0].responses)
    assert captures[0].usage_records == captures[0].attempted == 4
    assert captures[0].tool_usage.llm.prompt_tokens == 40
    assert any("abandoned-wrong-answer" in str(getattr(record, "data", {})) for record in caplog.records)


async def test_parallel_blind_solvers_and_reference_after_verified_miss():
    runner = FakeRunner(success_cycles={1})
    result = await run(runner)
    assert result.status == "finalized" and len(result.cycles) == 2
    roles = [role for role, _, _ in runner.calls]
    assert roles[-2:] == ["verifier", "reference"]
    assert runner.reference_calls == 1
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1]
    assert result.cycles[-1].analysis is None
    for role, _, message in runner.calls:
        if role in {"solver", "grounded_solver"}:
            assert "kākāpō" not in message and "source_support" not in message


async def test_five_cycles_keep_exhausted_draft_without_publication():
    runner = FakeRunner(success_cycles={1, 2, 3, 4, 5})
    result = await run(runner)
    assert result.status == "exhausted" and result.finalized is None
    assert len(result.cycles) == 5 and runner.reference_calls == 1
    assert not any(role == "verifier" for role, _, _ in runner.calls)
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1, 2, 3, 4]
    assert result.cycles[-1].analysis is None


@pytest.mark.parametrize("both_miss", [False, True])
async def test_excessive_claims_trigger_verification_only_when_both_solver_f1_scores_are_below_one(both_miss):
    """A correct required answer with excess must use F1, not recall, for stopping."""

    class ExcessiveRunner(FakeRunner):
        async def assess(self, draft, result, session, deadline):
            assessment_index = sum(role == "assessed" for role, _, _ in self.calls)
            self.calls.append(("assessed", session.cycle, ""))
            return FastJudgeAssessment(
                expected_components=[{"component_id": "bird", "is_correct": True}],
                excessive_components=[{"component_id": "wrong-extra"}]
                if both_miss or assessment_index % 2 == 0
                else [],
            )

    runner = ExcessiveRunner(success_cycles={1, 2, 3, 4, 5})
    result = await run(runner)
    if both_miss:
        assert result.status == "finalized" and len(result.cycles) == 1
        assert all(score < 1 for score in result.cycles[0].f1_scores.values())
        assert result.cycles[0].verification is not None
    else:
        assert result.status == "exhausted" and len(result.cycles) == 5
        assert all(sorted(cycle.f1_scores.values()) == [pytest.approx(2 / 3), 1] for cycle in result.cycles)
        assert not any(role == "verifier" for role, _, _ in runner.calls)


async def test_reviewer_rejection_is_not_a_stop_gate():
    runner = FakeRunner()
    runner.reject_reviews = True
    result = await run(runner)
    assert result.status == "finalized"
    assert sum(role == "author" for role, _, _ in runner.calls) == 3


async def test_terminal_reference_configuration_fault_escapes_candidate_wrapper():
    failure = BatchTerminalGenerationError("provider_authentication", "denied", stage="reference")

    class FailedReferenceRunner(FakeRunner):
        async def derive_reference(self, draft, verification, session, deadline):
            raise failure

    with pytest.raises(BatchTerminalGenerationError) as caught:
        await run(FailedReferenceRunner())
    assert caught.value is failure


async def test_verifier_rejections_exhaust_and_terminal_correction_is_unresolved():
    rejected = await run(FakeRunner(reject_verifier=True))
    assert rejected.status == "exhausted" and len(rejected.cycles) == 5
    runner = FakeRunner()
    runner.corrected_reference = True
    corrected = await run(runner)
    assert corrected.status == "unresolved_reference" and corrected.finalized is None


async def test_formatting_only_failure_does_not_trigger_verifier_and_schema_is_public():
    class StructuredRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            result = await super().invoke(role, message, session, deadline)
            if role == "author":
                draft = QuestionDraft.model_validate_json(result.answer)
                return AgentResult(
                    answer=StructuredQuestionDraft(
                        **draft.model_dump(),
                        output_schema_json=json.dumps(
                            {
                                "type": "object",
                                "properties": {"bird": {"type": "string"}},
                                "required": ["bird"],
                                "additionalProperties": False,
                            }
                        ),
                        structured_answer_json='{"bird":"kākāpō"}',
                    ).model_dump_json()
                )
            return result

    runner = StructuredRunner(success_cycles={1, 2, 3, 4, 5})
    result = await CandidatePipeline(runner=runner).run(
        seed(),
        mode="structured",
        fast=True,
        session=CandidateSession(task_id="structured", effective_date=datetime.now(UTC).date()),
        deadline=monotonic() + 10,
    )
    assert result.status == "exhausted"
    assert all(not cycle.format_results["solver"]["valid"] for cycle in result.cycles)
    assert not any(role == "verifier" for role, _, _ in runner.calls)
    public_messages = [
        json.loads(message) for role, _, message in runner.calls if role in {"solver", "grounded_solver"}
    ]
    assert all(set(packet) == {"question", "output_schema"} for packet in public_messages)
    assert all("kākāpō" not in json.dumps(packet) for packet in public_messages)


async def test_verifier_rejection_feedback_reaches_next_author_without_solver_traces():
    runner = FakeRunner(reject_verifier=True)
    result = await run(runner)
    assert result.status == "exhausted"
    second_input = next(json.loads(message) for role, cycle, message in runner.calls if role == "author" and cycle == 2)
    assert second_input["shared_context"]["answer_verification"] == {"pass": False, "feedback": "repair scope"}
    assert second_input["shared_context"]["solver_analysis"] == result.cycles[0].analysis == "Research report"
    roles = [role for role, cycle, _ in runner.calls if cycle == 1]
    assert roles.index("verifier") < roles.index("analyst")
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1, 2, 3, 4]
    for role, _, message in runner.calls:
        if role == "verifier":
            packet = json.loads(message)
            assert "solver_results" not in packet and "search_worker_responses" not in packet
            assert packet["author_trace"][0]["summary_status"] == "missing"


async def test_completed_peer_retained_when_other_solver_fails():
    class PartialRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            if role == "grounded_solver":
                raise ValueError("provider contract failure")
            if role == "solver":
                return AgentResult(answer="completed peer")
            return await super().invoke(role, message, session, deadline)

    runner = PartialRunner()
    result = await run(runner)
    assert result.status == "operational_failure"
    assert result.cycles[0].solver_results["solver"].answer == "completed peer"
    assert not any(role in {"analyst", "verifier", "reference"} for role, _, _ in runner.calls)


async def test_judge_failure_settles_peer_and_retains_completed_assessment():
    class PartialJudgeRunner(FakeRunner):
        async def assess(self, draft, answer, session, deadline):
            self.started.add("judge")
            if not hasattr(self, "first_judge_failed"):
                self.first_judge_failed = True
                raise ValueError("invalid judge contract")
            await asyncio.sleep(0.01)
            self.peer_judge_finished = True
            return await super().assess(draft, answer, session, deadline)

    runner = PartialJudgeRunner()
    result = await run(runner)
    assert result.status == "operational_failure" and runner.peer_judge_finished
    assert len(result.cycles[0].assessments) == len(result.cycles[0].f1_scores) == 1
    assert not any(role in {"analyst", "verifier", "reference"} for role, _, _ in runner.calls)
    assert result.cycles[0].analysis is None


async def test_terminal_cycle_skips_analyst_and_accounts_only_executed_work(caplog):
    """An immediately approved task must not pay for unused revision research."""
    caplog.set_level("DEBUG", logger="harnyx_commons.miner_task_generation")
    runner = RecordingRunner()
    result = await run(runner)
    assert result.status == "finalized"
    assert [role for role, _, _ in runner.calls] == [
        "author",
        "reviewer",
        "solver",
        "grounded_solver",
        "assessment",
        "assessment",
        "verifier",
        "reference",
    ]
    assert result.cycles[0].analysis is None
    assert [stage.stage for stage in result.stage_summaries].count("analyst") == 0
    assert result.attempted_calls == result.tool_usage.llm.call_count == len(runner.captures) == 8
    assert result.missing_usage_calls == 0
    assert result.tool_usage.llm.total_tokens == 15 * len(runner.captures)
    assert result.tool_usage.actual_total_cost_usd == pytest.approx(
        sum(capture.tool_usage.actual_total_cost_usd for capture in runner.captures)
    )
    draft, verification, sources = runner.reference_inputs[0]
    assert draft == result.cycles[0].draft and verification == result.cycles[0].verification
    assert verification.passed and runner.reference_calls == 1
    assert sources == [*seed().supporting_sources, *draft.source_support]
    records = [
        record.data["record"] for record in caplog.records if record.message == "task_generation.cycle.completed"
    ]
    assert len(records) == 1 and records[0]["analysis"] is None
    assert set(records[0]["solver_results"]) == {"solver", "grounded_solver"}


async def test_one_perfect_solver_continues_with_analysis_after_both_assessments():
    """One scored miss cannot stop a task or omit the next author's research."""

    class OnePerfectRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            result = await super().invoke(role, message, session, deadline)
            if role == "solver" and session.cycle == 1:
                result.answer = "kea"
            return result

    runner = OnePerfectRunner(success_cycles={1})
    result = await run(runner)
    assert result.status == "finalized" and len(result.cycles) == 2
    assert sorted(result.cycles[0].f1_scores.values()) == [0, 1]
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1]
    roles = [role for role, cycle, _ in runner.calls if cycle == 1]
    assert roles[-3:] == ["assessment", "assessment", "analyst"]
    assert "verifier" not in roles
    context = next(
        json.loads(message)["shared_context"]
        for role, cycle, message in runner.calls
        if role == "author" and cycle == 2
    )
    assert context["solver_analysis"] == result.cycles[0].analysis == "Research report"


@pytest.mark.parametrize("outcome", ["approval", "perfect", "rejection"])
async def test_fifth_cycle_skips_analyst_and_preserves_stop_outcome(outcome, caplog):
    """Cycle five still verifies paired misses and gives approval precedence."""

    class FifthCycleRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            if role == "verifier":
                self.reject_verifier = outcome == "rejection" or session.cycle < 5
            return await super().invoke(role, message, session, deadline)

    caplog.set_level("DEBUG", logger="harnyx_commons.miner_task_generation")
    runner = FifthCycleRunner(success_cycles=set(range(1, 6)) if outcome == "perfect" else set())
    result = await run(runner)
    assert len(result.cycles) == 5
    assert result.status == ("finalized" if outcome == "approval" else "exhausted")
    assert (result.finalized is not None) == (outcome == "approval")
    assert [cycle for role, cycle, _ in runner.calls if role == "analyst"] == [1, 2, 3, 4]
    assert result.cycles[-1].analysis is None and runner.reference_calls == 1
    assert [cycle for role, cycle, _ in runner.calls if role == "verifier"] == (
        [] if outcome == "perfect" else [1, 2, 3, 4, 5]
    )
    records = [
        record.data["record"] for record in caplog.records if record.message == "task_generation.cycle.completed"
    ]
    assert [record["cycle"] for record in records] == [1, 2, 3, 4, 5]
    assert all(record["analysis"] == "Research report" for record in records[:-1])
    assert records[-1]["analysis"] is None


async def test_assessments_still_run_in_parallel():
    """Moving assessments must not serialize the two independent judge calls."""
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class GatedRunner(FakeRunner):
        async def assess(self, draft, result, session, deadline):
            self.started.add(result.answer)
            self.assessment_count = getattr(self, "assessment_count", 0) + 1
            if self.assessment_count == 2:
                both_entered.set()
            await release.wait()
            return await super().assess(draft, result, session, deadline)

    task = asyncio.create_task(run(GatedRunner()))
    try:
        await asyncio.wait_for(both_entered.wait(), 1)
        assert not task.done()
        release.set()
        assert (await task).status == "finalized"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_verifier_failure_retains_scores_without_analyst_or_reference():
    """Unfinished verification cannot become a miss approval or a revision."""

    class FailedVerifierRunner(FakeRunner):
        async def invoke(self, role, message, session, deadline):
            if role == "verifier":
                self.calls.append((role, session.cycle, message))
                raise ValueError("invalid verification")
            return await super().invoke(role, message, session, deadline)

    runner = FailedVerifierRunner()
    result = await run(runner)
    assert result.status == "operational_failure" and result.error_type == "ValueError"
    assert set(result.cycles[0].assessments) == {"solver", "grounded_solver"}
    assert set(result.cycles[0].f1_scores) == {"solver", "grounded_solver"}
    assert result.cycles[0].analysis is None and result.cycles[0].verification is None
    assert not any(role in {"analyst", "reference"} for role, _, _ in runner.calls)
    assert sum(role == "author" for role, _, _ in runner.calls) == 1


@pytest.mark.parametrize("failure", ["exception", "deadline", "cancellation"])
async def test_failed_continuing_analyst_retains_evidence_and_real_accounting(failure):
    """Failed paid revision work keeps settled scores, feedback and real receipts."""
    runner = RecordingRunner(reject_verifier=True, analyst_failure=failure)
    result = CandidateResult(seed=seed(), mode="plain_text", status="operational_failure")
    task = asyncio.create_task(
        CandidatePipeline(runner=runner).run(
            seed(),
            mode="plain_text",
            fast=False,
            session=CandidateSession(task_id="analyst-failure", effective_date=datetime.now(UTC).date()),
            deadline=monotonic() + (0.2 if failure == "deadline" else 10),
            result=result,
        )
    )
    try:
        await asyncio.wait_for(runner.analyst_entered.wait(), 1)
        if failure == "cancellation":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task is result
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert result.status == "operational_failure"
    assert (
        result.error_type
        == {"exception": "ValueError", "deadline": "TimeoutError", "cancellation": "CancelledError"}[failure]
    )
    cycle = result.cycles[0]
    assert set(cycle.solver_results) == set(cycle.assessments) == set(cycle.f1_scores) == {"solver", "grounded_solver"}
    assert cycle.verification is not None and not cycle.verification.passed
    assert cycle.analysis is None
    assert [role for role, _, _ in runner.calls][-2:] == ["verifier", "analyst"]
    assert runner.reference_calls == 0 and sum(role == "author" for role, _, _ in runner.calls) == 1
    analyst = next(stage for stage in result.stage_summaries if stage.stage == "analyst")
    captured = runner.captures[-1]
    assert analyst.outcome == "failed" and analyst.attempted_calls == captured.attempted == 1
    assert analyst.tool_usage == captured.tool_usage
    assert analyst.available_cost_usd == captured.available_cost_usd > 0
    assert result.attempted_calls == result.tool_usage.llm.call_count == len(runner.captures)
    assert result.missing_usage_calls == 0
