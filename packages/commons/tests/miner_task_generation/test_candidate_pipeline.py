"""Protect stopping, solver blindness and terminal reference ownership."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from time import monotonic

import pytest

from harnyx_commons.domain.miner_task import ReferenceAnswer
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.miner_task_generation.contracts import (
    AgentResult,
    BatchTerminalGenerationError,
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
        return FastJudgeAssessment.model_validate(
            {
                "expected_components": [{"component_id": "bird", "is_correct": result.answer == "kākāpō"}],
                "excessive_components": [],
            }
        )

    async def derive_reference(self, draft, verification, session, deadline):
        self.reference_calls += 1
        self.calls.append(("reference", session.cycle, draft.question))
        return TerminalReference(
            reference_answer=ReferenceAnswer(text=draft.answer), factual_correction=self.corrected_reference
        )


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

    result = await run(Runner())
    assert result.status == "finalized"
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
    for role, _, message in runner.calls:
        if role in {"solver", "grounded_solver"}:
            assert "kākāpō" not in message and "source_support" not in message


async def test_five_cycles_keep_exhausted_draft_without_publication():
    runner = FakeRunner(success_cycles={1, 2, 3, 4, 5})
    result = await run(runner)
    assert result.status == "exhausted" and result.finalized is None
    assert len(result.cycles) == 5 and runner.reference_calls == 1
    assert not any(role == "verifier" for role, _, _ in runner.calls)


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
    assert not any(role in {"verifier", "reference"} for role, _, _ in runner.calls)
