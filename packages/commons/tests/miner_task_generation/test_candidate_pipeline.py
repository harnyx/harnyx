"""Prevent unsupported, untested or replayed challenges from authorizing publication."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from time import monotonic

import pytest

from harnyx_commons.domain.miner_task import AnswerCitation, CitationExcerpt, ReferenceAnswer
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_generation.candidate_pipeline import CandidatePipeline, settle_pair
from harnyx_commons.miner_task_generation.contracts import (
    AgentResult,
    AuthorDeclaration,
    AuthorOutcome,
    AuthorReference,
    CandidateResult,
    CandidateSession,
    GenerationCheckpoint,
    QuestionDraft,
    StructuredQuestionDraft,
    TerminalReference,
)

pytestmark = pytest.mark.anyio("asyncio")


def draft(structured=False):
    fields = dict(
        question="Which item is recorded?",
        answer="item",
        answer_components=["item"],
        scope="named edition",
        source_support=[{"url": "https://example.org/report", "evidence": "item"}],
        revision_note="Verified deciding passage",
    )
    if structured:
        return StructuredQuestionDraft(
            **fields,
            output_schema_json=json.dumps(
                {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                    "additionalProperties": False,
                }
            ),
            structured_answer_json='{"item":"item"}',
        )
    return QuestionDraft(**fields)


class FakeRunner:
    def __init__(self, *, correct=1, expected=3, failure=None):
        self.correct, self.expected, self.failure = correct, expected, failure
        self.calls = []
        self.solvers = set()
        self.assessors = set()
        self.solver_pair = asyncio.Event()
        self.assessment_pair = asyncio.Event()
        self.reference_calls = []
        self.reference_assessments = []

    async def invoke(self, role, message, session, deadline):
        self.calls.append((role, message))
        self.solvers.add(role)
        if len(self.solvers) == 2:
            self.solver_pair.set()
        await asyncio.wait_for(self.solver_pair.wait(), 1)
        if self.failure:
            raise self.failure
        return AgentResult(
            answer=role, provider_responses=[{"provider": "vertex", "response": {"text": "full evidence"}}]
        )

    async def assess(self, draft, result, session, deadline):
        self.assessors.add(result.answer)
        if len(self.assessors) == 2:
            self.assessment_pair.set()
        await asyncio.wait_for(self.assessment_pair.wait(), 1)
        return FastJudgeAssessment.model_validate(
            {
                "expected_components": [
                    {"component_id": f"fact-{i}", "is_correct": i < self.correct} for i in range(self.expected)
                ],
                "excessive_components": [],
            }
        )

    async def aclose(self):
        pass

    async def assess_reference(self, draft, reference, session, deadline):
        self.reference_calls.append((draft.model_copy(deep=True), reference))
        if self.reference_assessments:
            return self.reference_assessments.pop(0)
        return FastJudgeAssessment.model_validate(
            {"expected_components": [{"component_id": "item", "is_correct": True}], "excessive_components": []}
        )


class FakeAuthor:
    def __init__(self, *, structured=False, rounds=1, duplicate=False, changed=False, status="completed", error=None):
        self.structured, self.rounds, self.duplicate, self.changed = structured, rounds, duplicate, changed
        self.status, self.error = status, error
        self.feedback = []
        self.sessions = []
        self.decisions = []

    async def run(self, *, session, mode, deadline, challenge, finish, checkpoint):
        self.sessions.append((session.task_id, mode, deadline))
        state = GenerationCheckpoint(
            candidate_id=session.task_id, prompt_sha256="hash", thread_id="thread", turn_id="turn", status="running"
        )
        await checkpoint(state)
        current = draft(mode == "structured")
        for number in range(1, self.rounds + 1):
            if number > 1:
                current.question = f"Revised question for round {number}"
            if self.duplicate:
                values = await asyncio.gather(challenge(current, number), challenge(current, number))
                assert values[0] is values[1]
                self.feedback.append(values[0])
            else:
                self.feedback.append(await challenge(current, number))
        if self.error:
            raise self.error
        if self.changed:
            current.answer = "changed"
            current.question = "Untested replacement question"
            current.source_support.clear()
            self.feedback[-1].record.draft.answer = "changed feedback object"
        declaration = AuthorDeclaration(
            status=self.status,
            reference=author_reference(mode == "structured") if self.status == "completed" else None,
            reconciliation="Constructed citations for the frozen key",
        )
        self.decisions.append(await finish(declaration))
        return AuthorOutcome(declaration, True, session.generation_checkpoint or state)

    async def aclose(self):
        pass


def author_reference(structured=False):
    return AuthorReference(
        answer_text=None if structured else "item [[1]]",
        note="The source identifies item [[1]]." if structured else None,
        citations=[
            AnswerCitation(url="https://example.org/report", excerpts=(CitationExcerpt(start=0, end=4, text="item"),))
        ],
    )


async def reference(draft, packet):
    return TerminalReference(reference_answer=ReferenceAnswer(text=draft.answer))


async def run(author=None, runner=None, *, render=reference, **kwargs):
    author, runner = author or FakeAuthor(), runner or FakeRunner()
    mode = "structured" if author.structured else "plain_text"
    session = CandidateSession(task_id="candidate", effective_date=datetime.now(UTC).date())
    result = CandidateResult(mode=mode, status="operational_failure")
    return await CandidatePipeline(author=author, runner=runner, render_reference=render).run(
        mode=mode, fast=True, session=session, result=result, deadline=monotonic() + 3, **kwargs
    )


@pytest.mark.parametrize("structured", [False, True])
async def test_half_f1_boundary_and_public_solver_blindness(structured):
    author, runner = FakeAuthor(structured=structured), FakeRunner()
    result = await run(author, runner)
    assert result.finalized is not None and len(result.cycles) == 1
    assert set(result.cycles[0].f1_scores.values()) == {0.5}
    assert (
        author.feedback[0].record.solver_results["solver"].provider_responses[0]["response"]["text"] == "full evidence"
    )
    assert runner.assessment_pair.is_set()
    assert len(result.reference_consistency_assessments) == len(runner.reference_calls) == 1
    for _, message in runner.calls:
        public = json.loads(message) if structured else {"question": message}
        assert set(public) == ({"question", "output_schema"} if structured else {"question"})
        assert "Verified deciding passage" not in message


async def test_concurrent_duplicate_round_runs_paid_pair_only_once():
    author, runner = FakeAuthor(duplicate=True), FakeRunner()
    result = await run(author, runner)
    assert result.finalized and len(runner.calls) == 2 and len(result.cycles) == 1


@pytest.mark.parametrize(
    "status,changed,expected",
    [
        ("exhausted", False, "exhausted"),
        ("error", False, "operational_failure"),
    ],
)
async def test_terminal_outcomes_do_not_publish_unqualified_drafts(status, changed, expected):
    result = await run(FakeAuthor(status=status, changed=changed))
    assert result.finalized is None and result.status == expected
    assert result.author_declaration is not None


async def test_one_solver_succeeding_prevents_publication():
    result = await run(runner=FakeRunner(expected=2))
    assert result.finalized is None and result.error_type == "not_paired_miss"


async def test_provider_error_retains_partial_cycle_without_zero_grade():
    result = await run(runner=FakeRunner(failure=RuntimeError("provider failed")))
    assert result.status == "operational_failure" and result.finalized is None
    assert result.cycles[0].f1_scores == {}


async def test_sixth_round_is_a_contract_error_without_sixth_pair():
    runner = FakeRunner(expected=2)
    result = await run(FakeAuthor(rounds=6), runner)
    assert result.error_type == "ValueError" and len(result.cycles) == 5 and len(runner.calls) == 10


async def test_five_round_exhaustion_retains_every_challenge_without_rerolls():
    result = await run(FakeAuthor(rounds=5, status="exhausted"), FakeRunner(expected=2))
    assert result.status == "exhausted" and len(result.cycles) == 5 and result.finalized is None


async def test_paired_miss_stops_development_before_another_paid_round():
    runner = FakeRunner()
    result = await run(FakeAuthor(rounds=2), runner)
    assert result.finalized is None and result.error_type == "ValueError"
    assert len(runner.calls) == 2 and len(result.cycles) == 1


async def test_failed_pair_cancels_running_peer_and_retains_completed_result():
    """A provider fault must stop paid peer work instead of waiting for its deadline."""
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked():
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failure():
        await waiting.wait()
        raise RuntimeError("provider failed")

    async def success():
        return "retained"

    completed = {}
    tasks = {
        "blocked": asyncio.create_task(blocked()),
        "failed": asyncio.create_task(failure()),
        "success": asyncio.create_task(success()),
    }
    with pytest.raises(RuntimeError, match="provider failed"):
        await asyncio.wait_for(settle_pair(tasks, completed), 1)
    assert cancelled.is_set() and completed == {"success": "retained"}
    assert all(task.done() for task in tasks.values())


@pytest.mark.parametrize("structured", [False, True])
async def test_finalization_uses_private_last_tested_snapshot_not_author_mutations(structured):
    """A terminal author cannot regenerate or mutate the query/key after its score."""
    seen = []

    async def capture_reference(tested, packet):
        seen.append(tested.model_copy(deep=True))
        return await reference(tested, packet)

    author = FakeAuthor(structured=structured, changed=True)
    result = await run(author, render=capture_reference)
    assert result.finalized is not None
    assert seen[0] == draft(structured)
    assert result.finalized.task.query.text == draft(structured).question


async def test_finalization_selects_last_tested_round_instead_of_first():
    """A supported revision must publish the question actually scored in the last round."""

    class Runner(FakeRunner):
        async def assess(self, question, result, session, deadline):
            self.expected = 2 if session.cycle == 1 else 3
            return await super().assess(question, result, session, deadline)

    result = await run(FakeAuthor(rounds=2), Runner())
    assert result.finalized is not None and len(result.cycles) == 2
    assert result.finalized.task.query.text == "Revised question for round 2"


async def test_completed_reference_cannot_publish_without_qualifying_scores():
    """Citation completion cannot substitute for a valid paired miss."""
    runner = FakeRunner(expected=2)
    result = await run(runner=runner)
    assert result.finalized is None and result.error_type == "not_paired_miss"
    assert not runner.reference_calls


async def test_concurrent_finish_accepts_and_renders_only_once():
    """Concurrent tool submissions must not duplicate rendering or terminal acceptance."""
    rendered = []

    async def render(tested, packet):
        rendered.append(packet)
        await asyncio.sleep(0)
        return await reference(tested, packet)

    class Author(FakeAuthor):
        async def run(self, *, finish, **kwargs):
            async def concurrent_finish(declaration):
                outcomes = await asyncio.gather(finish(declaration), finish(declaration), return_exceptions=True)
                accepted = [value for value in outcomes if not isinstance(value, Exception)]
                rejected = [value for value in outcomes if isinstance(value, ValueError)]
                assert len(accepted) == len(rejected) == 1
                assert "terminal declaration" in str(rejected[0])
                return accepted[0]

            return await super().run(finish=concurrent_finish, **kwargs)

    result = await run(Author(), render=render)
    assert result.finalized is not None and len(rendered) == 1


@pytest.mark.parametrize("contradiction", [False, True])
async def test_report_that_omits_or_contradicts_tested_answer_cannot_publish(contradiction):
    runner = FakeRunner()
    runner.reference_assessments.append(
        FastJudgeAssessment.model_validate(
            {
                "expected_components": [{"component_id": "item", "is_correct": contradiction}],
                "excessive_components": [{"component_id": "different answer"}] if contradiction else [],
            }
        )
    )
    result = await run(runner=runner)
    assert result.finalized is None and result.reference is None
    assert result.error_type == "ReferenceValidationError"
    assert len(result.reference_consistency_assessments) == 1
    assert len(result.cycles) == 1 and set(result.cycles[0].f1_scores.values()) == {0.5}


async def test_report_assessment_provider_failure_cannot_be_treated_as_consistent():
    class Runner(FakeRunner):
        async def assess_reference(self, *args):
            raise RuntimeError("assessment unavailable")

    result = await run(runner=Runner())
    assert result.status == "operational_failure" and result.error_type == "RuntimeError"
    assert result.finalized is None and not result.reference_consistency_assessments
