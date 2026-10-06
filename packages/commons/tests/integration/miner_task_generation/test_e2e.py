"""Deterministic composed lifecycle through public schemas and source-proof rendering."""

import json
from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace

import pytest

from harnyx_commons.domain.tool_usage_accounting import known_zero_actual_cost_tool_usage
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
from harnyx_commons.miner_task_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.miner_task_generation.contracts import (
    AgentResult,
    CandidateSession,
    QuestionDraft,
    ReferenceProof,
    ReviewDecision,
    Seed,
    SeedSource,
    SourceSupport,
    StageRunResult,
    StructuredQuestionDraft,
)
from harnyx_commons.miner_task_generation.source_workspace import SourceDocument, SourceWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.mark.parametrize(
    ("mode", "terminal_value", "proposal", "corrected", "comparison"),
    [
        ("plain_text", 1200, 1200, False, "agree"),
        ("plain_text", 1300, 1200, False, "disagree"),
        ("plain_text", 1200, 1200, False, "excess"),
        ("plain_text", 1200, 1200, False, "malformed"),
        ("plain_text", 1200, 1200, False, "provider"),
        ("plain_text", 1200, 1200, False, "deadline"),
        ("structured", 1200, 1200, False, None),
        ("structured", 1300, 1200, True, None),
        ("structured", 1, True, True, None),
        ("structured", 0, False, True, None),
    ],
)
async def test_full_candidate_lifecycle_renders_public_reference_from_registered_evidence(
    monkeypatch, mode, terminal_value, proposal, corrected, comparison
):
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.org/report",
            final_url="https://example.org/report",
            media_type="text/plain",
            content="Alpha value 1200",
            fetched_bytes=16,
        )
    )
    line = workspace.lines(source)[0]
    evidence = workspace.register_evidence(
        claim="Alpha value 1200", start_line_id=line.line_id, end_line_id=line.line_id
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": ["integer", "boolean"]}},
        "required": ["value"],
        "additionalProperties": False,
        "allOf": [{"properties": {"value": {"type": ["integer", "boolean"]}}}],
    }
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.SourceWorkspace", lambda: workspace)

    async def reference_stage(self, **kwargs):
        prompt = json.loads(kwargs["prompt"])
        assert prompt["question"] == "What value does Alpha have in the named report?"
        assert {"url": "https://example.org/report", "evidence": "Alpha value 1200"} in prompt["source_support"]
        assert "solver-private-trace" not in kwargs["prompt"]
        assert prompt["verification"] == {"pass": True, "feedback": ""}
        proof = ReferenceProof(
            status="finalized",
            answer_text=f"{terminal_value} [[1]]" if mode == "plain_text" else None,
            structured_answer_json=f'{{ "value": {terminal_value} }}' if mode == "structured" else None,
            citation_evidence_ids=(evidence.evidence_id,),
            answers=({"answer_id": "A1"},),
            proof_steps=(
                {
                    "step_id": "S1",
                    "statement": "Alpha value 1200",
                    "kind": "supported",
                    "evidence_ids": (evidence.evidence_id,),
                },
            ),
        )
        assert not kwargs["output_validator"](proof)
        return StageRunResult(proof, 1, known_zero_actual_cost_tool_usage())

    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run_stage", reference_stage
    )

    class Judge:
        async def invoke(self, request):
            assert comparison is not None  # Structured references need no semantic call.
            if comparison == "provider":
                raise RuntimeError("provider interrupted")
            if comparison == "deadline":
                import asyncio

                await asyncio.Event().wait()
            assessment = FastJudgeAssessment(
                expected_components=[{"component_id": "value", "is_correct": comparison != "disagree"}],
                excessive_components=[{"component_id": "extra"}] if comparison == "excess" else [],
            )
            return SimpleNamespace(
                raw_text="{}" if comparison == "malformed" else assessment.model_dump_json(),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

    terminal_runner = GenerationAgentRunner(project_id=None, judge=Judge())

    class Runner:
        async def invoke(self, role, message, session, deadline):
            assert role != "analyst"
            if role == "author":
                payload = dict(
                    question="What value does Alpha have in the named report?",
                    answer="1200",
                    answer_components=["1200"],
                    scope="named report",
                    source_support=[SourceSupport(url="https://example.org/report", evidence="Alpha value 1200")],
                    revision_note="current",
                )
                draft = (
                    StructuredQuestionDraft(
                        **payload,
                        output_schema_json=json.dumps(schema),
                        structured_answer_json=json.dumps({"value": proposal}),
                    )
                    if mode == "structured"
                    else QuestionDraft(**payload)
                )
                return AgentResult(answer=draft.model_dump_json())
            if role in {"reviewer", "verifier"}:
                return AgentResult(answer=ReviewDecision(passed=True, feedback="").model_dump_json(by_alias=True))
            if role in {"solver", "grounded_solver"}:
                assert "source_support" not in message and "1200" not in message
                return AgentResult(
                    answer='{"value":900}' if mode == "structured" else "900",
                    provider_responses=[{"response": {"text": "solver-private-trace"}}],
                )
            return AgentResult(answer="Recorded solver route analysis")

        async def assess(self, *args):
            return FastJudgeAssessment(
                expected_components=[{"component_id": "value", "is_correct": False}], excessive_components=[]
            )

        async def derive_reference(self, draft, verification, session, deadline):
            return await terminal_runner.derive_reference(draft, verification, session, deadline)

    seed = Seed(
        seed_id="alpha",
        seed_question="Alpha value?",
        reference_answer="1200",
        scope="report",
        event_date="2026-10-01",
        supporting_sources=[
            SeedSource(url="https://example.org/report", evidence="Alpha value 1200", publication_date="2026-10-01")
        ],
    )
    result = await CandidatePipeline(runner=Runner()).run(
        seed,
        mode=mode,
        fast=True,
        session=CandidateSession(task_id="e2e", effective_date=datetime.now(UTC).date()),
        deadline=monotonic() + (0.1 if comparison == "deadline" else 5),
    )
    assert result.cycles[-1].analysis is None
    if comparison in {"malformed", "provider", "deadline"}:
        assert result.status == "operational_failure" and result.finalized is None
        return
    if comparison in {"disagree", "excess"}:
        assert result.status == "unresolved_reference" and result.finalized is None
        assert result.reference and "disagrees" in result.reference.unsupported_reason
        assert result.reference.reference_answer.text == f"{terminal_value} [[1]]"
        return
    if corrected:
        assert result.reference and result.reference.factual_correction
        assert result.status == "unresolved_reference" and result.finalized is None
        return
    assert result.status == "finalized" and result.finalized
    task = result.finalized.task
    assert task.query.output_schema == (schema if mode == "structured" else None)
    assert task.query.fast is True and task.reference_answer.citations
    assert task.reference_answer.text == ('{"value":1200}' if mode == "structured" else "1200 [[1]]")
    assert "seed" not in task.model_dump_json() and "solver_results" not in task.model_dump_json()
