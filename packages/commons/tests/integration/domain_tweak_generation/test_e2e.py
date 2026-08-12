from __future__ import annotations

import pytest
from pydantic import BaseModel

from harnyx_commons.domain.tool_usage_accounting import known_zero_actual_cost_tool_usage
from harnyx_commons.domain_tweak_generation import (
    AuditResult,
    CandidatePipeline,
    DossierAnswer,
    DossierFact,
    DossierRequirement,
    GroundedQuestionDossier,
    PortfolioAllocation,
    PortfolioPacket,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    ShortfallRefillPipeline,
    SourceDocument,
    SourceWorkspace,
    StageRunResult,
)

pytestmark = pytest.mark.integration


class _ScriptedRunner:
    def __init__(self, outputs: list[BaseModel]) -> None:
        self.outputs = outputs

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        del kwargs
        return StageRunResult(self.outputs.pop(0), 1, known_zero_actual_cost_tool_usage())


class _NoFetch:
    async def fetch(self, url: str, *, document_kind: str) -> SourceDocument:
        raise AssertionError(f"unexpected fetch: {url} ({document_kind})")


def _workspace() -> SourceWorkspace:
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="HEADER\tName\tValue\nROW\tAlpha\t1200\nROW\tBeta\t900",
            fetched_bytes=58,
        )
    )
    lines = workspace.lines(source)
    workspace.register_evidence(claim="Alpha", start_line_id=lines[1].line_id, end_line_id=lines[1].line_id)
    workspace.register_evidence(claim="Beta", start_line_id=lines[2].line_id, end_line_id=lines[2].line_id)
    return workspace


@pytest.mark.anyio
async def test_composed_pipeline_reaches_plain_miner_task_without_hidden_form_leakage() -> None:
    dossier = GroundedQuestionDossier(
        status="ready",
        subject="Published comparison",
        route_summary="route",
        question="Which named row has the larger value?",
        answers=(DossierAnswer(answer_id="A1", value="Alpha"),),
        requirements=(DossierRequirement(description="compare"),),
        source_facts=(
            DossierFact(statement="Alpha 1200", evidence_ids=("E1",)),
            DossierFact(statement="Beta 900", evidence_ids=("E2",)),
        ),
        derivation="1200 exceeds 900",
        why_not_one_page="The bounded roster and decisive values come from separate records.",
        substantive_final_condition="The exact numeric comparison removes Beta.",
        response_mode="plain_text",
    )
    proof = ReferenceProof(
        status="finalized",
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(
            ProofStep(step_id="S1", statement="Alpha is 1200", kind="supported", evidence_ids=("E1",)),
            ProofStep(step_id="S2", statement="Beta is 900", kind="supported", evidence_ids=("E2",)),
            ProofStep(step_id="S3", statement="Alpha is larger", kind="derived", depends_on_step_ids=("S1", "S2")),
        ),
    )
    runner = _ScriptedRunner(
        [
            PortfolioPacket(
                allocations=(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),)
            ),
            dossier,
            proof,
            AuditResult(status="pass", explanation="complete"),
        ]
    )
    candidate = CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_NoFetch(),
        workspace_factory=_workspace,
    )
    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidate,
    ).generate_batch(target_count=1)

    assert result.finalized_tasks[0].task.query.output_schema is None
    assert result.finalized_tasks[0].task.reference_answer.citations
    assert result.slot_attempt_count == 1
    assert result.finalized_tasks[0].task.task_id.int != 0


@pytest.mark.anyio
async def test_composed_pipeline_reaches_structured_miner_task_on_the_same_shared_path() -> None:
    """Future failure: structured generation must not require a parallel materialization lifecycle."""
    dossier = GroundedQuestionDossier(
        status="ready",
        subject="Published comparison",
        route_summary="route",
        question="Return an object whose integer field value is Alpha's published value in whole units.",
        answers=(DossierAnswer(answer_id="A1", value="Alpha"),),
        requirements=(DossierRequirement(description="read the value"),),
        source_facts=(DossierFact(statement="Alpha 1200", evidence_ids=("E1",)),),
        derivation="Read Alpha's row",
        why_not_one_page="The roster identity and value are separate records.",
        substantive_final_condition="The numeric value determines the field.",
        response_mode="structured",
        output_schema_json='{"$schema":"https://json-schema.org/draft/2020-12/schema",'
        '"type":"object","properties":{"value":{"type":"integer"}},'
        '"required":["value"],"additionalProperties":false}',
        structured_answer_json='{"value":1200}',
    )
    proof = ReferenceProof(
        status="finalized",
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(
            ProofStep(step_id="S1", statement="Alpha is 1200", kind="supported", evidence_ids=("E1",)),
        ),
        structured_answer_json='{"value":1200}',
    )
    runner = _ScriptedRunner(
        [
            PortfolioPacket(
                allocations=(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),)
            ),
            dossier,
            proof,
            AuditResult(status="pass", explanation="complete"),
        ]
    )

    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=CandidatePipeline(
            runner=runner,  # type: ignore[arg-type]
            source_fetcher=_NoFetch(),
            workspace_factory=_workspace,
        ),
    ).generate_batch(target_count=1)

    task = result.finalized_tasks[0].task
    assert task.query.output_schema is not None
    assert task.reference_answer.text == '{"value":1200}'
    assert task.reference_answer.citations
    assert result.slot_attempt_count == 1
