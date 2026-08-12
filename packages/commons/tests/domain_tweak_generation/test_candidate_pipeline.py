from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from harnyx_commons.domain.tool_usage import LlmUsageSummary, ToolUsageSummary
from harnyx_commons.domain_tweak_generation import (
    AuditResult,
    BatchTerminalGenerationError,
    CandidateFailure,
    CandidatePipeline,
    DomainTweakFinalizedTask,
    DossierAnswer,
    DossierFact,
    DossierRequirement,
    GroundedQuestionDossier,
    PortfolioAllocation,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    SourceDocument,
    SourceWorkspace,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation.candidate_pipeline import (
    AGENT_STAGE_TIMEOUT_SECONDS,
    _question_generation_contract_defects,
)


class _Runner:
    def __init__(self, outputs: Sequence[BaseModel]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        self.calls.append(kwargs)
        return StageRunResult(self.outputs.pop(0), 1.0, ToolUsageSummary.zero())


class _UnusedFetcher:
    async def fetch(self, url: str, *, document_kind: str) -> SourceDocument:
        del url, document_kind
        raise AssertionError("prepopulated workspace must not fetch")


class _UnexpectedRunner:
    async def run_stage(self, **_kwargs: object) -> StageRunResult:
        raise RuntimeError("broken local stage adapter")


class _WrongOutputRunner:
    async def run_stage(self, **_kwargs: object) -> StageRunResult:
        usage = ToolUsageSummary(
            llm=LlmUsageSummary(actual_cost=0.75),
            actual_total_cost_usd=0.75,
            actual_cost_by_provider={"vertex": 0.75},
        )
        return StageRunResult(AuditResult(status="pass", explanation="wrong boundary"), 1.0, usage)


class _RepairAcquisitionRunner(_Runner):
    def __init__(self, outputs: Sequence[BaseModel], workspace: SourceWorkspace) -> None:
        super().__init__(outputs)
        self.workspace = workspace

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        self.calls.append(kwargs)
        if kwargs["stage"] != "reference_repair":
            return StageRunResult(self.outputs.pop(0), 1.0, ToolUsageSummary.zero())
        source = self.workspace.store(
            SourceDocument(
                requested_url="https://example.org/stronger-report",
                final_url="https://example.org/stronger-report",
                media_type="text/plain",
                content="HEADER\tName\tValue\nROW\tAlpha\t1200",
                fetched_bytes=40,
            )
        )
        lines = self.workspace.lines(source)
        evidence = self.workspace.register_evidence(
            claim="Stronger Alpha value",
            start_line_id=lines[1].line_id,
            end_line_id=lines[1].line_id,
        )
        repaired = ReferenceProof(
            status="finalized",
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="The stronger report establishes Alpha at 1200.",
                    kind="supported",
                    evidence_ids=(evidence.evidence_id,),
                ),
            ),
        )
        return StageRunResult(repaired, 1.0, ToolUsageSummary.zero())


def _workspace() -> SourceWorkspace:
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="HEADER\tName\tValue\r\nROW\tAlpha\t1200\r\nROW\tBeta\t900",
            fetched_bytes=58,
        )
    )
    lines = workspace.lines(source)
    workspace.register_evidence(claim="Alpha", start_line_id=lines[1].line_id, end_line_id=lines[1].line_id)
    workspace.register_evidence(claim="Beta", start_line_id=lines[2].line_id, end_line_id=lines[2].line_id)
    return workspace


def _dossier(*, question: str = "Which named row has the larger value?") -> GroundedQuestionDossier:
    return GroundedQuestionDossier(
        status="ready",
        subject="Published comparison",
        route_summary="Reconcile a bounded table with separate status evidence",
        question=question,
        answers=(DossierAnswer(answer_id="A1", value="Alpha"),),
        requirements=(DossierRequirement(description="compare every bounded row"),),
        source_facts=(
            DossierFact(statement="Alpha 1200", evidence_ids=("E1",)),
            DossierFact(statement="Beta 900", evidence_ids=("E2",)),
        ),
        derivation="Compare the two complete rows and preserve table order",
        why_not_one_page="The status evidence is separate from the bounded table",
        substantive_final_condition="The value comparison removes Beta",
    )


def _proof() -> ReferenceProof:
    return ReferenceProof(
        status="finalized",
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(
            ProofStep(step_id="S1", statement="Alpha is 1200.", kind="supported", evidence_ids=("E1",)),
            ProofStep(step_id="S2", statement="Beta is 900.", kind="supported", evidence_ids=("E2",)),
            ProofStep(
                step_id="S3",
                statement="Alpha is the larger value.",
                kind="derived",
                depends_on_step_ids=("S1", "S2"),
            ),
        ),
    )


@pytest.mark.anyio
async def test_single_question_generation_call_owns_question_and_dossier() -> None:
    """Future failure: QG must not regress to a source-form-conditioned second agent call."""
    runner = _Runner((_dossier(), _proof(), AuditResult(status="pass", explanation="complete")))
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert [call["stage"] for call in runner.calls] == ["question_generation", "reference", "audit"]
    assert all(call["timeout_seconds"] == AGENT_STAGE_TIMEOUT_SECONDS for call in runner.calls)
    assert "source_form" not in str(runner.calls[0]["prompt"])
    audit_tools = runner.calls[2]["tool_set"].allowed_tools  # type: ignore[union-attr]
    assert audit_tools == (
        "mcp__audit_vfs__list_sources",
        "mcp__audit_vfs__regex_search",
        "mcp__audit_vfs__read_lines",
    )


@pytest.mark.anyio
async def test_audit_rejection_gets_one_reference_repair_and_second_read_only_audit() -> None:
    """Future failure: a correctable proof defect must get one material repair and no silent pass."""
    runner = _Runner(
        (
            _dossier(),
            _proof(),
            AuditResult(status="reject", defects=("bind the second operand",), explanation="gap"),
            _proof(),
            AuditResult(status="pass", explanation="complete"),
        )
    )
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.repaired
    assert [call["stage"] for call in runner.calls] == [
        "question_generation",
        "reference",
        "audit",
        "reference_repair",
        "audit",
    ]
    audit_calls = [call for call in runner.calls if call["stage"] == "audit"]
    assert audit_calls[0]["tool_set"].allowed_tools == audit_calls[1]["tool_set"].allowed_tools  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_reference_repair_can_acquire_stronger_source_and_owns_final_citations() -> None:
    """Future failure: accepted rendering must not retain evidence rejected before source-upgrading repair."""
    workspace = _workspace()
    initial = ReferenceProof(
        status="finalized",
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(
            ProofStep(
                step_id="S1",
                statement="The initial report establishes Alpha at 1200.",
                kind="supported",
                evidence_ids=("E1",),
            ),
        ),
    )
    runner = _RepairAcquisitionRunner(
        (
            _dossier(),
            initial,
            AuditResult(status="reject", defects=("upgrade the source",), explanation="weak source"),
            AuditResult(status="pass", explanation="complete"),
        ),
        workspace,
    )
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=lambda: workspace,
    ).run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.task.reference_answer.citations is not None
    assert tuple(item.url for item in outcome.task.reference_answer.citations) == (
        "https://example.org/stronger-report",
    )


@pytest.mark.anyio
async def test_no_generate_retains_first_typed_blocker() -> None:
    """Future failure: a genuine QG blocker must remain terminal and observable."""
    runner = _Runner(
        (
            GroundedQuestionDossier(
                status="no_generate",
                failure_reason="The complete public roster cannot be established",
                failure_class="reasoning_no_generate",
            ),
        )
    )
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "reasoning_no_generate"
    assert outcome.terminal_stage == "question_generation"
    assert outcome.failure_reason == "The complete public roster cannot be established"
    assert outcome.source_failure_id is None


def test_source_failure_must_match_workspace_evidence() -> None:
    """Future failure: an incidental fetch failure must not become the declared terminal cause."""
    dossier = GroundedQuestionDossier(
        status="no_generate",
        failure_reason="The selected source could not be used",
        failure_class="source_unavailable",
        source_failure_id="source_failure:1",
    )
    assert _question_generation_contract_defects(dossier, SourceWorkspace()) == (
        "question-generation source_failure_id was not observed by the workspace",
    )


def test_source_failure_id_must_resolve_to_declared_class() -> None:
    dossier = GroundedQuestionDossier(
        status="no_generate",
        failure_reason="The required document was rejected",
        failure_class="source_fetch_rejected",
        source_failure_id="source_failure:1",
    )
    workspace = SimpleNamespace(
        source_failure=lambda failure_id: SimpleNamespace(
            failure_id=failure_id,
            failure_class="source_unavailable",
        )
    )
    assert _question_generation_contract_defects(dossier, workspace) == (
        "question-generation source_failure_id does not match its declared failure_class",
    )


@pytest.mark.anyio
async def test_wrong_internal_stage_output_becomes_batch_terminal() -> None:
    pipeline = CandidatePipeline(
        runner=_WrongOutputRunner(),  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )
    with pytest.raises(BatchTerminalGenerationError) as captured:
        await pipeline.run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))
    assert captured.value.failure_class == "unexpected_pipeline_failure"
    assert captured.value.stage == "question_generation"
    assert captured.value.tool_usage.actual_total_cost_usd == 0.75


@pytest.mark.anyio
async def test_unexpected_pipeline_exception_becomes_typed_batch_terminal_fault() -> None:
    pipeline = CandidatePipeline(
        runner=_UnexpectedRunner(),  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )
    with pytest.raises(BatchTerminalGenerationError) as captured:
        await pipeline.run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))
    assert captured.value.stage == "question_generation"


@pytest.mark.anyio
async def test_packet_size_first_exposed_by_question_is_proof_invalid() -> None:
    runner = _Runner((_dossier(question="Q" * 128_001), _proof()))
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")))
    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "proof_invalid"
    assert outcome.terminal_stage == "reference"
