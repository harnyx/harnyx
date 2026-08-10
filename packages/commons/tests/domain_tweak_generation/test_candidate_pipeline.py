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
    GenerationForm,
    PortfolioAllocation,
    ProofStep,
    QuestionPacket,
    ReferenceAnswerSelection,
    ReferenceProof,
    SourceDocument,
    SourceDossier,
    SourceWorkspace,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation.candidate_pipeline import _dossier_contract_defects


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
        return StageRunResult(
            QuestionPacket(
                status="giveup",
                form_transfer_explanation="wrong boundary type",
                giveup_reason="wrong boundary type",
            ),
            1.0,
            ToolUsageSummary(
                llm=LlmUsageSummary(actual_cost=0.75),
                actual_total_cost_usd=0.75,
                actual_cost_by_provider={"vertex": 0.75},
            ),
        )


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
            content="HEADER\tName\tValue\nROW\tAlpha\t1200\nROW\tBeta\t900",
            fetched_bytes=58,
        )
    )
    lines = workspace.lines(source)
    workspace.register_evidence(claim="Alpha", start_line_id=lines[1].line_id, end_line_id=lines[1].line_id)
    workspace.register_evidence(claim="Beta", start_line_id=lines[2].line_id, end_line_id=lines[2].line_id)
    return workspace


def _dossier() -> SourceDossier:
    return SourceDossier(
        status="ready",
        subject="Published comparison",
        route_summary="route",
        question_plan="compare two bounded rows",
        answers=(DossierAnswer(answer_id="A1", value="Alpha"),),
        requirements=(DossierRequirement(description="compare"),),
        source_facts=(
            DossierFact(statement="Alpha 1200", evidence_ids=("E1",)),
            DossierFact(statement="Beta 900", evidence_ids=("E2",)),
        ),
        derivation="1200 is greater than 900",
    )


@pytest.mark.anyio
async def test_audit_rejection_gets_one_reference_repair_and_second_audit() -> None:
    """Future failure: a correctable proof defect must not discard an otherwise good candidate."""
    dossier = _dossier()
    proof = ReferenceProof(
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
    runner = _Runner(
        (
            dossier,
            QuestionPacket(
                status="generated",
                question="Which named row has the larger value?",
                form_transfer_explanation="preserved",
            ),
            proof,
            AuditResult(status="reject", defects=("bind the second operand",), explanation="gap"),
            proof,
            AuditResult(status="pass", explanation="complete"),
        )
    )
    pipeline = CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )
    outcome = await pipeline.run(
        GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
    )

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.repaired
    assert outcome.route_context is not None
    assert outcome.route_context.subject == "Published comparison"
    assert outcome.route_context.source_urls == ("https://example.com/report",)
    assert [call["stage"] for call in runner.calls] == [
        "dossier",
        "question",
        "reference",
        "audit",
        "reference_repair",
        "audit",
    ]
    assert "Hidden source form" not in str(runner.calls[0]["prompt"])
    assert "Hidden source form" in str(runner.calls[1]["prompt"])


@pytest.mark.anyio
async def test_reference_repair_can_acquire_stronger_source_and_owns_final_citations() -> None:
    """Future failure: accepted rendering must not retain evidence rejected before source-upgrading repair."""
    workspace = _workspace()
    initial_proof = ReferenceProof(
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
            QuestionPacket(
                status="generated",
                question="Which named row has the supported larger value?",
                form_transfer_explanation="preserved",
            ),
            initial_proof,
            AuditResult(status="reject", defects=("upgrade the source",), explanation="weak source"),
            AuditResult(status="pass", explanation="complete"),
        ),
        workspace,
    )
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=lambda: workspace,
    ).run(
        GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
    )

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.route_context is not None
    assert outcome.route_context.source_urls == ("https://example.org/stronger-report",)
    assert outcome.task.reference_answer.citations is not None
    assert tuple(item.url for item in outcome.task.reference_answer.citations) == (
        "https://example.org/stronger-report",
    )
    reference_calls = [call for call in runner.calls if call["stage"] in {"reference", "reference_repair"}]
    assert all(call["web_search"] is True for call in reference_calls)
    assert all(call["tool_set"].search_result_registrar is not None for call in reference_calls)  # type: ignore[union-attr]


@pytest.mark.anyio
async def test_candidate_giveup_retains_stable_failure_class_and_stage() -> None:
    """Future failure: a discarded candidate must retain bounded aggregate attribution without raw model text."""
    runner = _Runner(
        (
            _dossier(),
            QuestionPacket(
                status="giveup",
                form_transfer_explanation="The required operation is unnatural for this dossier.",
                giveup_reason="the dossier cannot support the source form's exhaustive boundary",
            ),
        )
    )
    pipeline = CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )

    outcome = await pipeline.run(
        GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
    )

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "reasoning_no_generate"
    assert outcome.terminal_stage == "question"


def test_dossier_source_failure_must_match_workspace_evidence() -> None:
    """Future failure: an incidental tool failure must not become the dossier's declared terminal cause."""
    dossier = SourceDossier(
        status="no_generate",
        unresolved_gaps=("source unavailable",),
        no_generate_reason="the selected source could not be used",
        failure_class="source_unavailable",
        source_failure_id="source_failure:1",
    )

    assert _dossier_contract_defects(dossier, SourceWorkspace()) == (
        "dossier source_failure_id was not observed by the workspace",
    )


def test_dossier_source_failure_id_must_resolve_to_the_declared_class() -> None:
    """Future failure: one failed fetch must not authorize a different terminal source cause."""
    dossier = SourceDossier(
        status="no_generate",
        unresolved_gaps=("required document rejected",),
        no_generate_reason="the required document was rejected by source policy",
        failure_class="source_fetch_rejected",
        source_failure_id="source_failure:1",
    )
    workspace = SimpleNamespace(
        source_failure=lambda failure_id: SimpleNamespace(
            failure_id=failure_id,
            failure_class="source_unavailable",
        )
    )

    assert _dossier_contract_defects(dossier, workspace) == (
        "dossier source_failure_id does not match its declared failure_class",
    )


@pytest.mark.anyio
async def test_wrong_internal_stage_output_becomes_batch_terminal() -> None:
    """Future failure: an internal output-type defect must not enter paid candidate refill."""
    pipeline = CandidatePipeline(
        runner=_WrongOutputRunner(),  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )

    with pytest.raises(BatchTerminalGenerationError) as captured:
        await pipeline.run(
            GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
            PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        )

    assert captured.value.failure_class == "unexpected_pipeline_failure"
    assert captured.value.stage == "dossier"
    assert captured.value.tool_usage.actual_total_cost_usd == 0.75


@pytest.mark.anyio
async def test_unexpected_pipeline_exception_becomes_a_typed_batch_terminal_fault() -> None:
    """Future failure: a shared host-code defect must not abort without an observable terminal class."""
    pipeline = CandidatePipeline(
        runner=_UnexpectedRunner(),  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    )

    with pytest.raises(BatchTerminalGenerationError) as captured:
        await pipeline.run(
            GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
            PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        )

    assert captured.value.failure_class == "unexpected_pipeline_failure"
    assert captured.value.stage == "dossier"


@pytest.mark.anyio
async def test_packet_size_first_exposed_by_actual_question_is_proof_invalid() -> None:
    """Future failure: a final-question packet overflow must discard one candidate, not abort refill."""
    runner = _Runner(
        (
            _dossier(),
            QuestionPacket(
                status="generated",
                question="Q" * 128_001,
                form_transfer_explanation="preserved",
            ),
            ReferenceProof(
                status="finalized",
                answers=(ReferenceAnswerSelection(answer_id="A1"),),
                proof_steps=(
                    ProofStep(
                        step_id="S1",
                        statement="Alpha is the supported answer.",
                        kind="supported",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
        )
    )

    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        GenerationForm(form_identity="form:1", source_index=1, form="Hidden source form"),
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
    )

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "proof_invalid"
    assert outcome.terminal_stage == "reference"
