import json
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
    CandidateStageError,
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


class _AuditFailureRunner(_Runner):
    async def run_stage(self, **kwargs: object) -> StageRunResult:
        if kwargs["stage"] == "audit":
            self.calls.append(kwargs)
            raise CandidateStageError(
                "transient_provider",
                "audit",
                "audit provider unavailable",
                elapsed_ms=9.0,
            )
        return await super().run_stage(**kwargs)


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
            answer_text="Alpha is established by the stronger report [[1]].",
            citation_evidence_ids=(evidence.evidence_id,),
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
        response_mode="plain_text",
    )


def _proof() -> ReferenceProof:
    return ReferenceProof(
        status="finalized",
        answer_text="## Result\n\nAlpha is 1200 [[1]] and Beta is 900 [[2]], so Alpha is larger.",
        citation_evidence_ids=("E1", "E2"),
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


def _structured_dossier() -> GroundedQuestionDossier:
    return _dossier(
        question="Return an object whose integer field value is Alpha's published value in whole units."
    ).model_copy(
        update={
            "response_mode": "structured",
            "output_schema_json": '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            '"title":"Published value","description":"Return the exact published value.",'
            '"type":"object","properties":{"value":{"type":"integer",'
            '"description":"Atomic published value in whole units."}},'
            '"required":["value"],"additionalProperties":false}',
            "structured_answer_json": '{"value":1200}',
        }
    )


def _structured_proof(*, value: int = 1200) -> ReferenceProof:
    return _proof().model_copy(update={"answer_text": None, "structured_answer_json": f'{{"value":{value}}}'})


@pytest.mark.anyio
async def test_single_question_generation_call_owns_question_and_dossier() -> None:
    """Future failure: QG must not regress to a source-form-conditioned second agent call."""
    runner = _Runner((_dossier(), _proof(), AuditResult(status="pass", explanation="complete")))
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

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
async def test_capability_preference_does_not_gate_candidate_selected_response_mode() -> None:
    """Future failure: preference drift must not become a hidden candidate rejection or retry trigger."""
    structured_runner = _Runner(
        (_structured_dossier(), _structured_proof(), AuditResult(status="pass", explanation="complete"))
    )
    structured = await CandidatePipeline(
        runner=structured_runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="evidence_grounded_calculation_or_proof",
    )
    plain_runner = _Runner((_dossier(), _proof(), AuditResult(status="pass", explanation="complete")))
    plain = await CandidatePipeline(
        runner=plain_runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=1, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="structured_field_semantics",
    )

    assert isinstance(structured, DomainTweakFinalizedTask)
    assert structured.task.query.output_schema is not None
    assert structured.task.reference_answer.text == '{"value":1200}'
    assert isinstance(plain, DomainTweakFinalizedTask)
    assert plain.task.query.output_schema is None
    assert [call["stage"] for call in structured_runner.calls] == ["question_generation", "reference", "audit"]
    assert [call["stage"] for call in plain_runner.calls] == ["question_generation", "reference", "audit"]
    reference_call = next(call for call in structured_runner.calls if call["stage"] == "reference")
    reference_payload = json.loads(str(reference_call["prompt"]).split("\n", 1)[1])
    assert reference_payload["dossier_hypothesis"]["output_schema_json"] == _structured_dossier().output_schema_json


def test_question_generation_contract_reports_invalid_structured_payload_before_reference() -> None:
    """Future failure: public schema/value defects must be visible QG feedback, not late task failures."""
    wrong_dialect = _structured_dossier().model_copy(
        update={
            "output_schema_json": _structured_dossier().output_schema_json.replace("2020-12", "2019-09")  # type: ignore[union-attr]
        }
    )

    assert any(
        "Draft 2020-12" in defect for defect in _question_generation_contract_defects(wrong_dialect, _workspace())
    )


def test_question_generation_contract_reports_json_numeric_limit_as_candidate_defect() -> None:
    """Future failure: one model-authored numeric literal must not terminate the complete batch."""
    numeric_limit = _structured_dossier().model_copy(
        update={"structured_answer_json": '{"value":' + ("9" * 5_000) + "}"}
    )

    defects = _question_generation_contract_defects(numeric_limit, _workspace())

    assert len(defects) == 1
    assert "could not be parsed" in defects[0]


@pytest.mark.anyio
async def test_answer_disclosing_structured_schema_cannot_finalize_candidate() -> None:
    """Future failure: a public schema must not reveal the private answer and become a finalized task."""
    leaking_dossier = _structured_dossier().model_copy(
        update={
            "output_schema_json": '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            '"type":"object","properties":{"value":{"type":"integer","const":1200}},'
            '"required":["value"],"additionalProperties":false}'
        }
    )
    runner = _Runner((leaking_dossier, _structured_proof()))

    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="structured_field_semantics",
    )

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "proof_invalid"
    assert outcome.terminal_stage == "reference"
    assert outcome.failure_reason is not None
    assert "must not disclose the answer" in outcome.failure_reason
    assert [call["stage"] for call in runner.calls] == ["question_generation", "reference"]


@pytest.mark.anyio
async def test_semantic_schema_disclosure_rejected_by_audit_cannot_finalize_candidate() -> None:
    """Future failure: semantic schema leakage must reach audit and never silently finalize."""
    leaking_dossier = _structured_dossier().model_copy(
        update={
            "output_schema_json": '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            '"type":"object","properties":{"answer1200":{"type":"integer"}},'
            '"required":["answer1200"],"additionalProperties":false}',
            "structured_answer_json": '{"answer1200":1200}',
        }
    )
    leaking_proof = _structured_proof().model_copy(update={"structured_answer_json": '{"answer1200":1200}'})
    audit_reject = AuditResult(
        status="reject",
        defects=("The public property name reveals the canonical value.",),
        explanation="The exact public schema discloses the answer.",
    )
    runner = _Runner((leaking_dossier, leaking_proof, audit_reject, leaking_proof, audit_reject))

    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="structured_field_semantics",
    )

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "audit_rejected"
    assert outcome.terminal_stage == "audit"
    assert outcome.failure_reason == "The exact public schema discloses the answer."
    assert [call["stage"] for call in runner.calls] == [
        "question_generation",
        "reference",
        "audit",
        "reference_repair",
        "audit",
    ]


@pytest.mark.anyio
async def test_audit_execution_failure_cannot_finalize_candidate() -> None:
    """Future failure: an unavailable semantic auditor must remain a terminal candidate outcome."""
    runner = _AuditFailureRunner((_dossier(), _proof()))

    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

    assert isinstance(outcome, CandidateFailure)
    assert outcome.failure_class == "transient_provider"
    assert outcome.terminal_stage == "audit"
    assert outcome.stage_summaries[-1].stage == "audit"
    assert outcome.stage_summaries[-1].outcome == "transient_provider"
    assert [call["stage"] for call in runner.calls] == ["question_generation", "reference", "audit"]


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
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

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
async def test_structured_repair_receives_immutable_contract_and_replaces_rejected_value() -> None:
    """Future failure: structured repair must not infer or mutate the fixed public schema."""
    runner = _Runner(
        (
            _structured_dossier(),
            _structured_proof(value=1200),
            AuditResult(status="reject", defects=("value should be 1300",), explanation="stale value"),
            _structured_proof(value=1300),
            AuditResult(status="pass", explanation="complete"),
        )
    )

    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.repaired
    assert outcome.task.reference_answer.text == '{"value":1300}'
    repair_call = next(call for call in runner.calls if call["stage"] == "reference_repair")
    payload = json.loads(str(repair_call["prompt"]).split("\n", 1)[1])
    assert payload["immutable_response_mode"] == "structured"
    assert payload["immutable_output_schema_json"] == _structured_dossier().output_schema_json


@pytest.mark.anyio
async def test_reference_repair_can_acquire_stronger_source_and_owns_final_citations() -> None:
    """Future failure: accepted rendering must not retain evidence rejected before source-upgrading repair."""
    workspace = _workspace()
    initial = ReferenceProof(
        status="finalized",
        answer_text="The initial report lists Alpha at 1200 [[1]].",
        citation_evidence_ids=("E1",),
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
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.task.reference_answer.citations is not None
    assert tuple(item.url for item in outcome.task.reference_answer.citations if item is not None) == (
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
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

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
        await pipeline.run(
            PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
            capability_preference="general_deep_research",
        )
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
        await pipeline.run(
            PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
            capability_preference="general_deep_research",
        )
    assert captured.value.stage == "question_generation"


@pytest.mark.anyio
async def test_public_question_over_ordinary_audit_target_reaches_finalized_task() -> None:
    """Future failure: the production pipeline must not impose a smaller reference-only query limit."""
    question = "Q" * 128_001
    runner = _Runner((_dossier(question=question), _proof(), AuditResult(status="pass", explanation="complete")))
    outcome = await CandidatePipeline(
        runner=runner,  # type: ignore[arg-type]
        source_fetcher=_UnusedFetcher(),
        workspace_factory=_workspace,
    ).run(
        PortfolioAllocation(slot=0, ecosystems=("a", "b", "c", "d", "e")),
        capability_preference="general_deep_research",
    )

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.task.query.text == question
