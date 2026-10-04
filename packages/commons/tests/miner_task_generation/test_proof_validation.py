import pytest

from harnyx_commons.miner_task_generation import (
    DossierAnswer,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    SourceDocument,
    SourceWorkspace,
)
from harnyx_commons.miner_task_generation.contracts import ReferenceQuestion
from harnyx_commons.miner_task_generation.proof_validation import (
    ProofValidationError,
    reference_contract_defects,
    validate_and_render_reference,
)


def _workspace() -> SourceWorkspace:
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="HEADER\tName\tValue\nROW\tAlpha\t1,200",
            fetched_bytes=20,
        )
    )
    lines = workspace.lines(source)
    workspace.register_evidence(
        claim="Alpha value",
        start_line_id=lines[1].line_id,
        end_line_id=lines[1].line_id,
    )
    return workspace


def _dossier(*, answer: str = "1200", question: str = "Which value?") -> ReferenceQuestion:
    return ReferenceQuestion(
        status="ready",
        question=question,
        answers=(DossierAnswer(answer_id="A1", value=answer),),
        response_mode="plain_text",
    )


def _structured_dossier(*, schema_dialect: str = "https://json-schema.org/draft/2020-12/schema") -> ReferenceQuestion:
    schema = (
        '{"$schema":"' + schema_dialect + '","type":"object","properties":{"value":{"type":"integer"}},'
        '"required":["value"],"additionalProperties":false}'
    )
    return _dossier(
        question="Return an object whose integer field value is Alpha's published value in whole units."
    ).model_copy(
        update={
            "response_mode": "structured",
            "output_schema_json": schema,
            "structured_answer_json": '{"value":1200}',
        }
    )


def test_author_owned_citation_markers_reach_public_reference() -> None:
    """Future failure: the host must preserve the author's exact public pointer mapping."""
    validated = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha has value 1200 [[1]].",
            note="The question used the wrong unit; this is the corrected value [[1]].",
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Alpha has value 1200.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
        ),
        workspace=_workspace(),
    )
    assert "[[1]]" in validated.reference_answer.text
    assert validated.reference_answer.note == ("The question used the wrong unit; this is the corrected value [[1]].")
    assert validated.reference_answer.citations is not None


def test_reference_note_is_optional_but_invalid_on_giveup() -> None:
    with pytest.raises(ValueError, match="giveup proof cannot contain a public answer, note, or citation positions"):
        ReferenceProof(
            status="giveup",
            answer_text=None,
            note="This cannot stand in for an answer.",
            citation_evidence_ids=(),
            giveup_reason="Evidence unavailable.",
        )


@pytest.mark.parametrize(
    ("dossier_answer", "corrected_value", "proof_statement"),
    (("1200", "Alpha [[1]]", "Alpha has value 1200."),),
)
def test_private_proof_fields_reject_public_citation_markers(
    dossier_answer: str,
    corrected_value: str | None,
    proof_statement: str,
) -> None:
    """Future failure: private audit fields must not introduce a second pointer mapping."""
    with pytest.raises(ProofValidationError, match="private proof fields cannot contain public citation markers"):
        validate_and_render_reference(
            dossier=_dossier(answer=dossier_answer),
            proof=ReferenceProof(
                status="finalized",
                answer_text="Alpha has value 1200 [[1]].",
                citation_evidence_ids=("E1",),
                answers=(ReferenceAnswerSelection(answer_id="A1", corrected_value=corrected_value),),
                proof_steps=(
                    ProofStep(
                        step_id="S1",
                        statement=proof_statement,
                        kind="supported",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
            workspace=_workspace(),
        )


def test_reference_preserves_authored_markdown_and_explicit_xml_without_host_rewriting() -> None:
    """Future failure: the host must preserve reader-facing synthesis and explicit requested forms."""
    markdown = "## Result\n\nAlpha is the published value [[1]]."
    markdown_reference = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text=markdown,
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
            ),
        ),
        workspace=_workspace(),
    )
    xml = "<answer><value>Alpha</value><evidence>[[1]]</evidence></answer>"
    xml_reference = validate_and_render_reference(
        dossier=_dossier(question="Return XML only."),
        proof=ReferenceProof(
            status="finalized",
            answer_text=xml,
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
            ),
        ),
        workspace=_workspace(),
    )

    assert markdown_reference.reference_answer.text == markdown
    assert xml_reference.reference_answer.text == xml


def test_reference_preserves_duplicate_and_unreferenced_nullable_citation_positions() -> None:
    """Future failure: reference materialization must not shift exact pointer positions."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/long-report",
            final_url="https://example.com/long-report",
            media_type="text/plain",
            content="PRIVATE HEADER establishes the decisive scope\nPUBLIC ROW Alpha 1200 " + ("x" * 120),
            fetched_bytes=190,
        )
    )
    lines = workspace.lines(source)
    workspace.register_evidence(
        claim="Alpha value",
        start_line_id=lines[1].line_id,
        end_line_id=lines[1].line_id,
    )
    validated = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha is supported [[1]]; Alpha repeats [[3]].",
            citation_evidence_ids=("E1", "missing-evidence", "E1"),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
            ),
        ),
        workspace=workspace,
    )

    citations = validated.reference_answer.citations
    assert citations is not None
    assert citations[0] is not None
    assert citations == (citations[0], None, citations[0])
    assert citations[0].excerpts
    assert all("PRIVATE HEADER" not in e.text for e in citations[0].excerpts)


@pytest.mark.parametrize(
    ("answer", "note", "citation_ids", "error"),
    [
        ("1200 [[2]]", None, ("E1",), "out of range"),
        ("1200 [[1]]", "Support [[0]]", ("E1",), "out of range"),
        ("1200 [[2]]", None, ("E1", "unknown"), "unresolved citation"),
    ],
)
def test_invalid_reference_citation_markers_cannot_pass_terminal_validation(answer, note, citation_ids, error):
    proof = ReferenceProof(
        status="finalized",
        answer_text=answer,
        note=note,
        citation_evidence_ids=citation_ids,
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(ProofStep(step_id="S1", statement="Alpha value 1200", kind="supported", evidence_ids=("E1",)),),
    )
    with pytest.raises(ProofValidationError, match=error):
        validate_and_render_reference(dossier=_dossier(), proof=proof, workspace=_workspace())
    assert error in reference_contract_defects(proof, dossier=_dossier(), workspace=_workspace())[0]


def test_structured_reference_preserves_note_in_public_response() -> None:
    """Future failure: structured answers must not drop their public explanatory note."""
    note = "The stated unit was incorrect; this is the corrected value [[1]]."
    proof = ReferenceProof(
        status="finalized",
        answer_text=None,
        citation_evidence_ids=("E1",),
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(
            ProofStep(
                step_id="S1",
                statement="Alpha has value 1200.",
                kind="supported",
                evidence_ids=("E1",),
            ),
        ),
        structured_answer_json='{"value":1200}',
        note=note,
    )

    validated = validate_and_render_reference(dossier=_structured_dossier(), proof=proof, workspace=_workspace())

    assert validated.reference_answer.note == note


def test_structured_contract_rejects_wrong_dialect_and_miner_unsubmittable_value() -> None:
    """Future failure: generated-safe shape alone must not bypass the exact public Query and Response contracts."""
    proof = ReferenceProof(
        status="finalized",
        answer_text=None,
        citation_evidence_ids=("E1",),
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(ProofStep(step_id="S1", statement="Alpha is proven.", kind="supported", evidence_ids=("E1",)),),
        structured_answer_json='{"value":1200}',
    )
    with pytest.raises(ProofValidationError, match="Draft 2020-12"):
        validate_and_render_reference(
            dossier=_structured_dossier(schema_dialect="https://json-schema.org/draft/2019-09/schema"),
            proof=proof,
            workspace=_workspace(),
        )

    string_schema = _structured_dossier().model_copy(
        update={
            "output_schema_json": '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            '"type":"object","properties":{"value":{"type":"string"}},'
            '"required":["value"],"additionalProperties":false}',
        }
    )
    oversized = proof.model_copy(update={"structured_answer_json": '{"value":"' + ("x" * 80_000) + '"}'})
    with pytest.raises(ProofValidationError, match="exceeds 80000"):
        validate_and_render_reference(dossier=string_schema, proof=oversized, workspace=_workspace())


def test_host_leaves_semantic_answer_support_to_agent_verification() -> None:
    """Semantic support cannot be approximated by comparing public and private strings."""
    validated = validate_and_render_reference(
        dossier=_dossier(answer="twelve hundred"),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha has value 1,200 [[1]].",
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Alpha has value 1,200.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
        ),
        workspace=_workspace(),
    )

    assert validated.reference_answer.text == "Alpha has value 1,200 [[1]]."


def test_reference_may_correct_value_but_not_answer_identity() -> None:
    """Future failure: reference upgrading must correct values without changing answer slots."""
    workspace = _workspace()
    corrected = ReferenceProof(
        status="finalized",
        answer_text="Alpha has value 1200 [[1]].",
        citation_evidence_ids=("E1",),
        answers=(ReferenceAnswerSelection(answer_id="A1", corrected_value="1200"),),
        proof_steps=(
            ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
        ),
    )
    wrong_identity = corrected.model_copy(update={"answers": (ReferenceAnswerSelection(answer_id="A2"),)})
    dossier = _dossier(answer="1100")

    validated = validate_and_render_reference(
        dossier=dossier,
        proof=corrected,
        workspace=workspace,
    )

    assert validated.reference_answer.text == "Alpha has value 1200 [[1]]."
    assert reference_contract_defects(
        wrong_identity,
        workspace=workspace,
        dossier=dossier,
    ) == ("reference answer IDs differ from the dossier contract",)


def test_reference_rejects_text_that_exceeds_the_public_miner_response_contract() -> None:
    """Future failure: finalized references must fit through the public miner response boundary."""
    with pytest.raises(ProofValidationError, match="public miner response contract"):
        validate_and_render_reference(
            dossier=_dossier(),
            proof=ReferenceProof(
                status="finalized",
                answer_text="x" * 80_001,
                citation_evidence_ids=("E1",),
                answers=(ReferenceAnswerSelection(answer_id="A1"),),
                proof_steps=(
                    ProofStep(
                        step_id="S1",
                        statement="Alpha has value 1200.",
                        kind="supported",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
            workspace=_workspace(),
        )


def test_public_sized_answer_and_citations_do_not_hit_a_reference_only_combined_limit() -> None:
    """Future failure: audit packing must not impose a smaller combined limit than the public contract."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/large-report",
            final_url="https://example.com/large-report",
            media_type="text/plain",
            content="x" * 25_000,
            fetched_bytes=25_000,
        )
    )
    line = workspace.lines(source)[0]
    workspace.register_evidence(
        claim="large public evidence",
        start_line_id=line.line_id,
        end_line_id=line.line_id,
    )
    suffix = " claim [[1]]."
    answer_text = ("A" * (80_000 - len(suffix))) + suffix

    validated = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text=answer_text,
            citation_evidence_ids=("E1", "E1"),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="The public evidence supports the answer.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
        ),
        workspace=workspace,
    )

    assert validated.reference_answer.text == answer_text
    assert validated.reference_answer.citations is not None
    assert len(validated.reference_answer.citations) == 2


def test_public_question_is_not_rejected_by_the_ordinary_audit_packet_target() -> None:
    """Future failure: reference audit packing must preserve the same public query miners receive."""
    question = "Q" * 128_001

    validated = validate_and_render_reference(
        dossier=_dossier(question=question),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha has value 1200 [[1]].",
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Alpha has value 1200.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
        ),
        workspace=_workspace(),
    )
    assert validated.reference_answer.text == "Alpha has value 1200 [[1]]."
