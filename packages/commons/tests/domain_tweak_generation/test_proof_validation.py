import pytest

from harnyx_commons.domain_tweak_generation import (
    DossierAnswer,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    SourceDocument,
    SourceWorkspace,
)
from harnyx_commons.domain_tweak_generation.proof_validation import (
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


def test_host_renders_citation_markers() -> None:
    """Future failure: marker syntax belongs to the host, not the model."""
    validated = validate_and_render_reference(
        question="Which value?",
        dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
        proof=ReferenceProof(
            status="finalized",
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
    assert validated.reference_answer.citations is not None


def test_public_reference_contains_only_raw_miner_projection_not_private_semantics() -> None:
    """Future failure: model-authored claims and audit annotations must never enter judge-visible citations."""
    validated = validate_and_render_reference(
        question="Which value?",
        dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
        proof=ReferenceProof(
            status="finalized",
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Private authored claim: Alpha has value 1200.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
        ),
        workspace=_workspace(),
    )

    citations = validated.reference_answer.citations
    assert citations is not None
    assert citations[0].title is None
    assert citations[0].note == "[slice 0:33]\nHEADER\tName\tValue\nROW\tAlpha\t1,200"
    assert "Private authored claim" not in citations[0].note
    assert "Claim:" not in citations[0].note
    assert "Supports:" not in citations[0].note
    assert "Verified excerpts:" not in citations[0].note


def test_host_leaves_semantic_answer_support_to_the_independent_audit() -> None:
    """Future failure: host validation must not replace semantic audit with substring matching."""
    validated = validate_and_render_reference(
        question="Which value?",
        dossier_answers=(DossierAnswer(answer_id="A1", value="twelve hundred"),),
        proof=ReferenceProof(
            status="finalized",
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

    assert validated.audit_packet["canonical_short_answers"] == ["twelve hundred"]
    assert validated.audit_packet["proof_steps"][0]["statement"] == "Alpha has value 1,200."


def test_model_authored_marker_is_rejected() -> None:
    with pytest.raises(ProofValidationError, match="model-authored"):
        validate_and_render_reference(
            question="Which value?",
            dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
            proof=ReferenceProof(
                status="finalized",
                answers=(ReferenceAnswerSelection(answer_id="A1"),),
                proof_steps=(
                    ProofStep(
                        step_id="S1",
                        statement="Alpha has value 1200 [[1]].",
                        kind="supported",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
            workspace=_workspace(),
        )


def test_reference_may_correct_value_but_not_answer_identity() -> None:
    """Future failure: reference upgrading must correct values without changing answer slots."""
    workspace = _workspace()
    corrected = ReferenceProof(
        status="finalized",
        answers=(ReferenceAnswerSelection(answer_id="A1", corrected_value="1200"),),
        proof_steps=(
            ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
        ),
    )
    wrong_identity = corrected.model_copy(update={"answers": (ReferenceAnswerSelection(answer_id="A2"),)})
    dossier_answers = (DossierAnswer(answer_id="A1", value="1100"),)

    validated = validate_and_render_reference(
        question="Which value?",
        dossier_answers=dossier_answers,
        proof=corrected,
        workspace=workspace,
    )

    assert "Short answers: 1200" in validated.reference_answer.text
    assert reference_contract_defects(
        wrong_identity,
        workspace=workspace,
        dossier_answers=dossier_answers,
    ) == ("reference answer IDs differ from the dossier contract",)


def test_reference_rejects_text_that_exceeds_the_public_miner_response_contract() -> None:
    """Future failure: finalized references must fit through the public miner response boundary."""
    with pytest.raises(ProofValidationError, match="public miner response contract"):
        validate_and_render_reference(
            question="Which value?",
            dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
            proof=ReferenceProof(
                status="finalized",
                answers=(ReferenceAnswerSelection(answer_id="A1", corrected_value="x" * 80_000),),
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


def test_unfit_nontrimmable_packet_envelope_is_a_proof_validation_error() -> None:
    """Future failure: model-driven packet size must stay candidate-local instead of aborting the batch."""
    with pytest.raises(ProofValidationError, match="required proof packet envelope"):
        validate_and_render_reference(
            question="Q" * 128_001,
            dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
            proof=ReferenceProof(
                status="finalized",
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


def test_unrelated_workspace_value_error_is_not_reclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: size handling must not hide unrelated workspace defects as model feedback."""

    def raise_unrelated_value_error(self: SourceWorkspace, **_kwargs: object) -> dict[str, object]:
        raise ValueError("unrelated workspace defect")

    monkeypatch.setattr(SourceWorkspace, "proof_packet", raise_unrelated_value_error)

    with pytest.raises(ValueError, match="unrelated workspace defect"):
        validate_and_render_reference(
            question="Which value?",
            dossier_answers=(DossierAnswer(answer_id="A1", value="1200"),),
            proof=ReferenceProof(
                status="finalized",
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
