import json

import pytest

from harnyx_commons.domain_tweak_generation import (
    DossierAnswer,
    DossierFact,
    DossierRequirement,
    GroundedQuestionDossier,
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
    validate_structured_payload,
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


def _dossier(*, answer: str = "1200", question: str = "Which value?") -> GroundedQuestionDossier:
    return GroundedQuestionDossier(
        status="ready",
        subject="Published value",
        route_summary="Read the published value",
        question=question,
        answers=(DossierAnswer(answer_id="A1", value=answer),),
        requirements=(DossierRequirement(description="Read the published value"),),
        source_facts=(DossierFact(statement="Alpha has a value", evidence_ids=("E1",)),),
        derivation="Select the published value",
        why_not_one_page="The roster identity and value occupy separate records",
        substantive_final_condition="The value record determines the answer",
        response_mode="plain_text",
    )


def _structured_dossier(
    *, schema_dialect: str = "https://json-schema.org/draft/2020-12/schema"
) -> GroundedQuestionDossier:
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
    assert validated.reference_answer.citations is not None


@pytest.mark.parametrize(
    ("dossier_answer", "corrected_value", "proof_statement"),
    (
        ("1200", "Alpha [[1]]", "Alpha has value 1200."),
        ("1200", None, "Alpha has value 1200 [[ 1 ]]."),
        ("1200 [[1]]", None, "Alpha has value 1200."),
    ),
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


def test_structured_reference_uses_rich_field_contract_without_polluting_atomic_fields() -> None:
    """Future failure: only an explicitly citation-bearing prose field may carry a marker."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Published result",'
        '"description":"Return the exact researched result.","type":"object","properties":{'
        '"candidate":{"type":"string","description":"Atomic identifier exactly as printed.",'
        '"minLength":1,"maxLength":100},'
        '"explanation":{"type":"string","description":"Explain the material result and cite it with [[n]].",'
        '"minLength":1,"maxLength":500},'
        '"score":{"type":"integer","description":"Atomic published score; no citation syntax."}},'
        '"required":["candidate","explanation","score"],'
        '"additionalProperties":false}'
    )
    dossier = _dossier(question="Return the requested object.").model_copy(
        update={
            "response_mode": "structured",
            "output_schema_json": schema,
            "structured_answer_json": '{"candidate":"Alpha","explanation":"Alpha is supported [[1]].","score":1200}',
        }
    )
    validated = validate_and_render_reference(
        dossier=dossier,
        proof=ReferenceProof(
            status="finalized",
            answer_text=None,
            citation_evidence_ids=("E1",),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(step_id="S1", statement="Alpha has value 1200.", kind="supported", evidence_ids=("E1",)),
            ),
            structured_answer_json=('{"candidate":"Alpha","explanation":"Alpha is supported [[1]].","score":1200}'),
        ),
        workspace=_workspace(),
    )

    assert validated.output_schema is not None
    assert validated.output_schema["properties"] == {
        "candidate": {
            "type": "string",
            "description": "Atomic identifier exactly as printed.",
            "minLength": 1,
            "maxLength": 100,
        },
        "explanation": {
            "type": "string",
            "description": "Explain the material result and cite it with [[n]].",
            "minLength": 1,
            "maxLength": 500,
        },
        "score": {
            "type": "integer",
            "description": "Atomic published score; no citation syntax.",
        },
    }
    assert validated.reference_answer.text == (
        '{"candidate":"Alpha","explanation":"Alpha is supported [[1]].","score":1200}'
    )


def test_generated_schema_preserves_every_supported_constraint_family() -> None:
    """Future failure: schema projection must not silently discard an admitted constraint family."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","title":"Bounded result",'
        '"description":"Exercise every supported constraint family.","type":"object","properties":{'
        '"label":{"type":"string","title":"Label","description":"Exact label.",'
        '"minLength":1,"maxLength":10},'
        '"ratio":{"type":"number","description":"Atomic ratio."},'
        '"flags":{"type":"array","description":"Atomic flags.","minItems":1,"maxItems":2,'
        '"items":{"type":"boolean","description":"Atomic flag."}}},'
        '"required":["label","ratio","flags"],'
        '"additionalProperties":false}'
    )

    parsed_schema, parsed_output = validate_structured_payload(
        schema,
        '{"label":"Alpha","ratio":1.5,"flags":[true]}',
    )

    assert parsed_schema["description"] == "Exercise every supported constraint family."
    assert parsed_schema["properties"]["flags"]["maxItems"] == 2
    assert parsed_output == {"label": "Alpha", "ratio": 1.5, "flags": [True]}


def test_generated_schema_rejects_unbounded_regex_patterns() -> None:
    """Future failure: model-authored regex must not enter unbounded response validation."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        '"value":{"type":"string","pattern":"^(a+)+$"}},"required":["value"],'
        '"additionalProperties":false}'
    )

    with pytest.raises(ProofValidationError, match="unsafe keywords.*pattern"):
        validate_structured_payload(schema, '{"value":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!"}')


@pytest.mark.parametrize(
    ("property_schema", "answer_json"),
    [
        ('{"type":"string","const":"Alpha"}', '"Alpha"'),
        ('{"type":"string","enum":["Alpha"]}', '"Alpha"'),
        ('{"type":"number","minimum":1.5,"maximum":1.5}', "1.5"),
        ('{"type":"integer","minimum":1200,"exclusiveMaximum":1201}', "1200"),
        ('{"type":"integer","enum":[1200,999],"minimum":1200}', "1200"),
        ('{"type":"string","enum":["Alpha","Beta"],"minLength":5}', '"Alpha"'),
        ('{"type":"string","minLength":5,"maxLength":5}', '"Alpha"'),
        ('{"type":"array","minItems":3,"maxItems":3,"items":{"type":"integer"}}', "[1,2,3]"),
    ],
)
def test_generated_schema_rejects_constraints_that_disclose_the_answer(
    property_schema: str,
    answer_json: str,
) -> None:
    """Future failure: public generated schemas must not encode the privately proved answer."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        f'"value":{property_schema}}},"required":["value"],'
        '"additionalProperties":false}'
    )

    with pytest.raises(ProofValidationError, match="must not disclose the answer"):
        validate_structured_payload(schema, f'{{"value":{answer_json}}}')


@pytest.mark.parametrize(
    ("property_schema", "answer_json"),
    [
        ('{"type":"string","maxLength":1200}', '"opaque"'),
        ('{"type":"array","minItems":3,"maxItems":4,"items":{"type":"string"}}', '["a","b","c"]'),
    ],
)
def test_generated_schema_allows_non_pinning_constraint_values_that_coincide_with_an_answer(
    property_schema: str,
    answer_json: str,
) -> None:
    """Future failure: an unrelated structural bound must not be mistaken for answer disclosure."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        f'"value":{property_schema}}},"required":["value"],'
        '"additionalProperties":false}'
    )
    _, answer = validate_structured_payload(
        schema,
        f'{{"value":{answer_json}}}',
    )

    assert answer is not None


def test_generated_schema_rejects_integral_float_constraint_metadata() -> None:
    """Future failure: generated length constraints must use JSON integers without numeric aliases."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        '"value":{"type":"string","maxLength":1200.0}},"required":["value"],'
        '"additionalProperties":false}'
    )

    with pytest.raises(ProofValidationError, match="JSON integer"):
        validate_structured_payload(
            schema,
            '{"value":"opaque"}',
        )


@pytest.mark.parametrize(
    ("property_name", "property_schema", "answer_json"),
    [
        (
            "eligible",
            '{"type":"boolean","description":"True if the record meets the published eligibility rule."}',
            "true",
        ),
        (
            "selectedalpha",
            '{"type":"string","description":"The selected entry is Alpha."}',
            '"Alpha"',
        ),
        (
            "amount",
            '{"type":"integer","title":"Published 1,200 amount"}',
            "1200",
        ),
    ],
)
def test_deterministic_validation_leaves_schema_annotation_semantics_to_audit(
    property_name: str,
    property_schema: str,
    answer_json: str,
) -> None:
    """Future failure: a lexical answer matcher must not return to deterministic schema validation."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        f'"{property_name}":{property_schema}}},'
        f'"required":["{property_name}"],"additionalProperties":false}}'
    )

    _, answer = validate_structured_payload(schema, f'{{"{property_name}":{answer_json}}}')

    assert answer == {property_name: json.loads(answer_json)}


def test_generated_schema_allows_unrelated_constraint_equal_to_another_field_value() -> None:
    """Future failure: structural metadata must retain field provenance instead of matching globally."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        '"count":{"type":"integer"},"label":{"type":"string","minLength":1}},'
        '"required":["count","label"],"additionalProperties":false}'
    )

    _, answer = validate_structured_payload(schema, '{"count":1,"label":"x"}')

    assert answer == {"count": 1, "label": "x"}


@pytest.mark.parametrize(
    ("property_schema", "answer_json"),
    [
        ('{"type":"string","maxLength":0}', '""'),
        ('{"type":"array","maxItems":0,"items":{"type":"string"}}', "[]"),
    ],
)
def test_generated_schema_rejects_zero_upper_bound_that_pins_empty_output(
    property_schema: str,
    answer_json: str,
) -> None:
    """Future failure: a zero upper bound must not publish an empty canonical answer."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        '"value":' + property_schema + '},"required":["value"],"additionalProperties":false}'
    )

    with pytest.raises(ProofValidationError, match="must not disclose the answer"):
        validate_structured_payload(schema, f'{{"value":{answer_json}}}')


def test_generated_schema_rejects_unique_items_before_nested_output_validation() -> None:
    """Future failure: generated array constraints must not add unbounded equality work per response."""
    schema = (
        '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{'
        '"rows":{"type":"array","uniqueItems":true,"items":{"type":"object","properties":{'
        '"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}}},'
        '"required":["rows"],"additionalProperties":false}'
    )

    with pytest.raises(ProofValidationError, match="extra=.*uniqueItems"):
        validate_structured_payload(schema, '{"rows":[{"value":1}]}')


def test_reference_preserves_duplicate_and_unresolved_citation_positions_for_audit() -> None:
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
            answer_text="Alpha is supported [[1]]; unresolved evidence stays [[2]]; Alpha repeats [[3]].",
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
    assert validated.audit_packet["validated_citations"] == [
        citations[0].model_dump(mode="json", exclude_none=True),
        None,
        citations[0].model_dump(mode="json", exclude_none=True),
    ]
    assert citations[0].note is not None
    assert "PRIVATE HEADER" not in citations[0].note
    assert "PRIVATE HEADER" in str(validated.audit_packet["selected_evidence"])


def test_generated_schema_rejects_untraversed_subschema_applicators() -> None:
    """Future failure: rich annotations must not let nested schemas evade generation bounds."""
    dossier = _structured_dossier().model_copy(
        update={
            "output_schema_json": (
                '{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",'
                '"properties":{"value":{"type":"integer","allOf":[{"minimum":0}]}},'
                '"required":["value"],"additionalProperties":false}'
            )
        }
    )
    proof = ReferenceProof(
        status="finalized",
        answer_text=None,
        citation_evidence_ids=("E1",),
        answers=(ReferenceAnswerSelection(answer_id="A1"),),
        proof_steps=(ProofStep(step_id="S1", statement="Alpha is proven.", kind="supported", evidence_ids=("E1",)),),
        structured_answer_json='{"value":1200}',
    )

    with pytest.raises(ProofValidationError, match="unsafe keywords.*allOf"):
        validate_and_render_reference(dossier=dossier, proof=proof, workspace=_workspace())


def test_structured_reference_uses_fixed_schema_canonical_json_and_complete_audit_envelope() -> None:
    """Future failure: structured reference values must reach the public task without losing audit semantics."""
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
    )

    validated = validate_and_render_reference(dossier=_structured_dossier(), proof=proof, workspace=_workspace())

    assert validated.output_schema is not None
    assert validated.reference_answer.text == '{"value":1200}'
    assert validated.audit_packet["response_mode"] == "structured"
    assert validated.audit_packet["output_schema"] == validated.output_schema
    assert validated.audit_packet["structured_answer"] == {"value": 1200}
    assert validated.reference_answer.citations


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


def test_structured_contract_normalizes_json_numeric_limit_as_proof_error() -> None:
    """Future failure: valid JSON rejected by the host parser must remain a candidate-local proof defect."""
    structured_answer_json = '{"value":' + ("9" * 5_000) + "}"

    with pytest.raises(ProofValidationError, match="could not be parsed"):
        validate_structured_payload(_structured_dossier().output_schema_json, structured_answer_json)


def test_public_reference_contains_only_raw_miner_projection_not_private_semantics() -> None:
    """Future failure: model-authored claims and audit annotations must never enter judge-visible citations."""
    validated = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha has value 1200 [[1]].",
            citation_evidence_ids=("E1",),
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
    assert citations[0] is not None
    assert citations[0].title is None
    assert citations[0].note == "[slice 0:33]\nHEADER\tName\tValue\nROW\tAlpha\t1,200"
    assert "Private authored claim" not in citations[0].note
    assert "Claim:" not in citations[0].note
    assert "Supports:" not in citations[0].note
    assert "Verified excerpts:" not in citations[0].note


def test_host_leaves_semantic_answer_support_to_the_independent_audit() -> None:
    """Future failure: host validation must not replace semantic audit with substring matching."""
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

    assert validated.audit_packet["canonical_short_answers"] == ["twelve hundred"]
    assert validated.audit_packet["proof_steps"][0]["statement"] == "Alpha has value 1,200."


def test_pointer_defects_remain_visible_instead_of_invalidating_reference_payload() -> None:
    """Future failure: the judge, not deterministic validation, owns pointer-quality defects."""
    validated = validate_and_render_reference(
        dossier=_dossier(),
        proof=ReferenceProof(
            status="finalized",
            answer_text="Alpha has value 1200 [[9]].",
            citation_evidence_ids=("missing-evidence",),
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

    assert validated.reference_answer.text.endswith("[[9]].")
    assert validated.reference_answer.citations == (None,)


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
    assert validated.audit_packet["answer_text"] == answer_text
    assert validated.audit_packet["validated_citations"][0] == validated.audit_packet["validated_citations"][1]
    assert "audit text truncated" in str(validated.audit_packet["selected_evidence"])


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

    assert validated.audit_packet["question"] == question


def test_unrelated_workspace_value_error_is_not_reclassified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: size handling must not hide unrelated workspace defects as model feedback."""

    def raise_unrelated_value_error(self: SourceWorkspace, **_kwargs: object) -> dict[str, object]:
        raise ValueError("unrelated workspace defect")

    monkeypatch.setattr(SourceWorkspace, "proof_packet", raise_unrelated_value_error)

    with pytest.raises(ValueError, match="unrelated workspace defect"):
        validate_and_render_reference(
            dossier=_dossier(),
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
