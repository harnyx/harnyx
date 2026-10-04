import pytest
from pydantic import ValidationError

from harnyx_commons.miner_task_generation.contracts import (
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    Seed,
)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"x": {"$ref": "#/$defs/missing"}}},
        {"type": "object", "properties": {"x": {"$ref": "https://example.invalid/schema.json"}}},
        {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
    ],
)
@pytest.mark.parametrize("answer", ["{}", '{"x":1}'])
def test_generation_rejects_invalid_public_schema_even_in_unused_branches_without_network(monkeypatch, schema, answer):
    """Drafts must share the SDK contract and never perform implicit remote retrieval."""
    import json

    from harnyx_commons.miner_task_generation.output_schema import format_assessment, validate_structured_draft

    def forbidden_network(*args, **kwargs):
        pytest.fail("Schema validation attempted a network lookup")

    monkeypatch.setattr("urllib.request.urlopen", forbidden_network)
    with pytest.raises(ValueError):
        validate_structured_draft(json.dumps(schema), answer)
    with pytest.raises(ValueError):
        format_assessment(answer, schema)


def test_generation_supports_sdk_local_references_and_composition():
    import json

    from harnyx_commons.miner_task_generation.output_schema import format_assessment, validate_structured_draft

    schema = {
        "type": "object",
        "$defs": {"value": {"type": "integer"}},
        "allOf": [{"properties": {"x": {"$ref": "#/$defs/value"}}, "required": ["x"]}],
    }
    assert validate_structured_draft(json.dumps(schema), '{"x":1}') == (schema, {"x": 1})
    assert format_assessment('{"x":1}', schema)["valid"]
    assert format_assessment('{"x":"wrong"}', schema)["kind"] == "schema_violation"


@pytest.mark.parametrize("field", ["seed_id", "seed_question", "reference_answer", "scope"])
def test_seed_cannot_admit_blank_research_content(field):
    """A seed must supply usable research content before paid candidate work."""
    values = {
        "seed_id": "city",
        "seed_question": "Which city?",
        "reference_answer": "Paris",
        "scope": "named report",
        "event_date": "2026-10-01",
        "supporting_sources": [
            {
                "url": "https://example.org/report",
                "evidence": "Paris",
                "publication_date": "2026-10-01",
            }
        ],
    }
    with pytest.raises(ValidationError):
        Seed.model_validate(values | {field: " \t\n"})


@pytest.mark.parametrize("text", ['{"x":1e999}', '{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_nonfinite_or_duplicate_json_cannot_be_marked_format_valid(text):
    from harnyx_commons.miner_task_generation.output_schema import format_assessment, strict_json

    with pytest.raises(ValueError):
        strict_json(text)
    assert not format_assessment(text, {"type": "object", "properties": {"x": {"type": "number"}}})["valid"]


def test_reference_proof_enforces_public_citation_position_limit() -> None:
    """Future failure: accepted references must never be silently truncated by the 200-position judge boundary."""
    common = {
        "status": "finalized",
        "answer_text": "Alpha is the published result [[1]].",
        "answers": (ReferenceAnswerSelection(answer_id="A1"),),
        "proof_steps": (
            ProofStep(step_id="S1", statement="Alpha is published.", kind="supported", evidence_ids=("E1",)),
        ),
    }

    accepted = ReferenceProof(**common, citation_evidence_ids=tuple("E1" for _ in range(200)))

    assert len(accepted.citation_evidence_ids) == 200
    with pytest.raises(ValidationError, match="at most 200"):
        ReferenceProof(**common, citation_evidence_ids=tuple("E1" for _ in range(201)))


@pytest.mark.parametrize(
    ("answer_text", "structured_answer_json"), [("Alpha [[1]]", None), (None, '{"answer":"Alpha"}')]
)
def test_finalized_reference_proof_requires_a_public_citation_position(
    answer_text: str | None,
    structured_answer_json: str | None,
) -> None:
    """Future failure: finalized public answers must not be accepted with only private proof evidence."""
    with pytest.raises(ValidationError, match="at least one public citation position"):
        ReferenceProof(
            status="finalized",
            answer_text=answer_text,
            citation_evidence_ids=(),
            answers=(ReferenceAnswerSelection(answer_id="A1"),),
            proof_steps=(
                ProofStep(
                    step_id="S1",
                    statement="Alpha is published.",
                    kind="supported",
                    evidence_ids=("E1",),
                ),
            ),
            structured_answer_json=structured_answer_json,
        )

    giveup = ReferenceProof(
        status="giveup",
        answer_text=None,
        citation_evidence_ids=(),
        structured_answer_json=None,
        giveup_reason="No public source establishes the required answer.",
    )
    assert giveup.status == "giveup"


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", " "),
        ("answer", "\t"),
        ("scope", " "),
        ("revision_note", " "),
        ("answer_components", [""]),
        ("answer_components", ["valid", " "]),
    ],
)
def test_draft_rejects_blank_public_and_grading_fields_before_agent_work(field, value):
    from harnyx_commons.miner_task_generation.contracts import QuestionDraft

    values = dict(
        question="Question?",
        answer="Answer",
        answer_components=["Answer"],
        scope="Fixed",
        source_support=[{"url": "https://example.org", "evidence": "Source"}],
        revision_note="Current",
    )
    values[field] = value
    with pytest.raises(ValidationError):
        QuestionDraft(**values)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ('{"value":true}', '{"value":1}', False),
        ('{"nested":[false]}', '{"nested":[0]}', False),
        ('{"a":1,"b":true}', '{"b":true,"a":1.0}', True),
        ('{"value":1}', '{"value":"1"}', False),
    ],
)
def test_reference_json_equality_preserves_types_without_object_order_or_number_format_changes(left, right, expected):
    from harnyx_commons.miner_task_generation.output_schema import same_json_value

    assert same_json_value(left, right) is expected
