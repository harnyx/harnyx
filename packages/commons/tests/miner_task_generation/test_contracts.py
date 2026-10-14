import pytest
from pydantic import ValidationError


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


@pytest.mark.parametrize("text", ['{"x":1e999}', '{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_nonfinite_or_duplicate_json_cannot_be_marked_format_valid(text):
    from harnyx_commons.miner_task_generation.output_schema import format_assessment, strict_json

    with pytest.raises(ValueError):
        strict_json(text)
    assert not format_assessment(text, {"type": "object", "properties": {"x": {"type": "number"}}})["valid"]


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
