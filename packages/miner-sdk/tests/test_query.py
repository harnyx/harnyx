from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.query import CitationRef, CitationSlice, Query, Response


def test_query_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "   "})


def test_query_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "hello", "other": "nope"})


def test_query_accepts_draft_2020_12_schema_and_preserves_nested_strings() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"answer": {"type": "string", "const": "  exact  "}},
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    query = Query(text=" question ", output_schema=schema)

    assert query.text == "question"
    assert query.output_schema == schema


def test_query_treats_ref_key_inside_const_as_instance_data() -> None:
    schema = {"const": {"$ref": "https://example.com/value"}}

    assert Query(text="question", output_schema=schema).output_schema == schema


def test_query_resolves_local_reference_from_nested_id_resource() -> None:
    schema = {
        "$defs": {
            "inner": {
                "$id": "urn:harnyx:inner",
                "$defs": {"answer": {"type": "string"}},
                "$ref": "#/$defs/answer",
            }
        },
        "$ref": "#/$defs/inner",
    }

    assert Query(text="question", output_schema=schema).output_schema == schema


@pytest.mark.parametrize("payload", [{}, {"output_schema": None}])
def test_missing_and_null_schema_select_legacy_mode(payload: dict[str, object]) -> None:
    query = Query.model_validate({"text": "question", **payload})

    assert query.output_schema is None
    assert query.model_dump(mode="json") == {"text": "question"}


def test_empty_schema_selects_structured_mode() -> None:
    assert Query(text="question", output_schema={}).output_schema == {}


@pytest.mark.parametrize(
    "schema",
    [
        {"$schema": "http://json-schema.org/draft-07/schema#"},
        {"$ref": "https://example.com/schema.json"},
        {"$dynamicRef": "https://example.com/schema.json#node"},
        {"$ref": "#/$defs/missing"},
        {
            "$defs": {
                "inner": {
                    "$id": "urn:harnyx:inner",
                    "$anchor": "node",
                    "type": "string",
                }
            },
            "$ref": "#node",
        },
        {"prefixItems": [{"type": "string"}], "$ref": "#/prefixItems/-1"},
    ],
)
def test_query_rejects_wrong_dialect_and_external_references(schema: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Query(text="question", output_schema=schema)


def test_query_rejects_schema_over_eighty_thousand_compact_json_chars() -> None:
    with pytest.raises(ValidationError):
        Query(text="question", output_schema={"description": "x" * 80_001})


@pytest.mark.parametrize(
    "schema",
    [
        {
            "properties": {"answer": {"$ref": "#/prefixItems/-1"}},
            "prefixItems": [{"type": "string"}],
        },
        {
            "$defs": {
                "inner": {
                    "$id": "urn:harnyx:inner",
                    "properties": {"answer": {"$ref": "#/prefixItems/-1"}},
                    "prefixItems": [{"type": "string"}],
                }
            },
            "$ref": "#/$defs/inner",
        },
    ],
)
def test_query_rejects_negative_array_pointer_from_active_resource(
    schema: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Query(text="question", output_schema=schema)


def test_response_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate({"text": ""})


@pytest.mark.parametrize("output", [False, 0, "", [], {}])
def test_response_accepts_every_non_null_json_value(output: object) -> None:
    response = Response(output=output)

    assert response.output == output
    assert response.text is None


@pytest.mark.parametrize("payload", [{}, {"output": None}, {"text": "answer", "output": {} }])
def test_response_requires_exactly_one_non_null_answer(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(payload)


def test_response_preserves_nested_null_and_whitespace_sensitive_strings() -> None:
    output = {"answer": [None, "  exact  "]}

    assert Response(output=output).output == output


def test_response_rejects_output_over_eighty_thousand_compact_json_chars() -> None:
    with pytest.raises(ValidationError):
        Response(output={"answer": "x" * 80_001})


def test_response_accepts_optional_citation_refs() -> None:
    response = Response.model_validate(
        {
            "text": "hello",
            "citations": [{"receipt_id": "receipt-1", "result_id": "result-1"}],
        }
    )

    assert response == Response(
        text="hello",
        citations=[CitationRef(receipt_id="receipt-1", result_id="result-1")],
    )
    assert response.citations is not None
    assert response.citations[0].slices == []


def test_response_accepts_targeted_citation_slices() -> None:
    response = Response.model_validate(
        {
            "text": "hello",
            "citations": [
                {
                    "receipt_id": "receipt-1",
                    "result_id": "result-1",
                    "slices": [{"start": 0, "end": 120}],
                }
            ],
        }
    )

    assert response == Response(
        text="hello",
        citations=[
            CitationRef(
                receipt_id="receipt-1",
                result_id="result-1",
                slices=[CitationSlice(start=0, end=120)],
            )
        ],
    )


def test_response_rejects_more_than_two_hundred_citations() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(
            {
                "text": "hello",
                "citations": [
                    {"receipt_id": f"receipt-{index}", "result_id": f"result-{index}"}
                    for index in range(201)
                ],
            }
        )


def test_response_rejects_more_than_four_hundred_materialized_segments() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate(
            {
                "text": "hello",
                "citations": [
                    {
                        "receipt_id": "receipt-1",
                        "result_id": "result-1",
                        "slices": [{"start": index * 100, "end": (index + 1) * 100} for index in range(401)],
                    }
                ],
            }
        )


def test_response_rejects_text_longer_than_eighty_thousand_chars() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate({"text": "x" * 80_001})


def test_citation_slice_requires_end_after_start() -> None:
    with pytest.raises(ValidationError):
        CitationSlice(start=10, end=10)
