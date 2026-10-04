"""Public schema validation; format does not establish correctness."""

from __future__ import annotations

import json
import math
from typing import Any

from harnyx_commons.json_types import JsonObject
from harnyx_miner_sdk.structured_output import validate_output_against_schema, validate_output_schema


def strict_json(text: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError(f"Non-JSON constant: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return number

    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant, parse_float=finite_float)


def validate_structured_draft(schema_json: str, answer_json: str) -> tuple[JsonObject, Any]:
    schema, answer = strict_json(schema_json), strict_json(answer_json)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Public answer schema must describe a JSON object")
    validate_output_schema(schema)
    validate_output_against_schema(answer, schema)
    return schema, answer


def format_assessment(answer: str, schema: JsonObject | None) -> dict[str, Any]:
    if schema is None:
        return {"valid": True, "kind": "plain_text", "errors": []}
    try:
        parsed = strict_json(answer)
    except (ValueError, RecursionError) as exc:
        return {"valid": False, "kind": "invalid_json", "errors": [str(exc)]}
    validate_output_schema(schema)
    try:
        validate_output_against_schema(parsed, schema)
    except ValueError as exc:
        return {"valid": False, "kind": "schema_violation", "errors": [str(exc)]}
    return {"valid": True, "kind": "valid", "errors": []}


def same_json_value(left: str, right: str) -> bool:
    """JSON numbers compare by value; booleans are a distinct JSON type."""

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return type(a) is type(b) and a == b
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    return equal(strict_json(left), strict_json(right))
