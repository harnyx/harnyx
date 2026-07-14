"""Validation helpers for miner entrypoint structured output."""

from __future__ import annotations

import json
from urllib.parse import unquote

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

from harnyx_miner_sdk.json_types import JsonObject, JsonValue

MAX_STRUCTURED_JSON_CHARS = 80_000
JSON_SCHEMA_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_OUTPUT_SCHEMA_BASE_URI = "urn:harnyx:output-schema"


def compact_json(value: JsonValue) -> str:
    """Render a JSON value deterministically without changing string content."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON") from exc
    return rendered


def validate_output_schema(schema: JsonObject) -> JsonObject:
    rendered = compact_json(schema)
    if len(rendered) > MAX_STRUCTURED_JSON_CHARS:
        raise ValueError("output schema exceeds 80000 compact JSON characters")

    dialect = schema.get("$schema")
    if dialect is not None and dialect != JSON_SCHEMA_DRAFT_2020_12:
        raise ValueError("output schema must use JSON Schema Draft 2020-12")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid output schema: {exc.message}") from exc
    _build_schema_registry(schema)
    return schema


def validate_output_size(output: JsonValue) -> JsonValue:
    if len(compact_json(output)) > MAX_STRUCTURED_JSON_CHARS:
        raise ValueError("response output exceeds 80000 compact JSON characters")
    return output


def validate_output_against_schema(output: JsonValue, schema: JsonObject) -> None:
    """Validate output at trusted host ingress, never inside the miner sandbox."""

    try:
        registry = _build_schema_registry(schema)
        Draft202012Validator(schema, registry=registry).validate(output)
    except (ValidationError, Unresolvable) as exc:
        raise ValueError(f"response output does not match output schema: {exc}") from exc


def _build_schema_registry(schema: JsonObject) -> Registry[JsonValue]:
    """Build an offline registry and resolve references at schema positions."""

    root = Resource.from_contents(schema, default_specification=DRAFT202012)
    registry: Registry[JsonValue] = (
        Registry().with_resource(_OUTPUT_SCHEMA_BASE_URI, root).crawl()
    )
    pending = [(root, registry.resolver(_OUTPUT_SCHEMA_BASE_URI).in_subresource(root))]
    while pending:
        resource, resolver = pending.pop()
        contents = resource.contents
        if isinstance(contents, dict):
            for keyword in ("$ref", "$dynamicRef"):
                reference = contents.get(keyword)
                if not isinstance(reference, str):
                    continue
                if not reference.startswith("#"):
                    raise ValueError("output schema references must be local fragments")
                document_root = resolver.lookup("#").contents
                _validate_json_pointer_array_indices(document_root, reference)
                try:
                    resolver.lookup(reference)
                except Unresolvable as exc:
                    raise ValueError(
                        f"output schema reference does not resolve: {reference}"
                    ) from exc
        for subresource in resource.subresources():
            pending.append((subresource, resolver.in_subresource(subresource)))
    return registry


def _validate_json_pointer_array_indices(contents: JsonValue, reference: str) -> None:
    """Enforce RFC 6901 array indices before `referencing` resolves the pointer."""

    fragment = unquote(reference[1:])
    if not fragment.startswith("/"):
        return
    current = contents
    for raw_token in fragment[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not token.isascii() or not token.isdecimal() or (
                len(token) > 1 and token.startswith("0")
            ):
                raise ValueError(f"output schema reference does not resolve: {reference}")
            index = int(token)
            if index >= len(current):
                return
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                return
            current = current[token]
        else:
            return


__all__ = [
    "JSON_SCHEMA_DRAFT_2020_12",
    "MAX_STRUCTURED_JSON_CHARS",
    "compact_json",
    "validate_output_against_schema",
    "validate_output_schema",
    "validate_output_size",
]
