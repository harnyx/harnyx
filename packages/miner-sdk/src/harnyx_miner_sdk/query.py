"""Query request/response contracts for miners."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from harnyx_miner_sdk.json_types import JsonObject, JsonValue
from harnyx_miner_sdk.structured_output import validate_output_schema, validate_output_size

_MINER_SDK_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    json_schema_mode_override="validation",
    strict=True,
    str_strip_whitespace=False,
)
_MAX_RESPONSE_CHARS = 80_000
_MAX_RESPONSE_CITATIONS = 200
_MAX_RESPONSE_EVIDENCE_SEGMENTS = 400


class Query(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str = Field(min_length=1)
    output_schema: JsonObject | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query text must not be blank")
        return stripped

    @field_validator("output_schema")
    @classmethod
    def validate_schema(cls, value: JsonObject | None) -> JsonObject | None:
        if value is None:
            return None
        return validate_output_schema(value)


class CitationSlice(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end <= self.start:
            raise ValueError("citation slice end must be greater than start")
        return self


class CitationRef(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    receipt_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    slices: list[CitationSlice] = Field(default_factory=list)


class Response(BaseModel):
    model_config = _MINER_SDK_STRICT_CONFIG

    text: str | None = Field(
        default=None,
        max_length=_MAX_RESPONSE_CHARS,
        exclude_if=lambda value: value is None,
    )
    output: JsonValue | None = Field(default=None, exclude_if=lambda value: value is None)
    citations: list[CitationRef] | None = Field(default=None, max_length=_MAX_RESPONSE_CITATIONS)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("response text must not be blank")
        return stripped

    @field_validator("output")
    @classmethod
    def validate_output(cls, value: JsonValue | None) -> JsonValue | None:
        if value is None:
            return None
        return validate_output_size(value)

    @model_validator(mode="after")
    def validate_total_evidence_segments(self) -> Self:
        if (self.text is None) == (self.output is None):
            raise ValueError("response must include exactly one non-null answer field")
        total_segments = sum(len(citation.slices) if citation.slices else 1 for citation in self.citations or ())
        if total_segments > _MAX_RESPONSE_EVIDENCE_SEGMENTS:
            raise ValueError("response citations exceed 400 materialized evidence segments")
        return self


__all__ = ["CitationRef", "CitationSlice", "Query", "Response"]
