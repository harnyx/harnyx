"""Shared miner response payload validation and citation hydration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.miner_task import AnswerCitation, Query, Response
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_call import SearchToolResult, ToolCall, ToolResultPolicy
from harnyx_commons.tools.types import is_citation_source
from harnyx_miner_sdk.json_types import JsonValue
from harnyx_miner_sdk.structured_output import validate_output_against_schema, validate_output_size

_MAX_RESPONSE_CHARS = 80_000
_MAX_CITATION_REFS = 200
_MAX_EVIDENCE_SEGMENTS_PER_RESPONSE = 400
_MIN_SLICE_CHARS = 100
_MAX_TOTAL_EVIDENCE_CHARS = 120_000


class MinerResponsePayloadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _MaterializedSelection:
    text: str
    char_count: int


@dataclass(frozen=True, slots=True)
class _HydratedCitation:
    answer_citation: AnswerCitation
    source_text_chars: int


@dataclass(frozen=True, slots=True)
class _HydratedCitations:
    citations: tuple[AnswerCitation, ...]
    source_text_chars: int


class _CitationSlicePayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        if self.end <= self.start:
            raise ValueError("citation slice end must be greater than start")
        return self


class _CitationRefPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    receipt_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    slices: list[_CitationSlicePayload] = Field(default_factory=list)


class _RawMinerResponsePayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
    )

    text: str | None = Field(default=None, max_length=_MAX_RESPONSE_CHARS)
    output: JsonValue | None = None
    citations: list[_CitationRefPayload] | None = Field(default=None, max_length=_MAX_CITATION_REFS)

    @model_validator(mode="after")
    def validate_total_evidence_segments(self) -> Self:
        if (self.text is None) == (self.output is None):
            raise ValueError("response must include exactly one non-null answer field")
        if self.output is not None:
            validate_output_size(self.output)
        total_segments = sum(len(citation.slices) if citation.slices else 1 for citation in self.citations or ())
        if total_segments > _MAX_EVIDENCE_SEGMENTS_PER_RESPONSE:
            raise ValueError("response citations exceed 400 materialized evidence segments")
        return self


def hydrate_miner_response_payload(
    payload: object,
    *,
    query: Query,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> Response:
    raw_response = _RawMinerResponsePayload.model_validate(payload, strict=True)
    try:
        _validate_answer_for_query(raw_response, query)
    except ValueError as exc:
        raise MinerResponsePayloadError(str(exc)) from exc
    hydrated_citations = _hydrate_citations(
        tuple(raw_response.citations or ()),
        session_id=session_id,
        receipt_log=receipt_log,
    )
    if hydrated_citations.source_text_chars > _MAX_TOTAL_EVIDENCE_CHARS:
        raise MinerResponsePayloadError("response citations exceed 120000 materialized source-text characters")
    return Response(
        text=raw_response.text,
        output=raw_response.output,
        citations=hydrated_citations.citations or None,
    )


def _validate_answer_for_query(response: _RawMinerResponsePayload, query: Query) -> None:
    if query.output_schema is None:
        if response.text is None:
            raise ValueError("legacy query response must use text")
        if not response.text.strip():
            raise ValueError("response text must not be blank")
        return
    if response.output is None:
        raise ValueError("structured query response must use output")
    validate_output_against_schema(response.output, query.output_schema)


def _hydrate_citations(
    citation_refs: tuple[_CitationRefPayload, ...],
    *,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> _HydratedCitations:
    hydrated: list[AnswerCitation] = []
    source_text_chars = 0
    for citation_ref in citation_refs:
        hydrated_citation = _hydrate_citation(
            citation_ref,
            session_id=session_id,
            receipt_log=receipt_log,
        )
        if hydrated_citation is None:
            continue
        hydrated.append(hydrated_citation.answer_citation)
        source_text_chars += hydrated_citation.source_text_chars
    return _HydratedCitations(citations=tuple(hydrated), source_text_chars=source_text_chars)


def _hydrate_citation(
    citation_ref: _CitationRefPayload,
    *,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> _HydratedCitation | None:
    receipt = receipt_log.lookup(citation_ref.receipt_id)
    if receipt is None or receipt.session_id != session_id:
        return None
    result = _lookup_referenceable_result(receipt, citation_ref.result_id)
    if result is None:
        return None
    source_text = _require_source_text(result.note)
    materialized = _materialize_selection(source_text, tuple(citation_ref.slices))
    return _HydratedCitation(
        answer_citation=AnswerCitation(
            url=result.url,
            note=materialized.text,
            title=result.title,
        ),
        source_text_chars=materialized.char_count,
    )


def _lookup_referenceable_result(
    receipt: ToolCall,
    result_id: str,
) -> SearchToolResult | None:
    if not receipt.is_successful():
        return None
    if not is_citation_source(receipt.tool):
        return None
    if receipt.details.result_policy is not ToolResultPolicy.REFERENCEABLE:
        return None
    for result in receipt.details.results:
        if result.result_id != result_id:
            continue
        if isinstance(result, SearchToolResult):
            return result
        return None
    return None


def _require_source_text(note: str | None) -> str:
    if note is None or not note.strip():
        raise MinerResponsePayloadError("cited result has no source text")
    return note


def _materialize_selection(
    source_text: str,
    slices: tuple[_CitationSlicePayload, ...],
) -> _MaterializedSelection:
    selected_slices = slices or (_CitationSlicePayload(start=0, end=len(source_text)),)
    parts: list[str] = []
    source_text_chars = 0
    for selected_slice in selected_slices:
        _validate_slice_against_source(source_text, selected_slice)
        excerpt = source_text[selected_slice.start : selected_slice.end]
        parts.append(f"[slice {selected_slice.start}:{selected_slice.end}]\n{excerpt}")
        source_text_chars += len(excerpt)
    return _MaterializedSelection(text="\n\n".join(parts), char_count=source_text_chars)


def _validate_slice_against_source(source_text: str, selected_slice: _CitationSlicePayload) -> None:
    if selected_slice.end > len(source_text):
        raise MinerResponsePayloadError("citation slice exceeds source text length")
    slice_length = selected_slice.end - selected_slice.start
    if slice_length >= _MIN_SLICE_CHARS:
        return
    if len(source_text) < _MIN_SLICE_CHARS and selected_slice.start == 0 and selected_slice.end == len(source_text):
        return
    raise MinerResponsePayloadError("citation slice must contain at least 100 characters")


__all__ = [
    "MinerResponsePayloadError",
    "hydrate_miner_response_payload",
]
