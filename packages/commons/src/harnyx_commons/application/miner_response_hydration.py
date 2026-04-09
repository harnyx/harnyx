"""Shared miner response payload validation and citation hydration."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.miner_task import AnswerCitation, Response
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_call import SearchToolResult, ToolCall, ToolResultPolicy
from harnyx_commons.tools.types import is_citation_source

_MAX_CITATION_REFS = 50


class _CitationRefPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    receipt_id: str
    result_id: str


class _RawMinerResponsePayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    text: str
    citations: list[_CitationRefPayload] | None = Field(default=None, max_length=_MAX_CITATION_REFS)


def hydrate_miner_response_payload(
    payload: object,
    *,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> Response:
    raw_response = _RawMinerResponsePayload.model_validate(payload, strict=True)
    citations = _hydrate_citations(
        tuple(raw_response.citations or ()),
        session_id=session_id,
        receipt_log=receipt_log,
    )
    return Response(text=raw_response.text, citations=citations or None)


def _hydrate_citations(
    citation_refs: tuple[_CitationRefPayload, ...],
    *,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> tuple[AnswerCitation, ...]:
    hydrated: list[AnswerCitation] = []
    for citation_ref in citation_refs:
        citation = _hydrate_citation(citation_ref, session_id=session_id, receipt_log=receipt_log)
        if citation is not None:
            hydrated.append(citation)
    return tuple(hydrated)


def _hydrate_citation(
    citation_ref: _CitationRefPayload,
    *,
    session_id: UUID,
    receipt_log: ReceiptLogPort,
) -> AnswerCitation | None:
    receipt = receipt_log.lookup(citation_ref.receipt_id)
    if receipt is None or receipt.session_id != session_id:
        return None
    result = _lookup_referenceable_result(receipt, citation_ref.result_id)
    if result is None:
        return None
    return AnswerCitation(url=result.url, note=result.note, title=result.title)


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


__all__ = ["hydrate_miner_response_payload"]
