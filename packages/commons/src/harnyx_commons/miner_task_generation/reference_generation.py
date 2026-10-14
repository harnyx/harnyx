"""Validate reference representation and public authorities without fetching source content."""

from __future__ import annotations

import asyncio

from harnyx_commons.application.miner_response_hydration import (
    MAX_CITATION_REFS,
    MAX_EVIDENCE_SEGMENTS_PER_RESPONSE,
    MAX_TOTAL_CITATION_EVIDENCE_CHARS,
    validate_citation_references,
)
from harnyx_commons.domain.miner_task import ReferenceAnswer, Response
from harnyx_commons.json_types import JsonValue
from harnyx_miner_sdk.structured_output import compact_json, validate_output_size

from .contracts import (
    AuthorReference,
    QuestionDraft,
    ReferenceValidationError,
    StructuredQuestionDraft,
    TerminalReference,
)
from .output_schema import validate_structured_draft
from .source_urls import public_addresses, validated_url


async def render_author_reference(draft: QuestionDraft, reference: AuthorReference) -> TerminalReference:
    structured_answer: JsonValue = None
    if isinstance(draft, StructuredQuestionDraft):
        _, structured_answer = validate_structured_draft(draft.output_schema_json, draft.structured_answer_json)
        validate_output_size(structured_answer)
    try:
        rendered = _render_author_reference(draft, reference, structured_answer)
        authorities = set()
        for citation in reference.citations:
            parsed = validated_url(citation.url)
            assert parsed.hostname is not None
            authorities.add((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
        for host, port in sorted(authorities):
            await asyncio.to_thread(public_addresses, host, port)
        return rendered
    except (ValueError, RecursionError) as exc:
        raise ReferenceValidationError(str(exc)) from exc


def _render_author_reference(
    draft: QuestionDraft, reference: AuthorReference, structured_answer: JsonValue
) -> TerminalReference:
    citations = tuple(reference.citations)
    if len(citations) > MAX_CITATION_REFS:
        raise ValueError("Reference exceeds the existing 200-citation limit")
    if sum(max(1, len(citation.excerpts)) for citation in citations) > MAX_EVIDENCE_SEGMENTS_PER_RESPONSE:
        raise ValueError("Reference exceeds the existing 400-evidence-segment limit")
    if (
        sum(len(excerpt.text) for citation in citations for excerpt in citation.excerpts)
        > MAX_TOTAL_CITATION_EVIDENCE_CHARS
    ):
        raise ValueError("Reference evidence exceeds the existing materialized citation limit")
    note = "Required final-answer components: " + "; ".join(draft.answer_components) + " Scope: " + draft.scope
    if reference.note:
        note += "\n" + reference.note
    if isinstance(draft, StructuredQuestionDraft):
        if reference.answer_text is not None or not reference.note or not reference.note.strip():
            raise ValueError("Structured reference requires answer_text=null and its complete research report in note")
        response = Response(output=structured_answer, note=note)
        text = compact_json(structured_answer)
    else:
        if reference.answer_text is None:
            raise ValueError("Plain-text reference construction requires the author's complete answer")
        if reference.note is not None:
            raise ValueError("Plain-text reference requires its complete research report in answer_text and note=null")
        text = reference.answer_text
        response = Response(text=text, note=note)
    validate_citation_references((response.text, response.output, response.note), citations)
    return TerminalReference(reference_answer=ReferenceAnswer(text=text, note=note, citations=citations))
