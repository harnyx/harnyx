"""Private complete-source VFS and narrow agent tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from array import array
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast, overload

import regex as safe_regex
from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import BaseModel, Field, field_validator

from harnyx_commons.application.miner_response_hydration import (
    MIN_CITATION_SLICE_CHARS,
    CitationSlice,
)
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain_tweak_generation.contracts import AgentToolSet, ProofStep, ResponseMode
from harnyx_miner_sdk.json_types import JsonObject, JsonValue

_MAX_READ_LINES = 128
_MAX_REGEX_MATCHES = 100
_MAX_AUDIT_TEXT_CHARACTERS = 32_000
_MAX_AUDIT_PACKET_CHARACTERS = 128_000
_MAX_LINK_RESULT_CHARACTERS = 32_000
_MAX_LINK_PATTERN_CHARACTERS = 1_000
_MAX_EVIDENCE_RECORDS = 32
_MAX_CERTIFICATES = 16
_AUDIT_TEXT_TRUNCATION_MARKER = "[... audit text truncated ...]"
_REGEX_SCAN_SECONDS = 2.0
_SIMILARITY_CHUNK_LINES = 128
_SIMILARITY_CHUNK_STRIDE = 96
_MAX_SIMILARITY_QUERY_TERMS = 32
_SIMILARITY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="source-similarity")
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*", re.IGNORECASE)
_DOCUMENT_KINDS = frozenset({"html", "text", "pdf", "xlsx", "static_json"})
SourceFailureClass = Literal["source_fetch_rejected", "source_extraction_limit", "source_unavailable"]
_SOURCE_FAILURE_CLASSES: frozenset[str] = frozenset(
    {
        "source_fetch_rejected",
        "source_extraction_limit",
        "source_unavailable",
    }
)


@dataclass(slots=True)
class _AuditTextSlot:
    container: dict[str, object]
    key: str
    original: str

    @property
    def restoration_units(self) -> int:
        if self._minimum_value() == self.original:
            return 0
        return max(1, len(self.original) - len(_AUDIT_TEXT_TRUNCATION_MARKER))

    def restore(self, units: int) -> None:
        if units <= 0:
            self.minimize()
        elif len(self.original) <= len(_AUDIT_TEXT_TRUNCATION_MARKER):
            self.container[self.key] = self.original
        else:
            self.container[self.key] = _shorten_audit_text(
                self.original,
                len(_AUDIT_TEXT_TRUNCATION_MARKER) + min(units, self.restoration_units),
            )

    def minimize(self) -> None:
        self.container[self.key] = self._minimum_value()

    def _minimum_value(self) -> str:
        return (
            self.original
            if len(json.dumps(self.original, ensure_ascii=False))
            <= len(json.dumps(_AUDIT_TEXT_TRUNCATION_MARKER, ensure_ascii=False))
            else _AUDIT_TEXT_TRUNCATION_MARKER
        )


def _serialize_audit_packet(packet: Mapping[str, object]) -> str:
    return json.dumps(packet, ensure_ascii=False, indent=2)


def _shorten_audit_text(text: str, maximum_characters: int) -> str:
    if len(text) <= maximum_characters or len(text) <= len(_AUDIT_TEXT_TRUNCATION_MARKER):
        return text
    retained_characters = maximum_characters - len(_AUDIT_TEXT_TRUNCATION_MARKER)
    head_characters = (retained_characters + 1) // 2
    tail_characters = retained_characters // 2
    tail = text[-tail_characters:] if tail_characters else ""
    return text[:head_characters] + _AUDIT_TEXT_TRUNCATION_MARKER + tail


def _restore_audit_text(
    packet: dict[str, object],
    slots: Sequence[_AuditTextSlot],
) -> None:
    if not slots:
        return
    maximum_restoration_units = max(slot.restoration_units for slot in slots)
    for slot in slots:
        slot.restore(maximum_restoration_units)
    if len(_serialize_audit_packet(packet)) <= _MAX_AUDIT_PACKET_CHARACTERS:
        return

    lower = 0
    upper = maximum_restoration_units
    accepted = 0
    while lower <= upper:
        candidate = (lower + upper) // 2
        for slot in slots:
            slot.restore(candidate)
        if len(_serialize_audit_packet(packet)) <= _MAX_AUDIT_PACKET_CHARACTERS:
            accepted = candidate
            lower = candidate + 1
        else:
            upper = candidate - 1
    for slot in slots:
        slot.restore(accepted)


def _audit_text_slots(packet: dict[str, object]) -> tuple[tuple[_AuditTextSlot, ...], tuple[_AuditTextSlot, ...]]:
    load_bearing: list[_AuditTextSlot] = []
    context: list[_AuditTextSlot] = []
    for evidence in cast(list[dict[str, object]], packet["selected_evidence"]):
        load_bearing.append(_AuditTextSlot(evidence, "excerpt", cast(str, evidence["excerpt"])))
        for line in cast(list[dict[str, object]], evidence["boundary_context"]):
            context.append(_AuditTextSlot(line, "text", cast(str, line["text"])))
    for certificate in cast(list[dict[str, object]], packet["scan_certificates"]):
        for line in cast(list[dict[str, object]], certificate["matched_lines"]):
            load_bearing.append(_AuditTextSlot(line, "text", cast(str, line["text"])))
        for line in cast(list[dict[str, object]], certificate["boundary_lines"]):
            context.append(_AuditTextSlot(line, "text", cast(str, line["text"])))
    return tuple(load_bearing), tuple(context)


def _structural_shape(value: object, *, depth: int = 0) -> object:
    if depth >= 6:
        return type(value).__name__
    if isinstance(value, Mapping):
        return {str(key)[:100]: _structural_shape(item, depth=depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, list):
        return {
            "list_length": len(value),
            "item_shapes": [_structural_shape(item, depth=depth + 1) for item in value[:3]],
        }
    return type(value).__name__


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_terms(value: str) -> tuple[str, ...]:
    terms = tuple(dict.fromkeys(match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(value)))
    if not terms:
        raise ValueError("similarity query must contain a searchable term")
    if len(terms) > _MAX_SIMILARITY_QUERY_TERMS:
        raise ValueError(f"similarity query exceeds {_MAX_SIMILARITY_QUERY_TERMS} distinct terms")
    return terms


@dataclass(frozen=True, slots=True)
class SourceDocument:
    requested_url: str
    final_url: str
    media_type: str
    content: str
    fetched_bytes: int
    links: tuple[SourceLink, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceLink:
    url: str
    text: str


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    source_candidate_id: str
    url: str
    title: str


@dataclass(frozen=True, slots=True)
class SourceFailureRecord:
    failure_id: str
    failure_class: SourceFailureClass
    source_candidate_id: str


class _SourceFetchAttemptError(RuntimeError):
    def __init__(self, failure: SourceFailureRecord, message: str) -> None:
        super().__init__(message)
        self.failure = failure


class _AgentSDKSearchItem(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)


class _AgentSDKSearchToolResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    tool_use_id: str = Field(min_length=1)
    content: tuple[_AgentSDKSearchItem, ...]

    @field_validator("content", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class _AgentSDKWebSearchResponse(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    query: str
    results: tuple[_AgentSDKSearchToolResult | str, ...]
    duration_seconds: float = Field(alias="durationSeconds", ge=0)
    search_count: int | None = Field(default=None, alias="searchCount", ge=0)

    @field_validator("results", mode="before")
    @classmethod
    def _tuple_from_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True, slots=True)
class StoredSource:
    source_id: str
    requested_url: str
    final_url: str
    media_type: str
    content: str
    content_sha256: str
    fetched_bytes: int
    links: tuple[SourceLink, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceLine:
    line_id: str
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class _SourceLines(Sequence[SourceLine]):
    source: StoredSource
    token: str
    starts: array[int]

    def __len__(self) -> int:
        return len(self.starts)

    @overload
    def __getitem__(self, index: int) -> SourceLine: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[SourceLine]: ...

    def __getitem__(self, index: int | slice) -> SourceLine | Sequence[SourceLine]:
        if isinstance(index, slice):
            return tuple(self._line(offset) for offset in range(*index.indices(len(self))))
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        return self._line(normalized)

    def _line(self, offset: int) -> SourceLine:
        start = self.starts[offset]
        end = self.starts[offset + 1] - 1 if offset + 1 < len(self.starts) else len(self.source.content)
        text = self.source.content[start:end]
        if text.endswith("\r"):
            text = text[:-1]
        ordinal = offset + 1
        return SourceLine(line_id=f"L:{self.token}:{ordinal}", ordinal=ordinal, text=text)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    source_id: str
    url: str
    claim: str
    start_line_id: str
    end_line_id: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class ScanCertificate:
    certificate_id: str
    source_id: str
    start_line_id: str
    end_line_id: str
    pattern: str
    match_line_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SimilarityChunk:
    start_line_id: str
    end_line_id: str
    token_count: int
    query_term_counts: tuple[int, ...]


def _bm25_score(
    chunk: _SimilarityChunk,
    average_length: float,
    inverse_document_frequencies: tuple[float, ...],
) -> float:
    if average_length == 0:
        return 0.0
    length_normalization = 1.5 * (0.25 + (0.75 * chunk.token_count / average_length))
    return sum(
        inverse_document_frequencies[index] * (count * 2.5 / (count + length_normalization))
        for index, count in enumerate(chunk.query_term_counts)
        if count
    )


class SourceFetcherPort(Protocol):
    async def fetch(
        self,
        url: str,
        *,
        document_kind: Literal["html", "text", "pdf", "xlsx", "static_json"],
    ) -> SourceDocument: ...


class SourceWorkspace:
    """Retains complete source bodies while exposing bounded addressable views."""

    def __init__(self) -> None:
        self._sources: dict[str, StoredSource] = {}
        self._source_lines: dict[str, _SourceLines] = {}
        self._line_tokens: dict[str, str] = {}
        self._evidence: dict[str, EvidenceRecord] = {}
        self._certificates: dict[str, ScanCertificate] = {}
        self._source_failures: dict[str, SourceFailureRecord] = {}
        self._source_candidates: dict[str, SourceCandidate] = {}
        self._next_evidence = 1
        self._next_certificate = 1
        self._next_source_candidate = 1

    @property
    def sources(self) -> tuple[StoredSource, ...]:
        return tuple(self._sources.values())

    @property
    def evidence(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence.values())

    @property
    def certificates(self) -> tuple[ScanCertificate, ...]:
        return tuple(self._certificates.values())

    def source_failure(self, failure_id: str) -> SourceFailureRecord | None:
        return self._source_failures.get(failure_id)

    def store(self, document: SourceDocument) -> StoredSource:
        digest = _sha256(document.content)
        source_id = f"source:{_sha256(document.final_url)[:12]}:{digest[:12]}"
        source = StoredSource(
            source_id=source_id,
            requested_url=document.requested_url,
            final_url=document.final_url,
            media_type=document.media_type,
            content=document.content,
            content_sha256=digest,
            fetched_bytes=document.fetched_bytes,
            links=document.links,
        )
        self._sources[source_id] = source
        token = _sha256(f"{source.source_id}\0{source.content_sha256}")[:10]
        starts = array("I", [0])
        starts.extend(
            index + 1
            for index, character in enumerate(source.content)
            if character == "\n" and index + 1 < len(source.content)
        )
        self._source_lines[source_id] = _SourceLines(source=source, token=token, starts=starts)
        self._line_tokens[token] = source_id
        return source

    def register_web_search_results(self, response: object) -> str:
        if not isinstance(response, Mapping):
            raise ValueError(
                "WebSearch response does not match the pinned Agent SDK contract; "
                f"structural_shape={_structural_shape(response)}"
            )
        try:
            search_response = _AgentSDKWebSearchResponse.model_validate(dict(response))
        except Exception as exc:
            raise ValueError(
                "WebSearch response does not match the pinned Agent SDK contract; "
                f"structural_shape={_structural_shape(response)}"
            ) from exc
        registered: list[dict[str, str]] = []
        for result in search_response.results:
            if isinstance(result, str):
                continue
            for item in result.content:
                candidate = self._register_source_candidate(url=item.url, title=item.title)
                registered.append(
                    {
                        "source_candidate_id": candidate.source_candidate_id,
                        "title": candidate.title[:300],
                    }
                )
        if not registered:
            return "Host registered no fetch candidates for this search."
        return "Host-registered fetch candidates:\n" + json.dumps(registered, ensure_ascii=False)

    def _register_source_candidate(self, *, url: str, title: str) -> SourceCandidate:
        candidate_id = f"source_candidate:{self._next_source_candidate}"
        self._next_source_candidate += 1
        candidate = SourceCandidate(
            source_candidate_id=candidate_id,
            url=url,
            title=title,
        )
        self._source_candidates[candidate_id] = candidate
        return candidate

    def get_source_candidate(self, source_candidate_id: str) -> SourceCandidate:
        try:
            return self._source_candidates[source_candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown source_candidate_id: {source_candidate_id}") from exc

    def source_metadata(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "source_id": source.source_id,
                "url": source.final_url,
                "media_type": source.media_type,
                "characters": len(source.content),
                "content_sha256": source.content_sha256,
            }
            for source in self.sources
        )

    def evidence_identities(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "url": item.url,
                "claim": item.claim,
                "start_line_id": item.start_line_id,
                "end_line_id": item.end_line_id,
            }
            for item in self.evidence
        )

    def get_source(self, source_id: str) -> StoredSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise ValueError(f"unknown source_id: {source_id}") from exc

    def lines(self, source: StoredSource) -> Sequence[SourceLine]:
        try:
            return self._source_lines[source.source_id]
        except KeyError as exc:
            raise ValueError(f"source is not retained in this workspace: {source.source_id}") from exc

    def resolve_line(self, line_id: str) -> tuple[StoredSource, SourceLine]:
        prefix, separator, raw_ordinal = line_id.rpartition(":")
        if not separator or not prefix.startswith("L:"):
            raise ValueError(f"unknown line_id: {line_id}")
        source_id = self._line_tokens.get(prefix[2:])
        if source_id is None:
            raise ValueError(f"unknown line_id: {line_id}")
        try:
            ordinal = int(raw_ordinal)
        except ValueError as exc:
            raise ValueError(f"unknown line_id: {line_id}") from exc
        source = self._sources[source_id]
        lines = self._source_lines[source_id]
        if ordinal <= 0 or ordinal > len(lines):
            raise ValueError(f"unknown line_id: {line_id}")
        line = lines[ordinal - 1]
        assert isinstance(line, SourceLine)
        return source, line

    def line_range(self, start_line_id: str, end_line_id: str) -> tuple[StoredSource, tuple[SourceLine, ...]]:
        source, start, end = self._line_bounds(start_line_id, end_line_id)
        line_count = end.ordinal - start.ordinal + 1
        if line_count > _MAX_READ_LINES:
            raise ValueError(f"line range is limited to {_MAX_READ_LINES} lines")
        chosen = tuple(self.lines(source)[start.ordinal - 1 : end.ordinal])
        if not any(item.text.strip() for item in chosen):
            raise ValueError("line range is blank")
        return source, chosen

    def citation_slices(self, evidence: EvidenceRecord) -> tuple[CitationSlice, ...]:
        source, start, end = self._line_bounds(evidence.start_line_id, evidence.end_line_id)
        lines = self._source_lines[source.source_id]
        start_offset = lines.starts[start.ordinal - 1]
        end_offset = lines.starts[end.ordinal] if end.ordinal < len(lines) else len(source.content)
        if len(source.content) < MIN_CITATION_SLICE_CHARS:
            start_offset = 0
            end_offset = len(source.content)
        elif end_offset - start_offset < MIN_CITATION_SLICE_CHARS:
            start_offset = max(0, min(start_offset, len(source.content) - MIN_CITATION_SLICE_CHARS))
            end_offset = max(end_offset, start_offset + MIN_CITATION_SLICE_CHARS)
            if end_offset > len(source.content):
                end_offset = len(source.content)
                start_offset = end_offset - MIN_CITATION_SLICE_CHARS
        return (CitationSlice(start=start_offset, end=end_offset),)

    def _line_bounds(self, start_line_id: str, end_line_id: str) -> tuple[StoredSource, SourceLine, SourceLine]:
        start_source, start = self.resolve_line(start_line_id)
        end_source, end = self.resolve_line(end_line_id)
        if start_source.source_id != end_source.source_id:
            raise ValueError("line range cannot cross sources")
        if start.ordinal > end.ordinal:
            raise ValueError("line range is reversed")
        return start_source, start, end

    def register_evidence(
        self,
        *,
        claim: str,
        start_line_id: str,
        end_line_id: str,
    ) -> EvidenceRecord:
        if len(self._evidence) >= _MAX_EVIDENCE_RECORDS:
            raise ValueError(f"evidence record limit is {_MAX_EVIDENCE_RECORDS}")
        source, start, end = self._line_bounds(start_line_id, end_line_id)
        if end.ordinal - start.ordinal + 1 > _MAX_READ_LINES:
            raise ValueError(f"evidence range is limited to {_MAX_READ_LINES} lines")
        source, chosen = self.line_range(start_line_id, end_line_id)
        chosen = self._attach_table_header(source, chosen)
        if len(chosen) > _MAX_READ_LINES:
            raise ValueError(f"evidence range including its table header is limited to {_MAX_READ_LINES} lines")
        context = self._boundary_context(source, chosen[0].ordinal, chosen[-1].ordinal)
        self._require_bounded_audit_text((*chosen, *context), label="evidence")
        evidence_id = f"E{self._next_evidence}"
        self._next_evidence += 1
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_id=source.source_id,
            url=source.final_url,
            claim=claim.strip(),
            start_line_id=chosen[0].line_id,
            end_line_id=chosen[-1].line_id,
            excerpt="\n".join(item.text for item in chosen),
        )
        if not record.claim:
            raise ValueError("evidence claim must not be blank")
        self._evidence[evidence_id] = record
        return record

    def register_regex_certificate(
        self,
        *,
        source_id: str,
        start_line_id: str,
        end_line_id: str,
        pattern: str,
    ) -> ScanCertificate:
        if len(self._certificates) >= _MAX_CERTIFICATES:
            raise ValueError(f"certificate limit is {_MAX_CERTIFICATES}")
        source, start, end = self._line_bounds(start_line_id, end_line_id)
        if source.source_id != source_id:
            raise ValueError("certificate range does not belong to source_id")
        matches = self._regex_matches(
            source,
            start.ordinal,
            end.ordinal,
            pattern,
            maximum_matches=_MAX_REGEX_MATCHES,
        )
        boundary_lines = self._certificate_boundaries(source, start.ordinal, end.ordinal)
        matched_lines = tuple(self.resolve_line(line_id)[1] for line_id in matches)
        self._require_bounded_audit_text((*boundary_lines, *matched_lines), label="certificate")
        certificate_id = f"C{self._next_certificate}"
        self._next_certificate += 1
        certificate = ScanCertificate(
            certificate_id=certificate_id,
            source_id=source_id,
            start_line_id=start_line_id,
            end_line_id=end_line_id,
            pattern=pattern,
            match_line_ids=matches,
        )
        self._certificates[certificate_id] = certificate
        return certificate

    def proof_packet(
        self,
        *,
        question: str,
        short_answers: Sequence[str],
        steps: Sequence[ProofStep],
        answer_text: str | None,
        validated_citations: Sequence[Mapping[str, object] | None],
        response_mode: ResponseMode = "plain_text",
        output_schema: JsonObject | None = None,
        structured_answer: JsonValue | None = None,
    ) -> dict[str, object]:
        evidence_ids = {evidence_id for step in steps for evidence_id in getattr(step, "evidence_ids", ())}
        certificate_ids = {
            certificate_id for step in steps for certificate_id in getattr(step, "scan_certificate_ids", ())
        }
        packet: dict[str, object] = {
            "question": question,
            "canonical_short_answers": list(short_answers),
            "answer_text": answer_text,
            "validated_citations": list(validated_citations),
            "response_mode": response_mode,
            "output_schema": output_schema,
            "structured_answer": structured_answer,
            "proof_steps": [step.model_dump(mode="json") for step in steps],
            "selected_evidence": [
                self._evidence_audit_view(self._evidence[evidence_id])
                for evidence_id in sorted(evidence_ids)
                if evidence_id in self._evidence
            ],
            "scan_certificates": [
                self._certificate_audit_view(self._certificates[certificate_id])
                for certificate_id in sorted(certificate_ids)
                if certificate_id in self._certificates
            ],
        }
        if len(_serialize_audit_packet(packet)) <= _MAX_AUDIT_PACKET_CHARACTERS:
            return packet
        load_bearing_slots, context_slots = _audit_text_slots(packet)
        for slot in (*load_bearing_slots, *context_slots):
            slot.minimize()
        if len(_serialize_audit_packet(packet)) > _MAX_AUDIT_PACKET_CHARACTERS:
            return packet
        _restore_audit_text(packet, load_bearing_slots)
        _restore_audit_text(packet, context_slots)
        assert len(_serialize_audit_packet(packet)) <= _MAX_AUDIT_PACKET_CHARACTERS
        return packet

    def _evidence_audit_view(self, evidence: EvidenceRecord) -> dict[str, object]:
        source, chosen = self.line_range(evidence.start_line_id, evidence.end_line_id)
        context = self._boundary_context(source, chosen[0].ordinal, chosen[-1].ordinal)
        return {**asdict(evidence), "boundary_context": [asdict(item) for item in context]}

    def _certificate_audit_view(self, certificate: ScanCertificate) -> dict[str, object]:
        source, start, end = self._line_bounds(certificate.start_line_id, certificate.end_line_id)
        boundary_lines = self._certificate_boundaries(source, start.ordinal, end.ordinal)
        return {
            **asdict(certificate),
            "boundary_lines": [asdict(item) for item in boundary_lines],
            "matched_lines": [asdict(self.resolve_line(line_id)[1]) for line_id in certificate.match_line_ids],
        }

    def _regex_matches(
        self,
        source: StoredSource,
        start_ordinal: int,
        end_ordinal: int,
        pattern: str,
        *,
        maximum_matches: int,
    ) -> tuple[str, ...]:
        matches, _ = self._scan_regex(
            source,
            start_ordinal,
            end_ordinal,
            pattern,
            retained_matches=maximum_matches,
            reject_overflow=True,
        )
        return matches

    def _scan_regex(
        self,
        source: StoredSource,
        start_ordinal: int,
        end_ordinal: int,
        pattern: str,
        *,
        retained_matches: int,
        reject_overflow: bool,
    ) -> tuple[tuple[str, ...], int]:
        try:
            compiled = safe_regex.compile(pattern, safe_regex.IGNORECASE)
        except safe_regex.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        deadline = time.monotonic() + _REGEX_SCAN_SECONDS
        matches: list[str] = []
        total = 0
        lines = self.lines(source)
        for ordinal in range(start_ordinal, end_ordinal + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError(f"regex scan exceeded {_REGEX_SCAN_SECONDS:g} seconds")
            line = lines[ordinal - 1]
            assert isinstance(line, SourceLine)
            try:
                matched = compiled.search(line.text, timeout=remaining) is not None
            except TimeoutError as exc:
                raise ValueError(f"regex scan exceeded {_REGEX_SCAN_SECONDS:g} seconds") from exc
            if not matched:
                continue
            total += 1
            if len(matches) < retained_matches:
                matches.append(line.line_id)
                continue
            if reject_overflow:
                raise ValueError(f"certificate match limit is {retained_matches}")
        return tuple(matches), total

    def _boundary_context(
        self,
        source: StoredSource,
        start_ordinal: int,
        end_ordinal: int,
    ) -> tuple[SourceLine, ...]:
        lines = self.lines(source)
        before = lines[max(0, start_ordinal - 3) : start_ordinal - 1]
        after = lines[end_ordinal : min(len(lines), end_ordinal + 2)]
        assert isinstance(before, tuple) and isinstance(after, tuple)
        return (*before, *after)

    def _certificate_boundaries(
        self,
        source: StoredSource,
        start_ordinal: int,
        end_ordinal: int,
    ) -> tuple[SourceLine, ...]:
        lines = self.lines(source)
        offsets = tuple(
            dict.fromkeys(
                (
                    start_ordinal - 1,
                    min(start_ordinal, end_ordinal - 1),
                    max(start_ordinal - 1, end_ordinal - 2),
                    end_ordinal - 1,
                )
            )
        )
        return tuple(lines[offset] for offset in offsets)

    def _require_bounded_audit_text(self, lines: Sequence[SourceLine], *, label: str) -> None:
        unique = {line.line_id: line for line in lines}
        total_characters = sum(len(line.text) for line in unique.values())
        if total_characters > _MAX_AUDIT_TEXT_CHARACTERS:
            raise ValueError(f"{label} audit text exceeds {_MAX_AUDIT_TEXT_CHARACTERS} characters")

    def _similarity_result(self, source: StoredSource, query: str, top_k: int) -> dict[str, object]:
        lines = self.lines(source)
        query_terms = _query_terms(query)
        query_term_indexes = {term: index for index, term in enumerate(query_terms)}
        chunks: list[_SimilarityChunk] = []
        document_frequencies = [0] * len(query_terms)
        total_tokens = 0
        for start in range(0, len(lines), _SIMILARITY_CHUNK_STRIDE):
            chunk = lines[start : min(len(lines), start + _SIMILARITY_CHUNK_LINES)]
            assert isinstance(chunk, tuple) and chunk
            term_counts = [0] * len(query_terms)
            token_count = 0
            for line in chunk:
                for match in _TOKEN_PATTERN.finditer(line.text):
                    token_count += 1
                    term_index = query_term_indexes.get(match.group(0).casefold())
                    if term_index is not None:
                        term_counts[term_index] += 1
            for term_index, count in enumerate(term_counts):
                if count:
                    document_frequencies[term_index] += 1
            total_tokens += token_count
            chunks.append(
                _SimilarityChunk(
                    start_line_id=chunk[0].line_id,
                    end_line_id=chunk[-1].line_id,
                    token_count=token_count,
                    query_term_counts=tuple(term_counts),
                )
            )
            if start + _SIMILARITY_CHUNK_LINES >= len(lines):
                break
        average_length = total_tokens / len(chunks)
        inverse_document_frequencies = tuple(
            math.log1p((len(chunks) - frequency + 0.5) / (frequency + 0.5)) for frequency in document_frequencies
        )
        scores = tuple(_bm25_score(chunk, average_length, inverse_document_frequencies) for chunk in chunks)
        ranked = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)[:top_k]
        return {
            "query": query,
            "method": "BM25",
            "chunks": [
                {
                    "score": float(scores[index]),
                    "start_line_id": chunks[index].start_line_id,
                    "end_line_id": chunks[index].end_line_id,
                }
                for index in ranked
            ],
        }

    async def _run_similarity_result(
        self,
        source: StoredSource,
        query: str,
        top_k: int,
    ) -> dict[str, object]:
        return await asyncio.get_running_loop().run_in_executor(
            _SIMILARITY_EXECUTOR,
            self._similarity_result,
            source,
            query,
            top_k,
        )

    def question_generation_tools(self, fetcher: SourceFetcherPort) -> AgentToolSet:
        return self._tool_set(
            server_name="question_generation_vfs",
            fetcher=fetcher,
            allow_certificates=False,
        )

    def reference_tools(self, fetcher: SourceFetcherPort) -> AgentToolSet:
        return self._tool_set(
            server_name="reference_vfs",
            fetcher=fetcher,
            allow_certificates=True,
        )

    def _source_listing_payload(self) -> dict[str, object]:
        return {"sources": self.source_metadata()}

    async def _regex_search_payload(self, args: Mapping[str, object]) -> dict[str, object]:
        source = self.get_source(str(args["source_id"]))
        context = max(0, min(int(str(args.get("context_lines", 3))), 12))
        limit = max(1, min(int(str(args.get("max_matches", 20))), _MAX_REGEX_MATCHES))
        lines = self.lines(source)
        match_ids, total_matches = await asyncio.to_thread(
            self._scan_regex,
            source,
            1,
            len(lines),
            str(args["pattern"]),
            retained_matches=limit,
            reject_overflow=False,
        )
        windows = []
        returned_lines: list[SourceLine] = []
        for line_id in match_ids:
            _, matched_line = self.resolve_line(line_id)
            index = matched_line.ordinal - 1
            window = lines[max(0, index - context) : min(len(lines), index + context + 1)]
            assert isinstance(window, tuple)
            returned_lines.extend(window)
            windows.append(
                {
                    "match_line_id": lines[index].line_id,
                    "start_line_id": window[0].line_id,
                    "end_line_id": window[-1].line_id,
                    "lines": [asdict(item) for item in window],
                }
            )
        self._require_bounded_audit_text(returned_lines, label="regex search")
        return {
            "pattern": str(args["pattern"]),
            "returned_match_count": len(windows),
            "total_match_count": total_matches,
            "truncated": total_matches > limit,
            "matches": windows,
        }

    def _read_lines_payload(self, args: Mapping[str, object]) -> dict[str, object]:
        source, chosen = self.line_range(str(args["start_line_id"]), str(args["end_line_id"]))
        self._require_bounded_audit_text(chosen, label="read_lines")
        return {
            "source_id": source.source_id,
            "start_line_id": chosen[0].line_id,
            "end_line_id": chosen[-1].line_id,
            "lines": [asdict(item) for item in chosen],
        }

    def audit_tools(self) -> AgentToolSet:
        """Expose the frozen audit boundary: retained-source listing, regex search, and bounded reads only."""

        def success(payload: object) -> dict[str, object]:
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}

        def failure(message: str) -> dict[str, object]:
            return {"content": [{"type": "text", "text": message}], "is_error": True}

        @tool("list_sources", "List retained source metadata without source bodies.", {})
        async def list_sources(_args: dict[str, Any]) -> dict[str, object]:
            return success(self._source_listing_payload())

        @tool(
            "regex_search",
            "Regex-search one complete retained source. Results are bounded and explicitly report truncation.",
            {"source_id": str, "pattern": str, "context_lines": int, "max_matches": int},
        )
        async def regex_search(args: dict[str, Any]) -> dict[str, object]:
            try:
                return success(await self._regex_search_payload(args))
            except Exception as exc:
                return failure(f"regex_search failed: {exc}")

        @tool(
            "read_lines",
            "Read at most 128 exact stable lines from one retained source.",
            {"start_line_id": str, "end_line_id": str},
        )
        async def read_lines(args: dict[str, Any]) -> dict[str, object]:
            try:
                return success(self._read_lines_payload(args))
            except Exception as exc:
                return failure(f"read_lines failed: {exc}")

        tools = (list_sources, regex_search, read_lines)
        server_name = "audit_vfs"
        server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=list(tools))
        return AgentToolSet(
            allowed_tools=tuple(f"mcp__{server_name}__{item.name}" for item in tools),
            mcp_servers={server_name: server},
        )

    async def _fetch_source(
        self,
        fetcher: SourceFetcherPort,
        *,
        source_candidate_id: str,
        document_kind: str,
        public_document_reason: str,
    ) -> StoredSource:
        candidate = self.get_source_candidate(source_candidate_id.strip())
        if document_kind not in _DOCUMENT_KINDS:
            raise ValueError("document_kind must be html, text, pdf, xlsx, or static_json")
        if not public_document_reason or len(public_document_reason) > 500:
            raise ValueError("public_document_reason must contain 1 to 500 characters")
        try:
            document = await fetcher.fetch(
                candidate.url,
                document_kind=document_kind,  # type: ignore[arg-type]
            )
            return self.store(document)
        except Exception as exc:
            code = getattr(exc, "code", None)
            failure_class = (
                cast(SourceFailureClass, code)
                if isinstance(code, str) and code in _SOURCE_FAILURE_CLASSES
                else "source_unavailable"
            )
            failure_id = f"source_failure:{len(self._source_failures) + 1}"
            failure = SourceFailureRecord(
                failure_id=failure_id,
                failure_class=failure_class,
                source_candidate_id=candidate.source_candidate_id,
            )
            self._source_failures[failure_id] = failure
            raise _SourceFetchAttemptError(failure, str(exc)) from exc

    def _tool_set(
        self,
        *,
        server_name: str,
        fetcher: SourceFetcherPort,
        allow_certificates: bool,
    ) -> AgentToolSet:
        def success(payload: object) -> dict[str, object]:
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}

        def failure(message: str) -> dict[str, object]:
            return {"content": [{"type": "text", "text": message}], "is_error": True}

        tools: list[Any] = []

        @tool(
            "fetch_source",
            "Fetch one host-registered ordinary public document into the private VFS; returns metadata only.",
            {"source_candidate_id": str, "document_kind": str, "public_document_reason": str},
        )
        async def fetch_source(args: dict[str, Any]) -> dict[str, object]:
            try:
                source = await self._fetch_source(
                    fetcher,
                    source_candidate_id=str(args.get("source_candidate_id", "")),
                    document_kind=str(args.get("document_kind", "")).strip(),
                    public_document_reason=str(args.get("public_document_reason", "")).strip(),
                )
                metadata = next(item for item in self.source_metadata() if item["source_id"] == source.source_id)
                return success(metadata)
            except _SourceFetchAttemptError as exc:
                return failure(
                    json.dumps(
                        {
                            "error": "fetch_source_failed",
                            "source_failure_id": exc.failure.failure_id,
                            "failure_class": exc.failure.failure_class,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as exc:
                return failure(f"fetch_source failed: {exc}")

        tools.append(fetch_source)

        @tool("list_sources", "List retained source metadata without source bodies.", {})
        async def list_sources(_args: dict[str, Any]) -> dict[str, object]:
            return success(self._source_listing_payload())

        tools.append(list_sources)

        @tool(
            "regex_search",
            "Regex-search one complete retained source. Results are bounded and explicitly report truncation.",
            {"source_id": str, "pattern": str, "context_lines": int, "max_matches": int},
        )
        async def regex_search(args: dict[str, Any]) -> dict[str, object]:
            try:
                return success(await self._regex_search_payload(args))
            except Exception as exc:
                return failure(f"regex_search failed: {exc}")

        tools.append(regex_search)

        @tool(
            "similarity_search",
            "BM25-search fixed line chunks using at most 32 distinct query terms; follow with read_lines before "
            "registering evidence.",
            {"source_id": str, "query": str, "top_k": int},
        )
        async def similarity_search(args: dict[str, Any]) -> dict[str, object]:
            try:
                source = self.get_source(str(args["source_id"]))
                query = str(args["query"]).strip()
                top_k = max(1, min(int(args.get("top_k", 6)), 12))
                result = await self._run_similarity_result(source, query, top_k)
                return success(result)
            except Exception as exc:
                return failure(f"similarity_search failed: {exc}")

        tools.append(similarity_search)

        @tool(
            "read_lines",
            "Read at most 128 exact stable lines from one retained source.",
            {"start_line_id": str, "end_line_id": str},
        )
        async def read_lines(args: dict[str, Any]) -> dict[str, object]:
            try:
                return success(self._read_lines_payload(args))
            except Exception as exc:
                return failure(f"read_lines failed: {exc}")

        tools.append(read_lines)

        @tool(
            "list_source_links",
            "List matching navigation targets retained from one fetched HTML source and register returned targets "
            "for fetch_source.",
            {"source_id": str, "pattern": str},
        )
        async def list_source_links(args: dict[str, Any]) -> dict[str, object]:
            try:
                source = self.get_source(str(args["source_id"]))
                if "html" not in source.media_type:
                    raise ValueError("list_source_links requires a fetched HTML source")
                raw_pattern = str(args.get("pattern", ""))
                if len(raw_pattern) > _MAX_LINK_PATTERN_CHARACTERS:
                    raise ValueError(
                        f"list_source_links pattern is limited to {_MAX_LINK_PATTERN_CHARACTERS} characters"
                    )
                pattern = raw_pattern.casefold()
                matches = tuple(
                    link
                    for link in source.links
                    if not pattern or pattern in link.url.casefold() or pattern in link.text.casefold()
                )
                returned = []
                for link in matches[:100]:
                    candidate = self._register_source_candidate(
                        url=link.url,
                        title=link.text or link.url,
                    )
                    item = {
                        "source_candidate_id": candidate.source_candidate_id,
                        "url": link.url,
                        "text": link.text,
                    }
                    proposed = {
                        "pattern": raw_pattern,
                        "returned_match_count": len(returned) + 1,
                        "total_match_count": len(matches),
                        "truncated": len(returned) + 1 < len(matches),
                        "links": [*returned, item],
                    }
                    if len(json.dumps(proposed, ensure_ascii=False)) > _MAX_LINK_RESULT_CHARACTERS:
                        self._source_candidates.pop(candidate.source_candidate_id)
                        break
                    returned.append(item)
                payload = {
                    "pattern": raw_pattern,
                    "returned_match_count": len(returned),
                    "total_match_count": len(matches),
                    "truncated": len(matches) > len(returned),
                    "links": returned,
                }
                return success(payload)
            except Exception as exc:
                return failure(f"list_source_links failed: {exc}")

        tools.append(list_source_links)

        @tool(
            "register_evidence",
            "Register one exact VFS line range and return a stable evidence ID.",
            {"claim": str, "start_line_id": str, "end_line_id": str},
        )
        async def register_evidence(args: dict[str, Any]) -> dict[str, object]:
            try:
                record = self.register_evidence(
                    claim=str(args["claim"]),
                    start_line_id=str(args["start_line_id"]),
                    end_line_id=str(args["end_line_id"]),
                )
                return success(
                    {
                        "evidence_id": record.evidence_id,
                        "source_id": record.source_id,
                        "start_line_id": record.start_line_id,
                        "end_line_id": record.end_line_id,
                    }
                )
            except Exception as exc:
                return failure(f"register_evidence failed: {exc}")

        tools.append(register_evidence)

        if allow_certificates:

            @tool(
                "register_regex_certificate",
                "Scan an entire declared stable line range and register its complete regex match set.",
                {"source_id": str, "start_line_id": str, "end_line_id": str, "pattern": str},
            )
            async def register_regex_certificate(args: dict[str, Any]) -> dict[str, object]:
                try:
                    certificate = await asyncio.to_thread(
                        self.register_regex_certificate,
                        source_id=str(args["source_id"]),
                        start_line_id=str(args["start_line_id"]),
                        end_line_id=str(args["end_line_id"]),
                        pattern=str(args["pattern"]),
                    )
                    return success(
                        {
                            "certificate_id": certificate.certificate_id,
                            "source_id": certificate.source_id,
                            "start_line_id": certificate.start_line_id,
                            "end_line_id": certificate.end_line_id,
                            "match_count": len(certificate.match_line_ids),
                        }
                    )
                except Exception as exc:
                    return failure(f"register_regex_certificate failed: {exc}")

            tools.append(register_regex_certificate)

        server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=tools)
        names = tuple(f"mcp__{server_name}__{item.name}" for item in tools)
        return AgentToolSet(
            allowed_tools=names,
            mcp_servers={server_name: server},
            search_result_registrar=self.register_web_search_results,
        )

    def _attach_table_header(
        self,
        source: StoredSource,
        chosen: tuple[SourceLine, ...],
    ) -> tuple[SourceLine, ...]:
        if not any(item.text.startswith("ROW\t") for item in chosen):
            return chosen
        lines = self.lines(source)
        start = chosen[0].ordinal - 1
        preceding = lines[max(0, start - 8) : start]
        headers = [item for item in preceding if item.text.startswith("HEADER\t")]
        if not headers:
            return chosen
        header = headers[-1]
        return tuple(lines[header.ordinal - 1 : chosen[-1].ordinal])


__all__ = [
    "EvidenceRecord",
    "ScanCertificate",
    "SourceCandidate",
    "SourceDocument",
    "SourceFetcherPort",
    "SourceLink",
    "SourceWorkspace",
    "StoredSource",
]
