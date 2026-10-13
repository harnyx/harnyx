"""Candidate-local source acquisition, exact line selection and public citations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from array import array
from bisect import bisect_right
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from agents import FunctionTool, RunContextWrapper, function_tool
from agents.exceptions import ModelBehaviorError
from agents.tool_context import ToolContext

from harnyx_commons.application.miner_response_hydration import (
    MAX_TOTAL_CITATION_EVIDENCE_CHARS,
    MIN_CITATION_SLICE_CHARS,
    CitationSlice,
    materialize_citation_slices,
    validate_citation_references,
)
from harnyx_commons.domain.miner_task import AnswerCitation, ReferenceAnswer, Response
from harnyx_commons.domain.tool_usage import SearchToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import (
    known_zero_actual_cost_tool_usage,
    merge_complete_actual_cost_usage,
)
from harnyx_commons.tools.extraction_models import (
    ExtractPagesRequest,
    ExtractPagesResponse,
    canonicalize_extraction_url,
)
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.provider_billing import (
    ProviderBillingMetadata,
    SearchProviderResult,
    billing_evidence_payload,
)
from harnyx_commons.tools.search_models import SearchWebSearchRequest
from harnyx_miner_sdk.structured_output import compact_json, validate_output_size
from harnyx_miner_sdk.tools.search_provider_extra import ParallelSearchExtra

from .contracts import QuestionDraft, RegisteredReferenceOutput, StructuredQuestionDraft
from .output_schema import same_json_value, strict_json, validate_structured_draft
from .source_fetch import (
    MAX_EXTRACTED_CHARACTERS,
    PublicSourceFetcher,
    SourceDocument,
    SourceFetchError,
    _public_addresses,
    _validated_document_url,
)

if TYPE_CHECKING:
    from .agent_runner import CallCapture
    from .reference_runner import ReferenceFeedback

logger = logging.getLogger("harnyx_commons.llm.calls")
FETCH_TIMEOUT_SECONDS = 30.0
SOURCE_SEGMENT_CHARACTERS = 1_000
SOURCE_PAGE_CHARACTERS = 16_000
SOURCE_LINK_PAGE_CHARACTERS = 16_000
SOURCE_SEARCH_MATCHES = 10
MAX_RETAINED_SOURCE_BYTES = 100 * 1024 * 1024
UsableState = Literal["native_reachable", "provider_recovered"]
_USABLE = {"native_reachable", "provider_recovered"}
_ID_MARKER = re.compile(r"\[\[(evidence:[1-9][0-9]*)\]\]")
_COMPARISON_MARKER = re.compile(
    rf"(?:(?P<indent>^[ \t]*)(?:{_ID_MARKER.pattern}[ \t]*)+|[ \t]*{_ID_MARKER.pattern}(?:[ \t]*(?=\r?$))?)",
    re.MULTILINE,
)
_INTERNAL_MARKER = re.compile(r"\[\[.*?\]\]", re.DOTALL)
_EXPOSED_ID = re.compile(r"\bevidence:[0-9]+\b")
T = TypeVar("T")


class ReferenceInputError(ValueError):
    """A model-selected source ID, cursor, query or range can be corrected in the tool loop."""


class SourceStorageLimitError(RuntimeError):
    """A complete source snapshot would exceed the candidate's retained-source budget."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    document: SourceDocument
    state: UsableState
    starts: array[int]

    def end(self, line: int) -> int:
        return self.starts[line] if line < len(self.starts) else len(self.document.content)

    def memory_bytes(self) -> int:
        """Conservatively count snapshot objects, Unicode text, offsets and navigation metadata."""
        document = self.document
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.starts)
            + sys.getsizeof(document)
            + sum(
                sys.getsizeof(value)
                for value in (
                    document.requested_url,
                    document.final_url,
                    document.media_type,
                    document.content,
                    document.fetched_bytes,
                    document.links,
                )
            )
            + sum(sys.getsizeof(link) + sys.getsizeof(link.url) + sys.getsizeof(link.text) for link in document.links)
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    source_id: str
    start: int
    end: int


class EvidenceStore:
    """Own source text versions and the exact excerpts selected from them."""

    def __init__(self) -> None:
        self.sources: dict[str, SourceSnapshot] = {}
        self.evidence: dict[str, Evidence] = {}
        self.states: dict[str, str] = {}
        self.aliases: dict[str, set[str]] = {}
        self.latest_source: dict[str, str] = {}
        self.retained_source_bytes = 0

    def observe(self, url: str, state: str, final_url: str | None = None) -> None:
        try:
            requested = canonicalize_extraction_url(url)
        except ValueError:
            self.states[url] = state
            return
        urls = {requested}
        if final_url is not None:
            final = canonicalize_extraction_url(final_url)
            previous = self.aliases.get(requested, {requested})
            if final not in previous:
                # A changed redirect separates its previous target from the new resource.
                remaining = previous - {requested}
                for key in remaining:
                    self.aliases[key] = remaining
                self.aliases[requested] = {requested}
            urls.add(final)
        for key in tuple(urls):
            urls.update(self.aliases.get(key, ()))
        for key in urls:
            self.aliases[key] = urls
            self.states[key] = state

    def retain(self, document: SourceDocument, state: UsableState) -> str:
        self.observe(document.requested_url, state, document.final_url)
        text = document.content
        requested_url = canonicalize_extraction_url(document.requested_url)
        final_url = canonicalize_extraction_url(document.final_url)
        source_id = self.latest_source.get(final_url) or self.latest_source.get(requested_url, "")
        previous = self.sources.get(source_id)
        if (
            previous is not None
            and canonicalize_extraction_url(previous.document.final_url) == final_url
            and previous.document.content == text
        ):
            snapshot = replace(previous, document=document, state=state)
            replaced_bytes = previous.memory_bytes()
        else:
            starts = array("I")
            offset = 0
            while offset < len(text):
                starts.append(offset)
                stop = min(len(text), offset + SOURCE_SEGMENT_CHARACTERS)
                newline = text.find("\n", offset, stop)
                offset = newline + 1 if newline >= 0 else stop
            source_id = f"source:{len(self.sources) + 1}"
            snapshot = SourceSnapshot(document, state, starts)
            replaced_bytes = 0
        retained_bytes = self.retained_source_bytes - replaced_bytes + snapshot.memory_bytes()
        if retained_bytes > MAX_RETAINED_SOURCE_BYTES:
            self.observe(document.requested_url, "source_storage_limit", document.final_url)
            raise SourceStorageLimitError("Candidate retained-source memory limit reached; find a smaller source.")
        self.sources[source_id] = snapshot
        self.retained_source_bytes = retained_bytes
        for url in self.aliases[requested_url]:
            self.latest_source[url] = source_id
        return source_id

    def usable_source(self, source_id: str) -> SourceSnapshot:
        source = self.sources.get(source_id)
        if source is None:
            raise ReferenceInputError("Unknown source ID")
        for url in (source.document.requested_url, source.document.final_url):
            key = canonicalize_extraction_url(url)
            if self.states.get(key) not in _USABLE or self.latest_source.get(key) != source_id:
                raise ReferenceInputError("Selected source failed its latest check or changed version")
        return source

    def _cursor(self, source_id: str, position: int, link: int, kind: str = "read") -> str:
        return f"{source_id}/{kind}/{position}/{link}"

    def _position(self, source_id: str, cursor: str | None, kind: str = "read") -> tuple[int, int]:
        if cursor is None:
            return 0, 0
        parts = cursor.split("/")
        if len(parts) != 4 or parts[0] != source_id or parts[1] != kind:
            raise ReferenceInputError(
                "Cursor belongs to another source version or operation/search query, or is malformed"
            )
        try:
            position, link = int(parts[2]), int(parts[3])
        except ValueError as exc:
            raise ReferenceInputError("Malformed source cursor") from exc
        if self._cursor(source_id, position, link, kind) != cursor:
            raise ReferenceInputError("Malformed source cursor")
        return position, link

    def view(self, source_id: str, cursor: str | None = None) -> dict[str, Any]:
        source = self.usable_source(source_id)
        first, link_first = self._position(source_id, cursor)
        if not 0 <= first <= len(source.starts) or not 0 <= link_first <= len(source.document.links):
            raise ReferenceInputError("Cursor is outside the retained source")
        lines = []
        size, last = 0, first
        while last < len(source.starts):
            segment = source.document.content[source.starts[last] : source.end(last + 1)]
            line = f"[L{last + 1}] {segment}" + ("" if segment.endswith("\n") else "\n")
            encoded_size = len(json.dumps(line, ensure_ascii=False))
            if size + encoded_size > SOURCE_PAGE_CHARACTERS:
                break
            lines.append(line)
            size += encoded_size
            last += 1
        links = []
        size, link_last = 0, link_first
        while link_last < len(source.document.links):
            link = asdict(source.document.links[link_last])
            encoded_size = len(json.dumps(link, ensure_ascii=False))
            if links and size + encoded_size > SOURCE_LINK_PAGE_CHARACTERS:
                break
            # One link fits the collector's URL/label limits, even if escaped JSON exceeds the page target.
            links.append(link)
            size += encoded_size
            link_last += 1
        more_text, more_links = last < len(source.starts), link_last < len(source.document.links)
        return {
            "source_id": source_id,
            "state": source.state,
            "url": source.document.final_url,
            "cursor": self._cursor(source_id, first, link_first),
            "next_cursor": self._cursor(source_id, last, link_last) if more_text or more_links else None,
            "start_line": first + 1 if lines else None,
            "end_line": last if lines else None,
            "total_lines": len(source.starts),
            "total_characters": len(source.document.content),
            "total_links": len(source.document.links),
            "text_complete": not more_text,
            "links_complete": not more_links,
            "links": links,
            "text": "".join(lines),
        }

    def search(self, source_id: str, query: str, cursor: str | None = None) -> dict[str, Any]:
        source = self.usable_source(source_id)
        if not query:
            raise ReferenceInputError("Search text must not be empty")
        kind = f"search:{sha256(query.encode()).hexdigest()}"
        position, link = self._position(source_id, cursor, kind)
        if not 0 <= position <= len(source.document.content) or link != 0:
            raise ReferenceInputError("Search cursor is outside the retained source")
        matches = []
        offset = source.document.content.find(query, position)
        while offset >= 0 and len(matches) < SOURCE_SEARCH_MATCHES:
            start_line = bisect_right(source.starts, offset)
            end_line = bisect_right(source.starts, offset + len(query) - 1)
            matches.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "cursor": self._cursor(source_id, max(0, start_line - 4), 0),
                }
            )
            position = offset + 1
            offset = source.document.content.find(query, position)
        return {
            "source_id": source_id,
            "matches": matches,
            "next_cursor": self._cursor(source_id, position, 0, kind) if offset >= 0 else None,
            "complete": offset < 0,
        }

    def select(self, source_id: str, start_line: int, end_line: int) -> dict[str, Any]:
        source = self.usable_source(source_id)
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or not 1 <= start_line <= end_line <= len(source.starts)
        ):
            raise ReferenceInputError("Line range must be inclusive, integer, and within the selected source")
        first, last = start_line, end_line
        text = source.document.content
        start, end = source.starts[first - 1], source.end(last)
        if not text[start:end].strip():
            raise ReferenceInputError("Selected source range is blank")
        minimum = min(MIN_CITATION_SLICE_CHARS, len(text))
        while end - start < minimum and last < len(source.starts):
            last += 1
            end = source.end(last)
        while end - start < minimum and first > 1:
            first -= 1
            start = source.starts[first - 1]
        evidence_id = f"evidence:{len(self.evidence) + 1}"
        self.evidence[evidence_id] = Evidence(source_id, start, end)
        preview_end, preview_size = start, 0
        while preview_end < end:
            next_end = min(end, preview_end + SOURCE_SEGMENT_CHARACTERS)
            encoded_size = len(json.dumps(text[preview_end:next_end], ensure_ascii=False))
            if preview_size + encoded_size > SOURCE_PAGE_CHARACTERS:
                break
            preview_end = next_end
            preview_size += encoded_size
        return {
            "evidence_id": evidence_id,
            "source_id": source_id,
            "state": source.state,
            "requested_start_line": start_line,
            "requested_end_line": end_line,
            "start_line": first,
            "end_line": last,
            "excerpt_start": start,
            "excerpt_end": end,
            "excerpt": text[start:preview_end],
            "excerpt_complete": preview_end == end,
            "excerpt_cursor": self._cursor(source_id, first - 1, len(source.document.links)),
            "expanded": (first, last) != (start_line, end_line),
        }

    def selection(self, evidence_id: str) -> Evidence:
        item = self.evidence.get(evidence_id)
        if item is None:
            raise ReferenceInputError("Unregistered evidence ID")
        self.usable_source(item.source_id)
        return item

    def citation(self, evidence_id: str) -> AnswerCitation:
        item = self.selection(evidence_id)
        document = self.sources[item.source_id].document
        retained = materialize_citation_slices(document.content, (CitationSlice(start=item.start, end=item.end),))
        return AnswerCitation(url=document.final_url, excerpts=retained.excerpts)


class ReferenceSources:
    """Apply the same native-first source boundary during source reads."""

    def __init__(
        self, fetcher: PublicSourceFetcher, parallel: ParallelClient, store: EvidenceStore, capture: CallCapture
    ) -> None:
        self.fetcher, self.parallel, self.store, self.capture = fetcher, parallel, store, capture
        self.counts: dict[str, int] = {}

    async def _provider_call(
        self, tool: str, url: str, call: Callable[[], Awaitable[SearchProviderResult[T]]]
    ) -> SearchProviderResult[T]:
        started = monotonic()
        billing: ProviderBillingMetadata | None = None
        event: dict[str, Any] = {"type": "reference.source_call", "tool": tool, "url": url}
        try:
            result = await call()
            billing = result.billing
            return result
        except BaseException as exc:
            billing = getattr(exc, "billing", None)
            event["error_type"] = type(exc).__name__
            raise
        finally:
            cost = billing.actual_cost_usd if billing is not None else None
            summary = replace(
                known_zero_actual_cost_tool_usage(),
                search_tool=SearchToolUsageSummary(call_count=1, reference_cost=cost or 0, actual_cost=cost),
                search_tool_cost=cost or 0,
                reference_total_cost_usd=cost or 0,
                reference_cost_by_provider={"parallel": cost} if cost is not None else {},
                actual_total_cost_usd=cost,
                actual_cost_by_provider={"parallel": cost} if cost is not None else {},
            )
            self.capture.tool_usage = merge_complete_actual_cost_usage(self.capture.tool_usage, summary)
            self.capture.available_cost_usd += cost or 0
            event.update(elapsed_ms=(monotonic() - started) * 1000, billing=billing_evidence_payload(billing))
            self.capture.event(event)

    async def _recovery(self, url: str) -> tuple[str, SourceDocument | None]:
        async def call() -> SearchProviderResult[ExtractPagesResponse]:
            return await self.parallel.extract_pages(
                ExtractPagesRequest(urls=(url,), max_chars_per_result=MAX_EXTRACTED_CHARACTERS + 1)
            )

        try:
            async with asyncio.timeout(FETCH_TIMEOUT_SECONDS):
                await self._validate_provider_url(url)
                result = await self._provider_call("parallel_recovery", url, call)
                if result.response.errors:
                    return "source_unavailable", None
                if len(result.response.pages) != 1:
                    raise ValueError("Parallel omitted the requested source")
                page = result.response.pages[0]
                await self._validate_provider_url(page.url)
                if page.content is None or not page.content.strip():
                    return "source_unavailable", None
                if len(page.content) > MAX_EXTRACTED_CHARACTERS:
                    return "source_extraction_limit", None
                return "provider_recovered", SourceDocument(url, page.url, "text/markdown", page.content, None)
        except TimeoutError:
            return "source_unavailable", None
        except SourceFetchError as exc:
            return exc.code, None

    async def _validate_provider_url(self, url: str) -> None:
        parsed = _validated_document_url(url, "text")
        await asyncio.to_thread(
            _public_addresses, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )

    async def acquire(self, url: str, phase: str) -> tuple[str, SourceDocument | None]:
        started = monotonic()
        state, document = "source_unavailable", None
        try:
            try:
                _validated_document_url(url, "text")
                document = await self.fetcher.fetch(url, timeout_seconds=FETCH_TIMEOUT_SECONDS)
                if not document.content.strip():
                    raise SourceFetchError("source_unavailable", "Native source contains no text")
                state = "native_reachable"
            except (SourceFetchError, TimeoutError) as exc:
                code = exc.code if isinstance(exc, SourceFetchError) else "source_unavailable"
                self.capture.event({"type": "reference.native_failure", "url": url, "phase": phase, "code": code})
                if code not in {"source_unavailable", "source_extraction_limit"}:
                    state = code
                    return state, None
                state, document = await self._recovery(url)
            return state, document
        finally:
            self.store.observe(url, state, document.final_url if document is not None else None)
            self.counts[state] = self.counts.get(state, 0) + 1
            self.capture.event(
                {
                    "type": "reference.reachability",
                    "url": url,
                    "phase": phase,
                    "state": state,
                    "elapsed_ms": (monotonic() - started) * 1000,
                }
            )

    def tools(self, feedback: ReferenceFeedback) -> list[FunctionTool]:
        def model_error(context: RunContextWrapper[Any], error: Exception) -> str:
            if isinstance(error, (ReferenceInputError, ModelBehaviorError)):
                return feedback.error(cast(ToolContext[Any], context).tool_name, str(error))
            raise error

        @function_tool(failure_error_function=model_error)
        async def search_sources(query: str) -> str:
            """Find public source URLs; returned snippets are navigation, never original evidence."""
            if not query.strip():
                raise ReferenceInputError("Search query must not be blank")
            result = await self._provider_call(
                "search_sources",
                query,
                lambda: self.parallel.search_web(
                    SearchWebSearchRequest(
                        provider="parallel",
                        search_queries=(query,),
                        num=5,
                        provider_extra=ParallelSearchExtra(mode="turbo"),
                    )
                ),
            )
            feedback.success("search_sources")
            return result.response.model_dump_json()

        @function_tool(failure_error_function=model_error)
        async def read_source(url: str) -> str:
            """Acquire/check a URL; retain all text and return the first numbered chunk and cursor."""
            state, document = await self.acquire(url, "writing")
            if document is None:
                feedback.success("read_source")
                return json.dumps(
                    {"url": url, "state": state, "source_id": None, "instruction": "Find a replacement source."}
                )
            try:
                source_id = self.store.retain(document, cast(UsableState, state))
            except SourceStorageLimitError as exc:
                self.counts["source_storage_limit"] = self.counts.get("source_storage_limit", 0) + 1
                status = {
                    "type": "reference.source_storage_limit",
                    "retained_source_bytes": self.store.retained_source_bytes,
                    "max_retained_source_bytes": MAX_RETAINED_SOURCE_BYTES,
                    "source_count": len(self.store.sources),
                }
                self.capture.event(status | {"url": url})
                logger.warning(
                    "task_generation.reference_source_storage_limit", extra={"data": self.capture.identity() | status}
                )
                feedback.success("read_source")
                return json.dumps(
                    {"url": url, "state": "source_storage_limit", "source_id": None, "instruction": str(exc)}
                )
            result = self.store.view(source_id)
            feedback.success("read_source")
            return json.dumps(result, ensure_ascii=False)

        @function_tool(failure_error_function=model_error)
        async def read_chunk(source_id: str, cursor: str) -> str:
            """Read a cursor-selected chunk from the usable retained source, without another fetch."""
            result = self.store.view(source_id, cursor)
            feedback.success("read_chunk")
            return json.dumps(result, ensure_ascii=False)

        @function_tool(failure_error_function=model_error)
        async def search_text(source_id: str, query: str, cursor: str | None = None) -> str:
            """Case-sensitive literal search in retained text; return match read cursors and search continuation."""
            result = self.store.search(source_id, query, cursor)
            feedback.success("search_text")
            return json.dumps(result, ensure_ascii=False)

        @function_tool(failure_error_function=model_error)
        async def select_evidence(source_id: str, start_line: int, end_line: int) -> str:
            """Select one-based inclusive lines; host copies exact text and returns its evidence ID and offsets."""
            result = self.store.select(source_id, start_line, end_line)
            feedback.success("select_evidence")
            return json.dumps(result, ensure_ascii=False)

        return [search_sources, read_source, read_chunk, search_text, select_evidence]


def reference_has_factual_correction(output: RegisteredReferenceOutput, draft: QuestionDraft) -> bool:
    """Compare values after rendering has validated all submitted evidence markers."""
    if not isinstance(draft, StructuredQuestionDraft):
        return False

    def content(value: Any) -> Any:
        if isinstance(value, str):
            return _COMPARISON_MARKER.sub(lambda marker: marker.group("indent") or "", value)
        if isinstance(value, list):
            return [content(item) for item in value]
        if isinstance(value, dict):
            return {key: content(item) for key, item in value.items()}
        return value

    uncited = content(strict_json(cast(str, output.structured_answer_json)))
    return not same_json_value(compact_json(uncited), draft.structured_answer_json)


def render_reference(output: RegisteredReferenceOutput, store: EvidenceStore, draft: QuestionDraft) -> ReferenceAnswer:
    selections = tuple(store.selection(evidence_id) for evidence_id in output.citation_evidence_ids)
    if sum(item.end - item.start for item in selections) > MAX_TOTAL_CITATION_EVIDENCE_CHARS:
        raise ValueError("Reference citations exceed the total evidence limit")
    citations = tuple(store.citation(evidence_id) for evidence_id in output.citation_evidence_ids)
    positions: dict[str, int] = {}
    for position, evidence_id in enumerate(output.citation_evidence_ids, 1):
        positions.setdefault(evidence_id, position)

    def markers(text: str | None) -> str | None:
        if text is None:
            return None

        def replace_marker(match: re.Match[str]) -> str:
            evidence_id = match.group(1)
            if evidence_id not in positions:
                raise ValueError("Inline evidence ID is absent from submitted citations")
            return f"[[{positions[evidence_id]}]]"

        without_ids = _ID_MARKER.sub("", text)
        if _INTERNAL_MARKER.search(without_ids) or "[[evidence" in without_ids or _EXPOSED_ID.search(without_ids):
            raise ValueError("Malformed, numeric or exposed internal citation marker")
        return _ID_MARKER.sub(replace_marker, text)

    def string_leaves(value: Any) -> Any:
        if isinstance(value, str):
            return markers(value)
        if isinstance(value, list):
            return [string_leaves(item) for item in value]
        if isinstance(value, dict):
            return {key: string_leaves(item) for key, item in value.items()}
        return value

    note = markers(output.note)
    if isinstance(draft, StructuredQuestionDraft):
        if output.structured_answer_json is None or output.answer_text is not None:
            raise ValueError("Structured reference requires its structured representation")
        value = string_leaves(strict_json(output.structured_answer_json))
        text = compact_json(value)
        validate_structured_draft(draft.output_schema_json, text)
        validate_output_size(value)
        response = Response(output=value, note=note)
    else:
        if output.answer_text is None or output.structured_answer_json is not None:
            raise ValueError("Plain reference requires answer_text")
        text = cast(str, markers(output.answer_text))
        response = Response(text=text, note=note)
    validate_citation_references((response.text, response.output, response.note), citations)
    return ReferenceAnswer(text=text, note=response.note, citations=citations)
