"""Batch-scoped acquisition, caching, and citation identity for source evidence."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from urllib.parse import SplitResult, urlsplit, urlunsplit

from harnyx_commons.domain.miner_task import AnswerCitation
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakDiscardReason,
    DomainTweakSourceEvidencePolicy,
    DomainTweakSourceEvidenceSummary,
)
from harnyx_commons.errors import ToolProviderError, is_tool_provider_credential_failure
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS,
    DomainTweakEvidenceDeclaration,
    DomainTweakReferenceClaim,
)
from harnyx_commons.tools.extraction_models import (
    ExtractPagesRequest,
    ExtractPagesResponse,
    PageExtractionError,
    canonicalize_extraction_url,
)
from harnyx_commons.tools.ports import PageExtractionProviderPort
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID = "reference_additional_explanation"
_PARALLEL_EXTRACT_BATCH_SIZE = 20
_INITIAL_WINDOW_CONTEXT_CHARS = 1_000
_MAX_CITATION_SOURCE_EXCERPT_CHARS = 2_000
_MAX_HYDRATED_CITATION_NOTE_CHARS = 120_000


class SourceEvidenceLimitError(RuntimeError):
    """A declared source-tool budget terminated one candidate."""

    def __init__(self, reason: DomainTweakDiscardReason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    source_id: str
    requested_url: str
    resolved_url: str
    title: str | None
    content: str | None
    error_type: str | None
    http_status_code: int | None
    content_limit_reached: bool

    @property
    def acquired(self) -> bool:
        return self.content is not None and self.error_type is None


@dataclass(frozen=True, slots=True)
class SourceWindow:
    window_id: str
    source_id: str
    url: str
    title: str | None
    offset: int
    content: str
    total_content_chars: int
    has_more: bool
    excerpt_match: str | None = None
    evidence_id: str | None = None

    def prompt_payload(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "offset": self.offset,
            "content": self.content,
            "total_content_chars": self.total_content_chars,
            "has_more": self.has_more,
            "excerpt_match": self.excerpt_match,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class _AcquisitionResult:
    snapshots: tuple[SourceSnapshot, ...]
    submitted_url_count: int
    positive_cache_hit_count: int
    negative_cache_hit_count: int


@dataclass(frozen=True, slots=True)
class _ProviderAccounting:
    provider_request_count: int = 0
    submitted_url_count: int = 0
    billing: tuple[ProviderBillingMetadata, ...] = ()


@dataclass(eq=False, slots=True)
class _FetchGroup:
    urls: tuple[str, ...]
    task: asyncio.Task[dict[str, SourceSnapshot]] | None = None
    accounting: list[_ProviderAccounting] = field(default_factory=list)
    claimed_accounting_count: int = 0
    waiter_count: int = 0
    accepting_waiters: bool = True


class BatchSourceEvidence:
    """Owns one generate_batch call's positive and negative source cache."""

    def __init__(
        self,
        *,
        provider: PageExtractionProviderPort,
        policy: DomainTweakSourceEvidencePolicy,
        client_model: str,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._client_model = client_model
        self._cache: dict[str, SourceSnapshot] = {}
        self._inflight: dict[str, _FetchGroup] = {}
        self._lock = asyncio.Lock()

    def new_session(
        self,
        *,
        objective: str,
        claim_ids: Sequence[str],
    ) -> SourceEvidenceSession:
        return SourceEvidenceSession(
            batch=self,
            policy=self._policy,
            objective=objective,
            claim_ids=claim_ids,
        )

    async def acquire_urls(
        self,
        urls: Sequence[str],
        *,
        objective: str,
        before_fetch: Callable[[int], Awaitable[None]] | None = None,
        on_provider_accounting: Callable[[_ProviderAccounting], None] | None = None,
    ) -> _AcquisitionResult:
        ordered_urls = tuple(dict.fromkeys(urls))
        if not ordered_urls:
            return _AcquisitionResult(
                snapshots=(),
                submitted_url_count=0,
                positive_cache_hit_count=0,
                negative_cache_hit_count=0,
            )

        owned_urls: list[str] = []
        waiting: dict[str, _FetchGroup] = {}
        cache_hits: dict[str, SourceSnapshot] = {}
        groups: list[_FetchGroup] = []
        seen_groups: set[_FetchGroup] = set()
        async with self._lock:
            for url in ordered_urls:
                cached = self._cache.get(url)
                if cached is not None:
                    cache_hits[url] = cached
                    continue
                pending = self._inflight.get(url)
                if pending is not None and not pending.accepting_waiters:
                    self._inflight.pop(url, None)
                    pending = None
                if pending is None:
                    owned_urls.append(url)
                    continue
                waiting[url] = pending
                if pending not in seen_groups:
                    seen_groups.add(pending)
                    groups.append(pending)

            if owned_urls:
                # Reserve candidate-local budget before the shared fetch becomes visible.
                if before_fetch is not None:
                    await before_fetch(len(owned_urls))
                group = _FetchGroup(urls=tuple(owned_urls))
                group.task = asyncio.create_task(
                    self._fetch_group(group, objective=objective)
                )
                for url in owned_urls:
                    self._inflight[url] = group
                    waiting[url] = group
                seen_groups.add(group)
                groups.append(group)

            for group in groups:
                group.waiter_count += 1

        group_results: dict[_FetchGroup, dict[str, SourceSnapshot]] = {}
        try:
            for group in groups:
                task = group.task
                if task is None:
                    raise RuntimeError("source fetch group task was not initialized")
                group_results[group] = await asyncio.shield(task)
        except BaseException:
            accounting = await self._release_groups(groups)
            _record_provider_accounting(accounting, on_provider_accounting)
            raise
        accounting = await self._release_groups(groups)
        _record_provider_accounting(accounting, on_provider_accounting)

        resolved = dict(cache_hits)
        for url, group in waiting.items():
            resolved[url] = group_results[group][url]
        snapshots = tuple(resolved[url] for url in ordered_urls)
        return _AcquisitionResult(
            snapshots=snapshots,
            submitted_url_count=len(owned_urls),
            positive_cache_hit_count=sum(snapshot.acquired for snapshot in cache_hits.values()),
            negative_cache_hit_count=sum(not snapshot.acquired for snapshot in cache_hits.values()),
        )

    async def _fetch_group(
        self,
        group: _FetchGroup,
        *,
        objective: str,
    ) -> dict[str, SourceSnapshot]:
        fetched: dict[str, SourceSnapshot] = {}
        try:
            for start in range(0, len(group.urls), _PARALLEL_EXTRACT_BATCH_SIZE):
                chunk = group.urls[start : start + _PARALLEL_EXTRACT_BATCH_SIZE]
                group.accounting.append(
                    _ProviderAccounting(
                        provider_request_count=1,
                        submitted_url_count=len(chunk),
                    )
                )
                try:
                    result = await self._provider.extract_pages(
                        ExtractPagesRequest(
                            urls=chunk,
                            objective=objective,
                            max_chars_per_result=self._policy.max_cached_chars_per_source,
                            max_age_seconds=600,
                            disable_cache_fallback=True,
                            client_model=self._client_model,
                        )
                    )
                except ToolProviderError as exc:
                    if is_tool_provider_credential_failure(exc):
                        raise
                    result = _provider_failure_result(chunk, exc)
                group.accounting.append(_ProviderAccounting(billing=(result.billing,)))
                chunk_snapshots = self._snapshots_from_response(chunk, result.response)
                fetched.update(chunk_snapshots)
                await self._commit_chunk_without_cancellation(group, chunk_snapshots)
            return fetched
        finally:
            await self._finish_group(group)

    async def _commit_chunk_without_cancellation(
        self,
        group: _FetchGroup,
        snapshots: dict[str, SourceSnapshot],
    ) -> None:
        commit = asyncio.create_task(self._commit_chunk(group, snapshots))
        try:
            await asyncio.shield(commit)
        except asyncio.CancelledError:
            await commit
            raise

    async def _commit_chunk(
        self,
        group: _FetchGroup,
        snapshots: dict[str, SourceSnapshot],
    ) -> None:
        async with self._lock:
            for url, snapshot in snapshots.items():
                self._cache[url] = snapshot
                if self._inflight.get(url) is group:
                    self._inflight.pop(url)

    async def _finish_group(self, group: _FetchGroup) -> None:
        async with self._lock:
            group.accepting_waiters = False
            for url in group.urls:
                if self._inflight.get(url) is group:
                    self._inflight.pop(url)

    async def _release_groups(
        self,
        groups: Sequence[_FetchGroup],
    ) -> tuple[_ProviderAccounting, ...]:
        accounting: list[_ProviderAccounting] = []
        tasks_to_cancel: list[asyncio.Task[dict[str, SourceSnapshot]]] = []
        async with self._lock:
            for group in groups:
                accounting.extend(group.accounting[group.claimed_accounting_count :])
                group.claimed_accounting_count = len(group.accounting)
                if group.waiter_count <= 0:
                    continue
                group.waiter_count -= 1
                task = group.task
                if group.waiter_count == 0 and task is not None and not task.done():
                    group.accepting_waiters = False
                    task.cancel()
                    tasks_to_cancel.append(task)
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        return tuple(accounting)

    def _snapshots_from_response(
        self,
        requested_urls: tuple[str, ...],
        response: ExtractPagesResponse,
    ) -> dict[str, SourceSnapshot]:
        pages = {_normalized_response_url(page.url): page for page in response.pages}
        errors = {_normalized_response_url(error.url): error for error in response.errors}
        snapshots: dict[str, SourceSnapshot] = {}
        for url in requested_urls:
            page = pages.get(url)
            if page is not None:
                if page.content is not None:
                    snapshots[url] = SourceSnapshot(
                        source_id=_source_id(url),
                        requested_url=url,
                        resolved_url=page.url,
                        title=page.title,
                        content=page.content,
                        error_type=None,
                        http_status_code=None,
                        content_limit_reached=(
                            len(page.content) >= self._policy.max_cached_chars_per_source
                        ),
                    )
                else:
                    snapshots[url] = SourceSnapshot(
                        source_id=_source_id(url),
                        requested_url=url,
                        resolved_url=page.url,
                        title=page.title,
                        content=None,
                        error_type="missing_full_content",
                        http_status_code=None,
                        content_limit_reached=False,
                    )
                continue
            error = errors[url]
            snapshots[url] = SourceSnapshot(
                source_id=_source_id(url),
                requested_url=url,
                resolved_url=url,
                title=None,
                content=None,
                error_type=error.error_type,
                http_status_code=error.http_status_code,
                content_limit_reached=False,
            )
        return snapshots


class SourceEvidenceSession:
    """Owns source-tool budgets and citation allowlisting for one candidate."""

    def __init__(
        self,
        *,
        batch: BatchSourceEvidence,
        policy: DomainTweakSourceEvidencePolicy,
        objective: str,
        claim_ids: Sequence[str],
    ) -> None:
        self._batch = batch
        self._policy = policy
        self._objective = objective
        self._claim_ids = frozenset((*claim_ids, DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID))
        self._sources: dict[str, SourceSnapshot] = {}
        self._candidate_source_urls: set[str] = set()
        self._source_claim_ids: dict[str, set[str]] = {}
        self._windows: dict[str, SourceWindow] = {}
        self._allowed_claim_ids_by_window_id: dict[str, set[str]] = {}
        self._initial_evidence: list[dict[str, object]] = []
        self._rounds_by_claim: dict[str, int] = {}
        self._budget_lock = asyncio.Lock()
        self._tool_usage = ToolUsageSummary.zero()
        self._terminal_reason: DomainTweakDiscardReason | None = None
        self._provider_request_count = 0
        self._provider_billable_units = 0
        self._provider_cost_usd = 0.0
        self._submitted_url_count = 0
        self._acquired_source_count = 0
        self._failed_source_count = 0
        self._positive_cache_hit_count = 0
        self._negative_cache_hit_count = 0
        self._initial_window_count = 0
        self._source_tool_call_count = 0
        self._cached_source_read_count = 0
        self._supplemental_acquisition_call_count = 0
        self._supplemental_url_count = 0
        self._acquisition_round_count = 0
        self._round_limit_activation_count = 0

    @property
    def terminal_reason(self) -> DomainTweakDiscardReason | None:
        return self._terminal_reason

    @property
    def allowed_window_ids(self) -> frozenset[str]:
        return frozenset(self._windows)

    @property
    def allowed_claim_ids_by_window_id(self) -> dict[str, frozenset[str]]:
        return {
            window_id: frozenset(claim_ids)
            for window_id, claim_ids in self._allowed_claim_ids_by_window_id.items()
        }

    @property
    def tool_usage(self) -> ToolUsageSummary:
        return self._tool_usage

    async def acquire_declared(
        self,
        declarations: Sequence[DomainTweakEvidenceDeclaration],
    ) -> tuple[dict[str, object], ...]:
        if len(declarations) > DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS:
            self._terminate("source_declared_url_limit")
        valid_urls: list[str] = []
        normalized_by_evidence: dict[str, str] = {}
        errors_by_evidence: dict[str, str] = {}
        for declaration in declarations:
            normalized, error = validate_direct_public_url(declaration.source_url)
            if error is not None:
                errors_by_evidence[declaration.evidence_id] = error
                continue
            assert normalized is not None
            normalized_by_evidence[declaration.evidence_id] = normalized
            valid_urls.append(normalized)

        self._candidate_source_urls.update(valid_urls)
        acquisition = await self._batch.acquire_urls(
            valid_urls,
            objective=self._objective,
            on_provider_accounting=self._record_provider_accounting,
        )
        self._record_acquisition(acquisition)
        snapshots_by_url = {snapshot.requested_url: snapshot for snapshot in acquisition.snapshots}
        remaining_initial_chars = self._policy.max_initial_prompt_chars
        for declaration in declarations:
            url_error = errors_by_evidence.get(declaration.evidence_id)
            if url_error is not None:
                self._failed_source_count += 1
                self._initial_evidence.append(
                    _initial_error_payload(declaration, "invalid_url", detail=url_error)
                )
                continue
            normalized = normalized_by_evidence[declaration.evidence_id]
            snapshot = snapshots_by_url[normalized]
            self._sources[snapshot.source_id] = snapshot
            if not snapshot.acquired:
                self._failed_source_count += 1
                self._initial_evidence.append(
                    _initial_error_payload(
                        declaration,
                        snapshot.error_type or "source_unavailable",
                        http_status_code=snapshot.http_status_code,
                    )
                )
                continue
            self._source_claim_ids.setdefault(snapshot.source_id, set()).update(
                declaration.supported_claim_ids
            )
            self._acquired_source_count += 1
            if snapshot.content_limit_reached:
                self._terminal_reason = "source_content_limit"
            window = self._initial_window(
                declaration,
                snapshot,
                available_chars=remaining_initial_chars,
            )
            if window is not None:
                self._record_window(window, declaration.supported_claim_ids)
                remaining_initial_chars -= len(window.content)
                self._initial_window_count += 1
            self._initial_evidence.append(
                {
                    "evidence_id": declaration.evidence_id,
                    "status": "acquired",
                    "source_id": snapshot.source_id,
                    "url": snapshot.resolved_url,
                    "title": snapshot.title,
                    "content_limit_reached": snapshot.content_limit_reached,
                    "window": window.prompt_payload() if window is not None else None,
                    "initial_content_omitted": window is None,
                }
            )
        return tuple(self._initial_evidence)

    async def read_cached_source(
        self,
        source_id: str,
        offset: int = 0,
        limit: int = 10_000,
    ) -> dict[str, object]:
        """Read one bounded page range already acquired for this candidate."""
        await self._begin_tool_call()
        self._cached_source_read_count += 1
        if limit <= 0 or limit > self._policy.max_tool_response_chars:
            self._terminate("source_tool_response_limit")
        if offset < 0:
            return {"status": "rejected", "reason": "offset_must_be_non_negative"}
        snapshot = self._sources.get(source_id)
        if snapshot is None or not snapshot.acquired:
            return {"status": "rejected", "reason": "source_id_not_acquired"}
        assert snapshot.content is not None
        if offset >= len(snapshot.content):
            return {
                "status": "rejected",
                "reason": "offset_out_of_range",
                "total_content_chars": len(snapshot.content),
            }
        window = _source_window(snapshot, offset=offset, limit=limit)
        self._record_window(window, self._source_claim_ids[source_id])
        return {"status": "acquired", **window.prompt_payload()}

    async def acquire_sources(
        self,
        claim_id: str,
        urls: list[str],
        offset: int = 0,
        limit: int = 10_000,
    ) -> dict[str, object]:
        """Acquire direct URLs for one claim within one aggregate response character budget."""
        await self._begin_tool_call()
        self._supplemental_acquisition_call_count += 1
        if claim_id not in self._claim_ids:
            return {"status": "rejected", "reason": "unknown_claim_id", "claim_id": claim_id}
        if limit <= 0 or limit > self._policy.max_tool_response_chars:
            self._terminate("source_tool_response_limit")
        if offset < 0:
            return {"status": "rejected", "reason": "offset_must_be_non_negative"}
        normalized_urls: list[str] = []
        rejected: list[dict[str, str]] = []
        for url in dict.fromkeys(urls):
            normalized, error = validate_direct_public_url(url)
            if error is not None:
                rejected.append({"url": url, "reason": error})
                continue
            assert normalized is not None
            normalized_urls.append(normalized)
        normalized_urls = list(dict.fromkeys(normalized_urls))
        await self._reserve_supplemental_urls(normalized_urls)
        acquisition = await self._batch.acquire_urls(
            normalized_urls,
            objective=self._objective,
            before_fetch=lambda _count: self._reserve_supplemental_round(claim_id),
            on_provider_accounting=self._record_provider_accounting,
        )
        self._record_acquisition(acquisition)
        results: list[dict[str, object]] = []
        content_limit_reached = False
        remaining_response_chars = limit
        remaining_window_count = sum(
            snapshot.acquired
            and snapshot.content is not None
            and offset < len(snapshot.content)
            for snapshot in acquisition.snapshots
        )
        for snapshot in acquisition.snapshots:
            self._sources[snapshot.source_id] = snapshot
            if not snapshot.acquired:
                self._failed_source_count += 1
                results.append(
                    {
                        "status": "error",
                        "url": snapshot.requested_url,
                        "error_type": snapshot.error_type,
                        "http_status_code": snapshot.http_status_code,
                    }
                )
                continue
            self._source_claim_ids.setdefault(snapshot.source_id, set()).add(claim_id)
            self._acquired_source_count += 1
            if snapshot.content_limit_reached:
                self._terminal_reason = "source_content_limit"
                content_limit_reached = True
            assert snapshot.content is not None
            if offset >= len(snapshot.content):
                results.append(
                    {
                        "status": "error",
                        "url": snapshot.resolved_url,
                        "error_type": "offset_out_of_range",
                        "total_content_chars": len(snapshot.content),
                    }
                )
                continue
            window_limit = remaining_response_chars // remaining_window_count
            remaining_window_count -= 1
            if window_limit <= 0:
                results.append(
                    {
                        "status": "acquired",
                        "source_id": snapshot.source_id,
                        "url": snapshot.resolved_url,
                        "title": snapshot.title,
                        "total_content_chars": len(snapshot.content),
                        "content_omitted": True,
                        "reason": "source_tool_response_limit",
                    }
                )
                continue
            window = _source_window(snapshot, offset=offset, limit=window_limit)
            remaining_response_chars -= len(window.content)
            self._record_window(window, (claim_id,))
            results.append({"status": "acquired", **window.prompt_payload()})
        if content_limit_reached:
            self._terminate("source_content_limit")
        return {
            "status": "completed",
            "claim_id": claim_id,
            "counted_round": acquisition.submitted_url_count > 0,
            "completed_rounds_for_claim": self._rounds_by_claim.get(claim_id, 0),
            "results": results,
            "rejected_urls": rejected,
        }

    def reference_prompt_sources(self) -> dict[str, object]:
        return {
            "initial_evidence": self._initial_evidence,
            "source_tool_contract": {
                "read_cached_source": "Read another range from an acquired source_id.",
                "acquire_sources": (
                    "Acquire direct URLs for one known unresolved claim within one aggregate "
                    "response character budget."
                ),
                "additional_explanation_claim_id": DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID,
                "max_tool_response_chars": self._policy.max_tool_response_chars,
                "max_source_tool_calls": self._policy.max_tool_calls_per_invocation,
                "max_supplemental_urls": self._policy.max_supplemental_urls_per_invocation,
                "max_acquisition_rounds_per_claim": self._policy.max_acquisition_rounds_per_claim,
            },
        }

    def gate_windows(self, window_ids: Sequence[str]) -> tuple[dict[str, object], ...]:
        return tuple(self._windows[window_id].prompt_payload() for window_id in window_ids)

    def hydrate_citations(
        self,
        window_ids: Sequence[str],
        claims: Sequence[DomainTweakReferenceClaim],
    ) -> tuple[AnswerCitation, ...]:
        citations: list[AnswerCitation] = []
        total_note_chars = 0
        for window_id in window_ids:
            window = self._windows.get(window_id)
            if window is None:
                raise ValueError(f"unknown citation window ID: {window_id}")
            supported_claims = [claim for claim in claims if window_id in claim.evidence_window_ids]
            if not supported_claims:
                raise ValueError(f"citation window has no manifest claim: {window_id}")
            note = _citation_note(window, supported_claims)
            total_note_chars += len(note)
            if total_note_chars > _MAX_HYDRATED_CITATION_NOTE_CHARS:
                raise ValueError(
                    "hydrated citation notes exceed "
                    f"{_MAX_HYDRATED_CITATION_NOTE_CHARS} characters"
                )
            citations.append(
                AnswerCitation(
                    url=window.url,
                    title=window.title or window.url,
                    note=note,
                )
            )
        return tuple(citations)

    def summary(self) -> DomainTweakSourceEvidenceSummary:
        return DomainTweakSourceEvidenceSummary(
            provider_request_count=self._provider_request_count,
            provider_billable_units=self._provider_billable_units,
            provider_cost_usd=self._provider_cost_usd,
            submitted_url_count=self._submitted_url_count,
            acquired_source_count=self._acquired_source_count,
            failed_source_count=self._failed_source_count,
            positive_cache_hit_count=self._positive_cache_hit_count,
            negative_cache_hit_count=self._negative_cache_hit_count,
            initial_window_count=self._initial_window_count,
            source_tool_call_count=self._source_tool_call_count,
            cached_source_read_count=self._cached_source_read_count,
            supplemental_acquisition_call_count=self._supplemental_acquisition_call_count,
            supplemental_url_count=self._supplemental_url_count,
            acquisition_round_count=self._acquisition_round_count,
            round_limit_activation_count=self._round_limit_activation_count,
        )

    async def _begin_tool_call(self) -> None:
        async with self._budget_lock:
            if self._terminal_reason is not None:
                raise SourceEvidenceLimitError(self._terminal_reason)
            if self._source_tool_call_count >= self._policy.max_tool_calls_per_invocation:
                self._terminate("source_tool_call_limit")
            self._source_tool_call_count += 1

    async def _reserve_supplemental_urls(self, urls: Sequence[str]) -> None:
        async with self._budget_lock:
            new_urls = tuple(url for url in urls if url not in self._candidate_source_urls)
            if (
                self._supplemental_url_count + len(new_urls)
                > self._policy.max_supplemental_urls_per_invocation
            ):
                self._terminate("source_supplemental_url_limit")
            self._candidate_source_urls.update(new_urls)
            self._supplemental_url_count += len(new_urls)

    async def _reserve_supplemental_round(self, claim_id: str) -> None:
        async with self._budget_lock:
            rounds = self._rounds_by_claim.get(claim_id, 0)
            if rounds >= self._policy.max_acquisition_rounds_per_claim:
                self._round_limit_activation_count += 1
                self._terminate("source_acquisition_round_limit")
            self._rounds_by_claim[claim_id] = rounds + 1
            self._acquisition_round_count += 1

    def _record_acquisition(self, acquisition: _AcquisitionResult) -> None:
        self._positive_cache_hit_count += acquisition.positive_cache_hit_count
        self._negative_cache_hit_count += acquisition.negative_cache_hit_count

    def _record_provider_accounting(self, accounting: _ProviderAccounting) -> None:
        self._provider_request_count += accounting.provider_request_count
        self._submitted_url_count += accounting.submitted_url_count
        for billing in accounting.billing:
            self._provider_billable_units += billing.billable_units or 0
            self._provider_cost_usd += billing.actual_cost_usd or 0.0
            self._tool_usage = merge_tool_usage_summaries(
                self._tool_usage,
                _tool_usage_from_billing(billing),
            )

    def _initial_window(
        self,
        declaration: DomainTweakEvidenceDeclaration,
        snapshot: SourceSnapshot,
        *,
        available_chars: int,
    ) -> SourceWindow | None:
        if available_chars <= 0:
            return None
        assert snapshot.content is not None
        offset, match = _excerpt_offset(snapshot.content, declaration.claimed_excerpt)
        limit = min(self._policy.max_tool_response_chars, available_chars)
        return _source_window(
            snapshot,
            offset=offset,
            limit=limit,
            excerpt_match=match,
            evidence_id=declaration.evidence_id,
        )

    def _record_window(self, window: SourceWindow, claim_ids: Iterable[str]) -> None:
        self._windows.setdefault(window.window_id, window)
        self._allowed_claim_ids_by_window_id.setdefault(window.window_id, set()).update(claim_ids)

    def _terminate(self, reason: DomainTweakDiscardReason) -> None:
        self._terminal_reason = reason
        raise SourceEvidenceLimitError(reason)


def validate_direct_public_url(url: str) -> tuple[str | None, str | None]:
    """Return a normalized direct public HTTP URL or a mechanical rejection reason."""
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError:
        return None, "invalid_url"
    if parsed.scheme.lower() not in {"http", "https"}:
        return None, "unsupported_url_scheme"
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None, "invalid_url_authority"
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return None, "non_public_url_host"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None, "invalid_url_host"
    else:
        if not address.is_global:
            return None, "non_public_url_host"
    normalized = _normalized_split_url(parsed, host=host, port=port)
    try:
        return canonicalize_extraction_url(normalized), None
    except ValueError:
        return None, "invalid_url"


def _normalized_split_url(parsed: SplitResult, *, host: str, port: int | None) -> str:
    display_host = f"[{host}]" if ":" in host else host
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = display_host if port is None or default_port else f"{display_host}:{port}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _normalized_response_url(url: str) -> str:
    try:
        return canonicalize_extraction_url(url)
    except ValueError:
        return url.strip()


def _source_id(url: str) -> str:
    return f"source_{hashlib.sha256(url.encode()).hexdigest()[:24]}"


def _window_id(source_id: str, offset: int, content: str) -> str:
    identity = f"{source_id}:{offset}:{len(content)}:{hashlib.sha256(content.encode()).hexdigest()}"
    return f"window_{hashlib.sha256(identity.encode()).hexdigest()[:24]}"


def _source_window(
    snapshot: SourceSnapshot,
    *,
    offset: int,
    limit: int,
    excerpt_match: str | None = None,
    evidence_id: str | None = None,
) -> SourceWindow:
    assert snapshot.content is not None
    content = snapshot.content[offset : offset + limit]
    return SourceWindow(
        window_id=_window_id(snapshot.source_id, offset, content),
        source_id=snapshot.source_id,
        url=snapshot.resolved_url,
        title=snapshot.title,
        offset=offset,
        content=content,
        total_content_chars=len(snapshot.content),
        has_more=offset + len(content) < len(snapshot.content),
        excerpt_match=excerpt_match,
        evidence_id=evidence_id,
    )


def _citation_note(
    window: SourceWindow,
    claims: Sequence[DomainTweakReferenceClaim],
) -> str:
    excerpt = window.content[:_MAX_CITATION_SOURCE_EXCERPT_CHARS]
    excerpt_end = window.offset + len(excerpt)
    direct = [claim.claim for claim in claims if len(claim.evidence_window_ids) == 1]
    collective = [claim.claim for claim in claims if len(claim.evidence_window_ids) > 1]
    claim_bindings: list[str] = []
    if direct:
        claim_bindings.append("Supports: " + " ".join(direct))
    if collective:
        claim_bindings.append(
            "Contributes with other cited sources to support: " + " ".join(collective)
        )
    return (
        f"[slice {window.offset}:{excerpt_end}]\n{excerpt}\n\n"
        + " ".join(claim_bindings)
    )


def _excerpt_offset(content: str, claimed_excerpt: str) -> tuple[int, str]:
    exact = content.find(claimed_excerpt)
    if exact >= 0:
        return max(0, exact - _INITIAL_WINDOW_CONTEXT_CHARS), "exact"
    needle_tokens = claimed_excerpt.split()
    if not needle_tokens:
        return 0, "none"
    token_matches = list(re.finditer(r"\S+", content))
    normalized_content = " ".join(match.group(0) for match in token_matches)
    normalized_excerpt = " ".join(needle_tokens)
    normalized_index = normalized_content.find(normalized_excerpt)
    if normalized_index < 0:
        return 0, "none"
    token_index = normalized_content[:normalized_index].count(" ")
    original_offset = token_matches[token_index].start()
    return max(0, original_offset - _INITIAL_WINDOW_CONTEXT_CHARS), "normalized"


def _initial_error_payload(
    declaration: DomainTweakEvidenceDeclaration,
    error_type: str,
    *,
    detail: str | None = None,
    http_status_code: int | None = None,
) -> dict[str, object]:
    return {
        "evidence_id": declaration.evidence_id,
        "status": "error",
        "url": declaration.source_url,
        "error_type": error_type,
        "http_status_code": http_status_code,
        "detail": detail,
        "window": None,
    }


def _provider_failure_result(
    urls: Sequence[str],
    exc: ToolProviderError,
) -> SearchProviderResult[ExtractPagesResponse]:
    response = ExtractPagesResponse(
        errors=tuple(
            PageExtractionError(
                url=url,
                error_type="provider_request_failed",
                http_status_code=exc.http_status,
                content=None,
            )
            for url in urls
        )
    )
    return SearchProviderResult(
        response=response,
        billing=exc.billing
        or ProviderBillingMetadata(
            actual_cost_provider="parallel",
            source="missing_provider_metadata",
            billable_units=0,
            service="extract",
        ),
    )


def _tool_usage_from_billing(billing: ProviderBillingMetadata) -> ToolUsageSummary:
    reference_cost = billing.actual_cost_usd or 0.0
    actual_cost = billing.actual_cost_usd
    return ToolUsageSummary(
        reference_total_cost_usd=reference_cost,
        reference_cost_by_provider={billing.actual_cost_provider: reference_cost},
        actual_total_cost_usd=actual_cost,
        actual_cost_by_provider=(
            {billing.actual_cost_provider: actual_cost} if actual_cost is not None else {}
        ),
    )


def _record_provider_accounting(
    accounting: Sequence[_ProviderAccounting],
    callback: Callable[[_ProviderAccounting], None] | None,
) -> None:
    if callback is None:
        return
    for item in accounting:
        callback(item)


__all__ = [
    "BatchSourceEvidence",
    "DOMAIN_TWEAK_ADDITIONAL_EXPLANATION_CLAIM_ID",
    "SourceEvidenceLimitError",
    "SourceEvidenceSession",
    "SourceSnapshot",
    "SourceWindow",
    "validate_direct_public_url",
]
