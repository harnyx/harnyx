from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import httpx
import pytest

from harnyx_commons.domain_tweak_generation.source_evidence import (
    BatchSourceEvidence,
    SourceEvidenceLimitError,
    _ProviderAccounting,
    validate_direct_public_url,
)
from harnyx_commons.domain_tweak_generation.types import DomainTweakSourceEvidencePolicy
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_parallel_extract
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS,
    DomainTweakEvidenceDeclaration,
    DomainTweakReferenceClaim,
)
from harnyx_commons.tools.extraction_models import (
    ExtractedPage,
    ExtractPagesRequest,
    ExtractPagesResponse,
    PageExtractionError,
)
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.ports import PageExtractionProviderPort
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _ExtractionProvider:
    content_by_url: dict[str, str]
    unavailable_urls: set[str] = field(default_factory=set)
    requests: list[ExtractPagesRequest] = field(default_factory=list)

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        pages = tuple(
            ExtractedPage(url=url, title=f"Title for {url}", content=self.content_by_url[url])
            for url in request.urls
            if url not in self.unavailable_urls
        )
        errors = tuple(
            PageExtractionError(
                url=url,
                error_type="fetch_error",
                http_status_code=404,
                content="Not Found",
            )
            for url in request.urls
            if url in self.unavailable_urls
        )
        return SearchProviderResult(
            response=ExtractPagesResponse(pages=pages, errors=errors),
            billing=ProviderBillingMetadata(
                actual_cost_provider="parallel",
                actual_cost_usd=0.001 * len(request.urls),
                billable_units=len(request.urls),
                source="request_body",
                service="extract",
            ),
        )


@dataclass
class _BlockingExtractionProvider:
    requests: list[ExtractPagesRequest] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return _successful_result(request)


@dataclass
class _SecondChunkBlockingExtractionProvider:
    requests: list[ExtractPagesRequest] = field(default_factory=list)
    second_chunk_started: asyncio.Event = field(default_factory=asyncio.Event)

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        if len(self.requests) == 2:
            self.second_chunk_started.set()
            await asyncio.Event().wait()
        return _successful_result(request)


@dataclass
class _FailingExtractionProvider:
    requests: list[ExtractPagesRequest] = field(default_factory=list)
    failure_code: ToolProviderFailureCode = ToolProviderFailureCode.PROVIDER_FAILED

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        raise ToolProviderError(
            "malformed provider response",
            provider="parallel",
            failure_code=self.failure_code,
        )


async def test_batch_cache_reuses_positive_and_negative_results() -> None:
    provider = _ExtractionProvider(
        content_by_url={"https://example.com/good": "A directly satisfies both predicates."},
        unavailable_urls={"https://example.com/missing"},
    )
    batch = _batch(provider)
    declarations = (
        _declaration("good", "https://example.com/good", "A directly satisfies"),
        _declaration("missing", "https://example.com/missing", "missing"),
    )

    first = batch.new_session(objective="question", claim_ids=("answer",))
    second = batch.new_session(objective="question", claim_ids=("answer",))
    await first.acquire_declared(declarations)
    await second.acquire_declared(declarations)

    assert len(provider.requests) == 1
    assert first.summary().provider_request_count == 1
    assert second.summary().provider_request_count == 0
    assert second.summary().positive_cache_hit_count == 1
    assert second.summary().negative_cache_hit_count == 1
    assert second.summary().failed_source_count == 1


async def test_typed_provider_failure_becomes_cached_source_error() -> None:
    provider = _FailingExtractionProvider()
    batch = _batch(provider)
    declaration = _declaration("source", "https://example.com/source", "source")
    first = batch.new_session(objective="question", claim_ids=("answer",))
    second = batch.new_session(objective="question", claim_ids=("answer",))

    first_result = await first.acquire_declared((declaration,))
    second_result = await second.acquire_declared((declaration,))

    assert len(provider.requests) == 1
    assert first_result[0]["error_type"] == "provider_request_failed"
    assert second_result[0]["error_type"] == "provider_request_failed"
    assert first.summary().provider_request_count == 1
    assert first.summary().failed_source_count == 1
    assert second.summary().negative_cache_hit_count == 1


@pytest.mark.parametrize(
    "failure_code",
    (
        ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE,
        ToolProviderFailureCode.AUTHENTICATION_FAILED,
    ),
)
async def test_provider_credential_failure_aborts_the_batch(
    failure_code: ToolProviderFailureCode,
) -> None:
    provider = _FailingExtractionProvider(failure_code=failure_code)
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))

    with pytest.raises(ToolProviderError) as exc_info:
        await session.acquire_declared(
            (_declaration("source", "https://example.com/source", "source"),)
        )

    assert exc_info.value.failure_code is failure_code
    assert len(provider.requests) == 1


async def test_cancelled_owner_does_not_cancel_shared_source_waiter() -> None:
    provider = _BlockingExtractionProvider()
    batch = _batch(provider)
    declaration = _declaration("shared", "https://example.com/shared", "shared")
    owner_session = batch.new_session(objective="owner", claim_ids=("answer",))
    waiter_session = batch.new_session(objective="waiter", claim_ids=("answer",))

    owner = asyncio.create_task(owner_session.acquire_declared((declaration,)))
    await provider.started.wait()
    owner.add_done_callback(lambda _: provider.release.set())
    asyncio.get_running_loop().call_soon(owner.cancel)

    waiter_result = await waiter_session.acquire_declared((declaration,))

    with pytest.raises(asyncio.CancelledError):
        await owner
    assert len(provider.requests) == 1
    assert waiter_result[0]["status"] == "acquired"
    assert owner_session.summary().provider_request_count == 1
    assert waiter_session.summary().provider_billable_units == 1


async def test_later_chunk_cancellation_preserves_completed_cache_and_accounting() -> None:
    provider = _SecondChunkBlockingExtractionProvider()
    batch = _batch(provider)
    urls = tuple(
        f"https://example.com/{index}"
        for index in range(21)
    )
    accounting: list[_ProviderAccounting] = []

    acquisition = asyncio.create_task(
        batch.acquire_urls(
            urls,
            objective="question",
            on_provider_accounting=accounting.append,
        )
    )
    await provider.second_chunk_started.wait()
    acquisition.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert len(provider.requests) == 2
    assert sum(item.provider_request_count for item in accounting) == 2
    assert sum(item.submitted_url_count for item in accounting) == 21
    assert sum(
        billing.billable_units
        for item in accounting
        for billing in item.billing
    ) == 20
    assert sum(
        billing.actual_cost_usd or 0.0
        for item in accounting
        for billing in item.billing
    ) == pytest.approx(0.02)

    cached_session = batch.new_session(objective="question", claim_ids=("answer",))
    await cached_session.acquire_declared(
        tuple(
            _declaration(str(index), url, f"content {index}")
            for index, url in enumerate(urls[:20])
        )
    )
    assert len(provider.requests) == 2
    assert cached_session.summary().positive_cache_hit_count == 20


async def test_declared_source_limit_fails_before_provider_work() -> None:
    provider = _ExtractionProvider(content_by_url={})
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))
    declarations = tuple(
        _declaration(
            str(index),
            f"https://example.com/{index}",
            f"content {index}",
        )
        for index in range(DOMAIN_TWEAK_MAX_DECLARED_EVIDENCE_URLS + 1)
    )

    with pytest.raises(SourceEvidenceLimitError) as exc_info:
        await session.acquire_declared(declarations)

    assert exc_info.value.reason == "source_declared_url_limit"
    assert session.terminal_reason == "source_declared_url_limit"
    assert provider.requests == []


async def test_source_extraction_cost_does_not_count_as_grounded_search() -> None:
    provider = _ExtractionProvider(
        content_by_url={"https://example.com/source": "direct source content"}
    )
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))

    await session.acquire_declared(
        (_declaration("source", "https://example.com/source", "direct source"),)
    )

    assert session.tool_usage.search_tool.call_count == 0
    assert session.tool_usage.search_tool_cost == 0.0
    assert session.tool_usage.reference_total_cost_usd == pytest.approx(0.001)
    assert session.tool_usage.actual_total_cost_usd == pytest.approx(0.001)


async def test_canonical_equivalent_urls_share_one_parallel_source_snapshot() -> None:
    submitted_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        submitted_urls.extend(payload["urls"])
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-canonical",
                "session_id": "session-canonical",
                "results": [
                    {
                        "url": "https://example.com/~user",
                        "full_content": "canonical source content",
                    }
                ],
                "errors": [],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        provider = ParallelClient(
            base_url="https://api.parallel.ai",
            api_key="parallel-key",
            client=http_client,
        )
        session = _batch(provider).new_session(objective="question", claim_ids=("answer",))

        result = await session.acquire_declared(
            (
                _declaration("encoded", "https://example.com/%7Euser", "canonical"),
                _declaration("decoded", "https://example.com/~user", "canonical"),
            )
        )

    assert [item["status"] for item in result] == ["acquired", "acquired"]
    assert submitted_urls == ["https://example.com/~user"]
    assert session.summary().provider_request_count == 1
    assert session.summary().submitted_url_count == 1


async def test_invalid_parallel_partition_preserves_provider_billing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-invalid-partition",
                "session_id": "session-invalid-partition",
                "results": [
                    {
                        "url": "https://example.com/source",
                        "full_content": "untrusted content",
                    }
                ],
                "errors": [
                    {
                        "url": "https://example.com/source",
                        "error_type": "fetch_error",
                    }
                ],
                "usage": [{"name": "sku_extract_excerpts", "count": 1}],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        provider = ParallelClient(
            base_url="https://api.parallel.ai",
            api_key="parallel-key",
            client=http_client,
        )
        session = _batch(provider).new_session(objective="question", claim_ids=("answer",))

        result = await session.acquire_declared(
            (_declaration("source", "https://example.com/source", "source"),)
        )

    assert result[0]["error_type"] == "provider_request_failed"
    assert session.summary().provider_request_count == 1
    assert session.summary().provider_billable_units == 1
    assert session.summary().provider_cost_usd == pytest.approx(
        price_parallel_extract(url_count=1)
    )
    assert session.tool_usage.actual_cost_by_provider["parallel"] == pytest.approx(
        price_parallel_extract(url_count=1)
    )


async def test_batched_acquisition_shares_one_response_content_budget() -> None:
    provider = _ExtractionProvider(
        content_by_url={
            "https://example.com/one": "abcdefghij",
            "https://example.com/two": "klmnopqrst",
        }
    )
    session = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(
            max_tool_response_chars=10,
            max_supplemental_urls_per_invocation=2,
        ),
    ).new_session(objective="question", claim_ids=("answer",))

    result = await session.acquire_sources(
        "answer",
        ["https://example.com/one", "https://example.com/two"],
        limit=10,
    )

    acquired = [item for item in result["results"] if item["status"] == "acquired"]
    assert len(acquired) == 2
    assert sum(len(str(item["content"])) for item in acquired) == 10
    assert session.terminal_reason is None


async def test_cached_source_paginates_and_only_acquired_windows_hydrate() -> None:
    provider = _ExtractionProvider(content_by_url={"https://example.com/source": "0123456789abcdefghij"})
    batch = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(
            max_cached_chars_per_source=100,
            max_initial_prompt_chars=10,
            max_tool_response_chars=10,
        ),
    )
    session = batch.new_session(objective="question", claim_ids=("answer",))
    initial = await session.acquire_declared((_declaration("source", "https://example.com/source", "0123"),))
    source_id = str(initial[0]["source_id"])

    page = await session.read_cached_source(source_id, offset=10, limit=10)
    window_id = str(page["window_id"])
    citations = session.hydrate_citations(
        (window_id,),
        (
            DomainTweakReferenceClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                evidence_window_ids=(window_id,),
                support_explanation="The acquired page range supports the claim.",
            ),
        ),
    )

    assert page["content"] == "abcdefghij"
    assert len(citations) == 1
    assert citations[0].url == "https://example.com/source"
    assert citations[0].note.startswith("[slice 10:20]\nabcdefghij\n\n")
    assert "A satisfies both predicates" in citations[0].note
    with pytest.raises(ValueError, match="unknown citation window ID"):
        session.hydrate_citations(("window-invented",), ())


async def test_windows_remain_bound_to_the_claim_that_acquired_them() -> None:
    provider = _ExtractionProvider(
        content_by_url={"https://example.com/source": "A directly satisfies both predicates."}
    )
    session = _batch(provider).new_session(objective="question", claim_ids=("answer", "other"))

    initial = await session.acquire_declared(
        (_declaration("source", "https://example.com/source", "A directly satisfies"),)
    )
    window_id = str(initial[0]["window"]["window_id"])

    assert session.allowed_claim_ids_by_window_id[window_id] == frozenset({"answer"})


async def test_hydration_describes_collective_support_without_per_source_overclaim() -> None:
    provider = _ExtractionProvider(
        content_by_url={
            "https://example.com/one": "The first part of the answer.",
            "https://example.com/two": "The second part of the answer.",
        }
    )
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))
    initial = await session.acquire_declared(
        (
            _declaration("one", "https://example.com/one", "first part"),
            _declaration("two", "https://example.com/two", "second part"),
        )
    )
    window_ids = tuple(str(item["window"]["window_id"]) for item in initial)
    claim = DomainTweakReferenceClaim(
        claim_id="answer",
        claim="The complete answer combines the first and second parts.",
        role="answer_determining",
        evidence_window_ids=window_ids,
        support_explanation="The two acquired pages jointly support the claim.",
    )

    citations = session.hydrate_citations(window_ids, (claim,))

    assert all("Contributes with other cited sources to support:" in item.note for item in citations)
    assert all("\n\nSupports:" not in item.note for item in citations)


async def test_hydrated_citation_source_excerpt_is_bounded() -> None:
    provider = _ExtractionProvider(
        content_by_url={"https://example.com/source": "A" * 5_000}
    )
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))
    initial = await session.acquire_declared(
        (_declaration("source", "https://example.com/source", "AAAA"),)
    )
    window_id = str(initial[0]["window"]["window_id"])
    claim = DomainTweakReferenceClaim(
        claim_id="answer",
        claim="A satisfies both predicates.",
        role="answer_determining",
        evidence_window_ids=(window_id,),
        support_explanation="The acquired source supports the claim.",
    )

    note = session.hydrate_citations((window_id,), (claim,))[0].note
    excerpt_header, remainder = note.split("\n", 1)
    excerpt, claim_binding = remainder.split("\n\n", 1)

    assert excerpt_header == "[slice 0:2000]"
    assert len(excerpt) == 2_000
    assert claim_binding == "Supports: A satisfies both predicates."


async def test_hydrated_citation_notes_have_one_total_payload_limit() -> None:
    provider = _ExtractionProvider(
        content_by_url={"https://example.com/source": "source content"}
    )
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))
    initial = await session.acquire_declared(
        (_declaration("source", "https://example.com/source", "source"),)
    )
    window_id = str(initial[0]["window"]["window_id"])
    claim = DomainTweakReferenceClaim(
        claim_id="answer",
        claim="A" * 120_001,
        role="answer_determining",
        evidence_window_ids=(window_id,),
        support_explanation="The acquired source supports the claim.",
    )

    with pytest.raises(ValueError, match="hydrated citation notes exceed 120000 characters"):
        session.hydrate_citations((window_id,), (claim,))


async def test_only_uncached_acquisition_counts_toward_three_round_limit() -> None:
    urls = {f"https://example.com/{index}": f"content {index}" for index in range(1, 5)}
    provider = _ExtractionProvider(content_by_url=urls)
    session = _batch(provider).new_session(objective="question", claim_ids=("answer",))

    rejected = await session.acquire_sources("answer", ["http://127.0.0.1/private"])
    assert rejected["counted_round"] is False

    first = await session.acquire_sources("answer", ["https://example.com/1"])
    cached = await session.acquire_sources("answer", ["https://example.com/1"])
    await session.acquire_sources("answer", ["https://example.com/2"])
    await session.acquire_sources("answer", ["https://example.com/3"])

    assert first["counted_round"] is True
    assert cached["counted_round"] is False
    assert session.summary().acquisition_round_count == 3

    with pytest.raises(SourceEvidenceLimitError) as exc_info:
        await session.acquire_sources("answer", ["https://example.com/4"])

    assert exc_info.value.reason == "source_acquisition_round_limit"
    assert session.terminal_reason == "source_acquisition_round_limit"
    assert session.summary().round_limit_activation_count == 1
    assert len(provider.requests) == 3


async def test_shared_cache_hits_count_toward_candidate_supplemental_url_limit() -> None:
    urls = tuple(f"https://example.com/{index}" for index in range(1, 4))
    provider = _ExtractionProvider(
        content_by_url={
            url: f"content {index}" for index, url in enumerate(urls, start=1)
        }
    )
    batch = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(max_supplemental_urls_per_invocation=2),
    )
    priming_session = batch.new_session(objective="first question", claim_ids=("answer",))
    await priming_session.acquire_declared(
        tuple(
            _declaration(f"source-{index}", url, f"content {index}")
            for index, url in enumerate(urls, start=1)
        )
    )
    session = batch.new_session(objective="second question", claim_ids=("answer",))

    await session.acquire_sources("answer", list(urls[:2]))

    with pytest.raises(SourceEvidenceLimitError) as exc_info:
        await session.acquire_sources("answer", [urls[2]])

    assert exc_info.value.reason == "source_supplemental_url_limit"
    assert session.summary().supplemental_url_count == 2
    assert session.summary().acquisition_round_count == 0
    assert len(provider.requests) == 1


async def test_candidate_source_rebinding_does_not_consume_supplemental_url_budget() -> None:
    url = "https://example.com/source"
    provider = _ExtractionProvider(content_by_url={url: "source content"})
    session = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(max_supplemental_urls_per_invocation=0),
    ).new_session(objective="question", claim_ids=("answer", "other"))
    await session.acquire_declared((_declaration("source", url, "source"),))

    rebound = await session.acquire_sources("other", [url])

    window_id = str(rebound["results"][0]["window_id"])
    assert session.allowed_claim_ids_by_window_id[window_id] == frozenset({"answer", "other"})
    assert rebound["counted_round"] is False
    assert session.summary().supplemental_url_count == 0
    assert session.summary().acquisition_round_count == 0


async def test_content_limit_is_explicit_candidate_terminal_state() -> None:
    provider = _ExtractionProvider(content_by_url={"https://example.com/source": "x" * 20})
    session = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(max_cached_chars_per_source=20),
    ).new_session(objective="question", claim_ids=("answer",))

    await session.acquire_declared((_declaration("source", "https://example.com/source", "xxxx"),))

    assert session.terminal_reason == "source_content_limit"


async def test_supplemental_content_limit_stops_current_and_future_source_tool_work() -> None:
    url = "https://example.com/source"
    provider = _ExtractionProvider(content_by_url={url: "x" * 20})
    session = _batch(
        provider,
        policy=DomainTweakSourceEvidencePolicy(max_cached_chars_per_source=20),
    ).new_session(objective="question", claim_ids=("answer",))

    with pytest.raises(SourceEvidenceLimitError) as exc_info:
        await session.acquire_sources("answer", [url], limit=20)

    assert exc_info.value.reason == "source_content_limit"
    assert session.terminal_reason == "source_content_limit"
    first_summary = session.summary()
    assert first_summary.provider_request_count == 1
    assert first_summary.provider_billable_units == 1
    assert first_summary.acquired_source_count == 1
    assert first_summary.source_tool_call_count == 1

    with pytest.raises(SourceEvidenceLimitError) as repeated_exc_info:
        await session.read_cached_source("unused-source-id")

    assert repeated_exc_info.value.reason == "source_content_limit"
    second_summary = session.summary()
    assert second_summary.provider_request_count == first_summary.provider_request_count
    assert second_summary.source_tool_call_count == first_summary.source_tool_call_count


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "http://localhost/private",
        "http://10.0.0.1/private",
        "https://user:password@example.com/private",
    ),
)
async def test_direct_source_url_rejects_non_public_or_credentialed_targets(url: str) -> None:
    normalized, reason = validate_direct_public_url(url)

    assert normalized is None
    assert reason is not None


def _batch(
    provider: PageExtractionProviderPort,
    *,
    policy: DomainTweakSourceEvidencePolicy | None = None,
) -> BatchSourceEvidence:
    return BatchSourceEvidence(
        provider=provider,
        policy=policy or DomainTweakSourceEvidencePolicy(),
        client_model="gemini-test",
    )


def _successful_result(
    request: ExtractPagesRequest,
) -> SearchProviderResult[ExtractPagesResponse]:
    return SearchProviderResult(
        response=ExtractPagesResponse(
            pages=tuple(
                ExtractedPage(
                    url=url,
                    title=f"Title for {url}",
                    content=f"content for {url}",
                )
                for url in request.urls
            )
        ),
        billing=ProviderBillingMetadata(
            actual_cost_provider="parallel",
            actual_cost_usd=0.001 * len(request.urls),
            billable_units=len(request.urls),
            source="request_body",
            service="extract",
        ),
    )


def _declaration(
    evidence_id: str,
    url: str,
    excerpt: str,
) -> DomainTweakEvidenceDeclaration:
    return DomainTweakEvidenceDeclaration(
        evidence_id=evidence_id,
        source_url=url,
        source_title="Source",
        source_locator=None,
        claimed_excerpt=excerpt,
        supported_claim_ids=("answer",),
        support_explanation="The page directly supports the answer claim.",
    )
