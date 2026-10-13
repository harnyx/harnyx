"""Prevent stale evidence, misplaced citations and repeated acquisition during finalization."""

import asyncio
import json
import tracemalloc
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.miner_task_generation.agent_runner import CallCapture
from harnyx_commons.miner_task_generation.contracts import (
    CandidateSession,
    QuestionDraft,
    RegisteredReferenceOutput,
    StructuredQuestionDraft,
)
from harnyx_commons.miner_task_generation.reference_evidence import (
    SOURCE_PAGE_CHARACTERS,
    SOURCE_SEARCH_MATCHES,
    EvidenceStore,
    ReferenceSources,
    SourceStorageLimitError,
    render_reference,
)
from harnyx_commons.miner_task_generation.reference_runner import ReferenceFeedback
from harnyx_commons.miner_task_generation.source_fetch import SourceDocument, SourceFetchError, SourceLink
from harnyx_commons.tools.extraction_models import ExtractedPage, ExtractPagesResponse, PageExtractionError
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

URL = "https://example.org/report"
TEXT = "Title and year 2026\nAlpha is 88 points.\nBeta is 82 points.\nGamma is 70 points.\n" + "context " * 18


@pytest.fixture
def native_admission(monkeypatch):
    from harnyx_commons.miner_task_generation import source_fetch

    queued = asyncio.Event()
    timeouts, native_calls, recoveries = [], [], []
    original_timeout = asyncio.timeout

    class Admission(asyncio.Semaphore):
        async def acquire(self):
            queued.set()
            return await super().acquire()

    def timeout(delay):
        context = original_timeout(delay)
        timeouts.append(context)
        return context

    async def fetch(url, _kind):
        native_calls.append(url)
        return source_fetch._FetchedBody(url, "text/plain", "", TEXT.encode())

    async def extract(*_args):
        return source_fetch._ExtractedSourcePayload(content=TEXT)

    class Parallel:
        async def extract_pages(self, request):
            recoveries.append(request)
            return SearchProviderResult(
                ExtractPagesResponse(pages=(ExtractedPage(url=URL, content=TEXT),)),
                ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.001),
            )

    monkeypatch.setattr(asyncio, "timeout", timeout)
    monkeypatch.setattr(source_fetch, "_fetch_isolated", fetch)
    monkeypatch.setattr(source_fetch, "_extract_isolated", extract)
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_args: ("93.184.216.34",)
    )
    fetcher = source_fetch.PublicSourceFetcher()
    fetcher._worker_slots = Admission(0)
    capture = CallCapture(CandidateSession("admission", date.today()), "reference", "gpt-6-luna", "openai")
    return SimpleNamespace(
        sources=ReferenceSources(fetcher, Parallel(), EvidenceStore(), capture),
        queued=queued,
        timeouts=timeouts,
        native_calls=native_calls,
        recoveries=recoveries,
        permits=fetcher._worker_slots,
    )


@pytest.mark.anyio
async def test_native_queue_wait_does_not_start_fetch_timeout_or_paid_recovery(native_admission):
    """Capacity pressure must not charge for recovery before native acquisition actually starts."""
    probe = native_admission
    task = asyncio.create_task(probe.sources.acquire(URL, "writing"))
    try:
        await asyncio.wait_for(probe.queued.wait(), 1)
        assert probe.timeouts == probe.native_calls == probe.recoveries == []
        probe.permits.release()
        state, doc = await asyncio.wait_for(task, 1)
        assert state == "native_reachable" and doc.content == TEXT.rstrip()
        assert len(probe.timeouts) == len(probe.native_calls) == 1 and probe.recoveries == []
        assert not probe.permits.locked()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_attempt_deadline_cancels_queued_source_without_starting_recovery(native_admission):
    """An expired reference still cancels capacity wait; cancellation never launches paid fallback."""
    probe = native_admission
    deadline = asyncio.timeout_at(None)

    async def acquire():
        async with deadline:
            return await probe.sources.acquire(URL, "writing")

    task = asyncio.create_task(acquire())
    try:
        await asyncio.wait_for(probe.queued.wait(), 1)
        deadline.reschedule(asyncio.get_running_loop().time() - 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, 1)
        assert probe.timeouts == probe.native_calls == probe.recoveries == []
        probe.permits.release()
        assert not probe.permits.locked()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_started_native_timeout_cleans_up_and_releases_permit_before_recovery(monkeypatch, native_admission):
    """A stalled active native leg must recover through Parallel without leaking the native worker permit."""
    probe = native_admission
    cleaned = asyncio.Event()

    async def stall(*_args):
        probe.timeouts[-1].reschedule(asyncio.get_running_loop().time() - 1)
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr("harnyx_commons.miner_task_generation.source_fetch._fetch_isolated", stall)
    probe.permits.release()
    state, doc = await asyncio.wait_for(probe.sources.acquire(URL, "writing"), 1)
    assert state == "provider_recovered" and doc.content == TEXT.rstrip()
    assert cleaned.is_set() and not probe.permits.locked()
    assert len(probe.recoveries) == 1


@pytest.mark.parametrize("large", ["unicode", "offsets", "navigation"])
def test_retained_source_budget_counts_unicode_offsets_and_links_without_truncation(monkeypatch, large):
    from harnyx_commons.miner_task_generation import reference_evidence

    store = EvidenceStore()
    first = store.retain(document(), "native_reachable")
    evidence = store.select(first, 1, 2)["evidence_id"]
    before = store.retained_source_bytes
    monkeypatch.setattr(reference_evidence, "MAX_RETAINED_SOURCE_BYTES", before + 20_000)
    big = document(
        "😀" * 10_000 if large == "unicode" else "\n" * 10_000 if large == "offsets" else "x" * 10_000,
        url=URL + "/large",
    )
    if large == "navigation":
        big = replace(big, links=(SourceLink(URL + "/navigation", "large label" * 3_000),))
    with pytest.raises(SourceStorageLimitError):
        store.retain(big, "native_reachable")
    assert store.retained_source_bytes == before and len(store.sources) == 1
    assert store.states[big.requested_url] == "source_storage_limit"
    assert store.citation(evidence).excerpts[0].text == TEXT[: store.evidence[evidence].end]
    small = document("x" * 10_000, url=URL + "/small")
    second = store.retain(small, "native_reachable")
    assert store.sources[second].document.content == small.content
    assert store.retained_source_bytes <= reference_evidence.MAX_RETAINED_SOURCE_BYTES


def test_source_budget_preserves_unchanged_rereads_but_rejects_accumulating_versions(monkeypatch):
    from harnyx_commons.miner_task_generation import reference_evidence

    store = EvidenceStore()
    original = document()
    source = store.retain(original, "native_reachable")
    selected = store.select(source, 1, 2)["evidence_id"]
    before = store.retained_source_bytes
    monkeypatch.setattr(reference_evidence, "MAX_RETAINED_SOURCE_BYTES", before)
    assert store.retain(replace(original), "native_reachable") == source
    assert store.retained_source_bytes == before and store.citation(selected)
    with pytest.raises(SourceStorageLimitError):
        store.retain(replace(original, content=TEXT + "changed"), "native_reachable")
    assert store.retained_source_bytes == before and store.sources[source].document.content == TEXT
    with pytest.raises(ValueError, match="latest check"):
        store.citation(selected)


@pytest.mark.anyio
async def test_source_storage_rejection_is_visible_and_writer_can_use_a_smaller_replacement(monkeypatch):
    from harnyx_commons.miner_task_generation import reference_evidence

    monkeypatch.setattr(reference_evidence, "MAX_RETAINED_SOURCE_BYTES", 2_000)
    reads = []

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            reads.append(url)
            return document(TEXT * 100 if url.endswith("large") else TEXT, url=url)

    store = EvidenceStore()
    capture = CallCapture(CandidateSession("capacity", date.today()), "reference", "gpt-6-luna", "openai")
    feedback = ReferenceFeedback(capture)
    sources = ReferenceSources(Native(), None, store, capture)
    tools = {tool.name: tool for tool in sources.tools(feedback)}

    async def read(url):
        args = json.dumps({"url": url})
        return json.loads(
            await tools["read_source"].on_invoke_tool(
                ToolContext(context=None, tool_name="read_source", tool_call_id="read", tool_arguments=args),
                args,
            )
        )

    feedback.error("read_chunk", "invalid cursor")
    rejected = await read(URL + "/large")
    assert rejected["state"] == "source_storage_limit" and rejected["source_id"] is None
    assert "smaller source" in rejected["instruction"] and feedback.retries == 0
    assert not store.sources and store.retained_source_bytes == 0
    acquired = await read(URL + "/small")
    selected = store.select(acquired["source_id"], 1, 2)["evidence_id"]
    assert render_reference(output([selected]), store, draft()).citations[0].url == URL + "/small"
    assert reads == [URL + "/large", URL + "/small"]
    assert sources.counts["source_storage_limit"] == 1
    assert any(e["response"]["type"] == "reference.source_storage_limit" for e in capture.responses)


def draft(structured=False):
    data = dict(
        question="Which value?",
        answer="88",
        answer_components=["88"],
        scope="2026",
        source_support=[{"url": URL, "evidence": "Alpha"}],
        revision_note="original",
    )
    if not structured:
        return QuestionDraft(**data)
    return StructuredQuestionDraft(
        **data,
        output_schema_json=json.dumps(
            {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "eligible": {"type": "boolean"},
                    "name": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["value", "eligible", "name", "explanation"],
                "additionalProperties": False,
            }
        ),
        structured_answer_json='{"value":88,"eligible":true,"name":"Alpha","explanation":"supported"}',
    )


def document(text=TEXT, *, url=URL, final_url=None):
    return SourceDocument(
        url, final_url or url, "text/plain", text, len(text), (SourceLink("https://example.org/unused", "unused"),)
    )


def output(ids, text="Alpha [[evidence:1]]", **kwargs):
    return RegisteredReferenceOutput(
        answer_text=text, structured_answer_json=None, note=None, citation_evidence_ids=ids, **kwargs
    )


def test_exact_selection_reordered_duplicate_positions_and_same_url_passages():
    store = EvidenceStore()
    source_id = store.retain(document(), "native_reachable")
    first = store.select(source_id, 1, 2)
    second = store.select(source_id, 4, 5)
    answer = render_reference(
        output([second["evidence_id"], first["evidence_id"], second["evidence_id"]]), store, draft()
    )
    assert answer.text == "Alpha [[2]]"
    assert answer.citations[0] == answer.citations[2] != answer.citations[1]
    for item in (first, second):
        excerpt = store.citation(item["evidence_id"]).excerpts[0]
        assert excerpt.text == TEXT[item["excerpt_start"] : item["excerpt_end"]]
        assert (excerpt.start, excerpt.end) == (item["excerpt_start"], item["excerpt_end"])


@pytest.mark.parametrize("marker", ["[[evidence:99]]", "[[1]]", "[[evidence:1]", "evidence:1"])
def test_bad_internal_marker_cannot_be_repaired_or_published(marker):
    store = EvidenceStore()
    source = store.retain(document(), "native_reachable")
    evidence = store.select(source, 1, 2)["evidence_id"]
    with pytest.raises(ValueError):
        render_reference(output([evidence], marker), store, draft())


def test_omitted_unregistered_or_stale_selection_is_rejected():
    store = EvidenceStore()
    source = store.retain(document(final_url=URL + "/final"), "native_reachable")
    evidence = store.select(source, 1, 2)["evidence_id"]
    with pytest.raises(ValueError, match="Unregistered"):
        render_reference(output(["evidence:99"]), store, draft())
    store.observe(URL + "/final", "source_unavailable")
    with pytest.raises(ValueError, match="latest check"):
        render_reference(output([evidence]), store, draft())
    store.retain(document("Different source version"), "native_reachable")
    with pytest.raises(ValueError, match="changed version"):
        store.citation(evidence)


@pytest.mark.parametrize("state", ["native_reachable", "provider_recovered"])
@pytest.mark.parametrize("read_final_url", [False, True])
def test_unchanged_reread_preserves_selected_evidence_and_refreshes_acquisition_metadata(state, read_final_url):
    store = EvidenceStore()
    original = document(final_url=URL + "/final")
    source = store.retain(original, "native_reachable")
    snapshot = store.sources[source]
    selected = store.select(source, 1, 2)
    refreshed = replace(
        original,
        requested_url=original.final_url if read_final_url else URL,
        media_type="text/markdown",
        fetched_bytes=None,
        links=(SourceLink("https://example.org/new", "new link"),),
    )
    assert store.retain(refreshed, state) == source
    answer = render_reference(output([selected["evidence_id"]]), store, draft())
    assert answer.text == "Alpha [[1]]"
    assert answer.citations[0].excerpts[0].text == selected["excerpt"]
    assert answer.citations[0].url == original.final_url
    assert store.view(source)["state"] == state
    assert store.view(source)["links"] == [{"url": "https://example.org/new", "text": "new link"}]
    assert store.sources[source].document == refreshed
    assert snapshot.document == original and snapshot.state == "native_reachable"


def test_redirect_to_already_retained_document_preserves_that_documents_selected_evidence():
    store = EvidenceStore()
    old_source = store.retain(document("Old target", final_url=URL + "/old"), "native_reachable")
    old_evidence = store.select(old_source, 1, 1)["evidence_id"]
    target = document(url=URL + "/final")
    target_source = store.retain(target, "native_reachable")
    evidence = store.select(target_source, 1, 2)["evidence_id"]
    assert store.retain(replace(target, requested_url=URL), "native_reachable") == target_source
    assert (
        render_reference(output([evidence], text=f"Alpha [[{evidence}]]"), store, draft()).citations[0].url
        == target.final_url
    )
    with pytest.raises(ValueError, match="changed version"):
        store.citation(old_evidence)


@pytest.mark.parametrize("state", ["native_reachable", "provider_recovered"])
def test_redirect_target_change_does_not_permanently_alias_distinct_documents(state):
    """A moved redirect must not make subsequent direct reads invalidate evidence from the other target."""
    store = EvidenceStore()
    first = document("First target evidence", final_url=URL + "/first")
    second = document("Second target evidence", final_url=URL + "/second")
    source_a = store.retain(first, state)
    source_b = store.retain(second, state)
    evidence_b = store.select(source_b, 1, 1)["evidence_id"]
    with pytest.raises(ValueError, match="changed version"):
        store.view(source_a)
    source_a = store.retain(replace(first, requested_url=first.final_url), state)
    evidence_a = store.select(source_a, 1, 1)["evidence_id"]
    assert store.retain(replace(second, requested_url=second.final_url), state) == source_b
    answer = render_reference(
        output([evidence_a, evidence_b], text=f"First [[{evidence_a}]] and second [[{evidence_b}]]"), store, draft()
    )
    assert [citation.url for citation in answer.citations] == [first.final_url, second.final_url]
    store.observe(first.final_url, "source_unavailable")
    with pytest.raises(ValueError, match="latest check"):
        store.citation(evidence_a)
    assert store.citation(evidence_b).url == second.final_url


def test_moving_one_redirect_preserves_other_current_aliases_and_scopes_failures():
    """Moving one alias must not detach another alias from its actual target or poison the unrelated target."""
    store = EvidenceStore()
    first = document("First target evidence", final_url=URL + "/first")
    second = document("Second target evidence", url=URL + "/second-alias", final_url=URL + "/second")
    store.retain(first, "native_reachable")
    source_a = store.retain(replace(first, requested_url=URL + "/first-alias"), "native_reachable")
    evidence_a = store.select(source_a, 1, 1)["evidence_id"]
    source_b = store.retain(second, "native_reachable")
    evidence_b = store.select(source_b, 1, 1)["evidence_id"]
    assert store.retain(replace(second, requested_url=URL), "native_reachable") == source_b
    assert store.citation(evidence_a).url == first.final_url
    store.observe(URL + "/first-alias", "source_unavailable")
    with pytest.raises(ValueError, match="latest check"):
        store.citation(evidence_a)
    assert store.citation(evidence_b).url == second.final_url
    store.observe(URL, "source_unavailable")
    with pytest.raises(ValueError, match="latest check"):
        store.citation(evidence_b)


@pytest.mark.parametrize("change", ["content", "final_url"])
def test_changed_reread_invalidates_evidence_even_if_only_redirect_target_changed(change):
    store = EvidenceStore()
    original = document()
    source = store.retain(original, "native_reachable")
    selected = store.select(source, 1, 2)["evidence_id"]
    changed = replace(original, **{change: TEXT + "changed" if change == "content" else URL + "/elsewhere"})
    assert store.retain(changed, "native_reachable") != source
    with pytest.raises(ValueError, match="changed version"):
        render_reference(output([selected]), store, draft())


def test_unchanged_reread_does_not_bypass_later_failure_or_resurrect_an_older_version():
    store = EvidenceStore()
    source = store.retain(document(), "native_reachable")
    selected = store.select(source, 1, 2)["evidence_id"]
    assert store.retain(document(), "native_reachable") == source
    store.observe(URL, "source_unavailable")
    with pytest.raises(ValueError, match="latest check"):
        render_reference(output([selected]), store, draft())
    assert store.retain(document(), "provider_recovered") == source
    assert store.citation(selected).url == URL
    store.retain(document(TEXT + "changed"), "native_reachable")
    assert store.retain(document(), "native_reachable") != source
    with pytest.raises(ValueError, match="changed version"):
        store.citation(selected)


def test_structured_prose_conversion_preserves_atomic_values_and_schema():
    store = EvidenceStore()
    source = store.retain(document(), "native_reachable")
    evidence = store.select(source, 1, 2)["evidence_id"]
    value = {
        "value": 88,
        "eligible": True,
        "name": "evidence: scientific report",
        "explanation": f"Points [[{evidence}]]",
    }
    final = RegisteredReferenceOutput(
        answer_text=None, structured_answer_json=json.dumps(value), note=None, citation_evidence_ids=[evidence]
    )
    answer = render_reference(final, store, draft(True))
    assert json.loads(answer.text) == value | {"explanation": "Points [[1]]"}
    assert type(json.loads(answer.text)["value"]) is int
    assert type(json.loads(answer.text)["eligible"]) is bool


@pytest.mark.parametrize("text", [TEXT, "row\n" * 20_000, "long row " * 20_000, '\x01漢\\"' * 20_000])
def test_paging_retains_and_reconstructs_complete_original_including_long_lines_and_escaped_text(text):
    store = EvidenceStore()
    links = tuple(SourceLink(f"https://example.org/{i}", "navigation " * 100) for i in range(50))
    source_id = store.retain(replace(document(text), links=links), "native_reachable")
    cursor = None
    recovered, recovered_links = [], []
    seen = set()
    while True:
        page = store.view(source_id, cursor)
        assert page["cursor"] not in seen
        seen.add(page["cursor"])
        assert len(json.dumps(page["text"], ensure_ascii=False)) <= SOURCE_PAGE_CHARACTERS
        if page["start_line"] is not None:
            source = store.sources[source_id]
            recovered.append(text[source.starts[page["start_line"] - 1] : source.end(page["end_line"])])
        recovered_links.extend(page["links"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert page["text_complete"] and page["links_complete"]
            break
    assert "".join(recovered) == text
    assert recovered_links == [{"url": link.url, "text": link.text} for link in links]
    assert store.sources[source_id].document.content == text


def test_dense_short_lines_keep_offset_memory_small_and_late_evidence_exact():
    """A valid short-line source must not exhaust worker memory before the writer can navigate it."""
    text = "start" + "\n" * 250_000 + "late evidence " * 10
    original = document(text)
    store = EvidenceStore()
    tracemalloc.start()
    try:
        source_id = store.retain(original, "native_reachable")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < len(text) * 8
    hit = store.search(source_id, "late evidence")["matches"][0]
    assert hit["start_line"] == 250_001
    selected = store.select(source_id, hit["start_line"], hit["end_line"])
    citation = store.citation(selected["evidence_id"])
    assert citation.excerpts[0].start == 250_005
    assert citation.excerpts[0].text == text[selected["excerpt_start"] : selected["excerpt_end"]]


def test_repeated_large_partial_selections_do_not_retain_copies_or_prevent_small_final_reference():
    """Unsubmitted selections must not amplify retained source memory during correction turns."""
    text = "context " * 250_000
    store = EvidenceStore()
    source_id = store.retain(document(text), "native_reachable")
    last_line = len(store.sources[source_id].starts)
    tracemalloc.start()
    try:
        selections = [store.select(source_id, 2, last_line - 1) for _ in range(20)]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len({item["evidence_id"] for item in selections}) == 20
    assert peak < len(text) * 4
    for item in selections:
        with pytest.raises(ValueError, match="total evidence limit"):
            render_reference(output([item["evidence_id"]]), store, draft())
    small = store.select(source_id, 1, 1)
    final = render_reference(output([small["evidence_id"]], text=f"Context [[{small['evidence_id']}]]"), store, draft())
    assert final.citations[0].excerpts[0].text == text[:1_000]


def test_final_evidence_size_includes_duplicate_positions_before_materialization(monkeypatch):
    """Repeated submitted IDs must not bypass the public total or copy oversized excerpts before rejection."""
    store = EvidenceStore()
    source_id = store.retain(document("context " * 10_000), "native_reachable")
    evidence_id = store.select(source_id, 2, len(store.sources[source_id].starts) - 1)["evidence_id"]

    def cannot_copy(*args, **kwargs):
        raise AssertionError("oversized final evidence must be rejected before materialization")

    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence.materialize_citation_slices", cannot_copy
    )
    with pytest.raises(ValueError, match="total evidence limit"):
        render_reference(output([evidence_id, evidence_id]), store, draft())


def test_literal_search_finds_later_cross_segment_text_and_continues_without_losing_overlapping_matches():
    store = EvidenceStore()
    text = "x" * 999 + "literal[.]target" + "x" * 40_000 + "aaa " * (SOURCE_SEARCH_MATCHES + 2)
    source_id = store.retain(document(text), "native_reachable")
    result = store.search(source_id, "literal[.]target")
    assert result["complete"] and len(result["matches"]) == 1
    hit = result["matches"][0]
    assert hit["start_line"] == 1 and hit["end_line"] == 2
    assert "literal[.]target" not in store.view(source_id)["text"]  # Labels split the cross-range query.
    selected = store.select(source_id, hit["start_line"], hit["end_line"])
    assert "literal[.]target" in selected["excerpt"]
    assert store.search(source_id, "LITERAL[.]TARGET")["matches"] == []
    first = store.search(source_id, "aa")
    assert len(first["matches"]) == SOURCE_SEARCH_MATCHES and not first["complete"]
    second = store.search(source_id, "aa", first["next_cursor"])
    third = store.search(source_id, "aa", second["next_cursor"])
    assert sum(len(page["matches"]) for page in (first, second, third)) == 2 * (SOURCE_SEARCH_MATCHES + 2)
    assert third["complete"] and third["next_cursor"] is None


def test_search_continuation_rejects_another_query_without_hiding_earlier_matches():
    """Reusing another query's cursor must return corrective feedback, never a false complete no-match."""
    store = EvidenceStore()
    source = store.retain(document("EARLY " + "later " * (SOURCE_SEARCH_MATCHES + 5)), "native_reachable")
    cursor = store.search(source, "later")["next_cursor"]
    with pytest.raises(ValueError):
        store.search(source, "EARLY", cursor)
    result = store.search(source, "EARLY")
    assert result["complete"] and len(result["matches"]) == 1
    assert result["matches"][0]["start_line"] == 1
    assert store.search(source, "later", cursor)["complete"]


@pytest.mark.anyio
async def test_wrong_search_query_cursor_returns_feedback_and_success_resets_retry_count():
    """A wrong continuation query must be a recoverable model error, without a new fetch or lost evidence."""
    store = EvidenceStore()
    source = store.retain(document("EARLY " + "later " * (SOURCE_SEARCH_MATCHES + 5)), "native_reachable")
    selected = store.select(source, 1, 1)["evidence_id"]
    cursor = store.search(source, "later")["next_cursor"]
    capture = CallCapture(CandidateSession("query-cursor", date.today()), "reference", "gpt-6-luna", "openai")
    feedback = ReferenceFeedback(capture)
    sources = ReferenceSources(SimpleNamespace(), SimpleNamespace(), store, capture)
    tool = next(tool for tool in sources.tools(feedback) if tool.name == "search_text")

    async def search(next_cursor):
        args = json.dumps({"source_id": source, "query": "EARLY", "cursor": next_cursor})
        return json.loads(
            await tool.on_invoke_tool(
                ToolContext(context=None, tool_name="search_text", tool_call_id="search", tool_arguments=args), args
            )
        )

    rejected = await search(cursor)
    assert "search query" in rejected["error"] and feedback.retries == 1
    recovered = await search(None)
    assert recovered["complete"] and recovered["matches"] and feedback.retries == 0
    assert store.citation(selected).url == URL


@pytest.mark.parametrize("failure", ["changed", "failed"])
def test_retained_read_and_search_cursors_cannot_bypass_latest_source_version_or_failed_check(failure):
    store = EvidenceStore()
    source_id = store.retain(document("original " * 20_000), "native_reachable")
    read_cursor = store.view(source_id)["next_cursor"]
    search_cursor = store.search(source_id, "original")["next_cursor"]
    if failure == "changed":
        store.retain(document("changed " * 20_000), "native_reachable")
    else:
        store.observe(URL, "source_unavailable")
    with pytest.raises(ValueError, match="latest check or changed version"):
        store.view(source_id, read_cursor)
    with pytest.raises(ValueError, match="latest check or changed version"):
        store.search(source_id, "original", search_cursor)


def test_wrong_source_operation_and_out_of_range_cursors_cannot_read_an_unrelated_range():
    store = EvidenceStore()
    source_id = store.retain(document("query " * 20_000), "native_reachable")
    read_cursor = store.view(source_id)["next_cursor"]
    search_cursor = store.search(source_id, "query")["next_cursor"]
    with pytest.raises(ValueError, match="another source version or operation"):
        store.view(source_id, search_cursor)
    with pytest.raises(ValueError, match="another source version or operation"):
        store.search(source_id, "query", read_cursor)
    other = store.retain(document(url=URL + "/other"), "native_reachable")
    with pytest.raises(ValueError, match="another source version or operation"):
        store.view(other, read_cursor)
    with pytest.raises(ValueError, match="outside"):
        store.view(source_id, f"{source_id}/read/999999/0")
    with pytest.raises(ValueError, match="empty"):
        store.search(source_id, "")


@pytest.mark.parametrize(
    "text,too_large", [("value " * 7_000, False), ('\x01漢\\"' * 7_000, False), ("value " * 100_000, True)]
)
def test_large_selection_returns_a_bounded_preview_but_retains_complete_public_evidence_and_size_validation(
    text, too_large
):
    store = EvidenceStore()
    source_id = store.retain(document(text), "native_reachable")
    selected = store.select(source_id, 1, len(store.sources[source_id].starts))
    assert len(json.dumps(selected["excerpt"], ensure_ascii=False)) <= SOURCE_PAGE_CHARACTERS
    assert not selected["excerpt_complete"]
    assert text.startswith(selected["excerpt"])
    page = store.view(source_id, selected["excerpt_cursor"])
    assert page["start_line"] == 1
    recovered = []
    while True:
        source = store.sources[source_id]
        recovered.append(text[source.starts[page["start_line"] - 1] : source.end(page["end_line"])])
        if page["end_line"] >= selected["end_line"]:
            break
        page = store.view(source_id, page["next_cursor"])
    assert "".join(recovered) == text
    assert store.citation(selected["evidence_id"]).excerpts[0].text == text
    if too_large:
        with pytest.raises(ValueError, match="total evidence limit"):
            render_reference(output([selected["evidence_id"]]), store, draft())
    else:
        final = render_reference(output([selected["evidence_id"]]), store, draft())
        assert final.citations[0].excerpts[0].text == text


@pytest.mark.parametrize("first,last", [(0, 1), (4, 3), (1, 99), (True, 2)])
def test_invalid_lines_cannot_select_an_unrelated_excerpt(first, last):
    store = EvidenceStore()
    source = store.retain(document(), "native_reachable")
    with pytest.raises(ValueError):
        store.select(source, first, last)


@pytest.mark.anyio
@pytest.mark.parametrize("recover", [False, True])
async def test_read_select_and_render_check_new_source_once_without_following_links(monkeypatch, recover):
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_args: ("93.184.216.34",)
    )
    calls = []

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            calls.append(("native", url))
            if recover:
                raise SourceFetchError("source_unavailable", "403")
            return document()

    class Parallel:
        async def extract_pages(self, request):
            calls.append(("parallel", request.urls[0]))
            return SearchProviderResult(
                ExtractPagesResponse(pages=(ExtractedPage(url=URL, content=TEXT),)),
                ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.001),
            )

    store = EvidenceStore()
    capture = CallCapture(
        CandidateSession("read", date.today()), "reference", "gpt-6-luna", "openai", physical_accounting=True
    )
    sources = ReferenceSources(Native(), Parallel(), store, capture)
    tools = {tool.name: tool for tool in sources.tools(ReferenceFeedback(capture))}
    read_args = json.dumps({"url": URL})
    read_ctx = ToolContext(context=None, tool_name="read_source", tool_call_id="read", tool_arguments=read_args)
    view = json.loads(await tools["read_source"].on_invoke_tool(read_ctx, read_args))
    selected = json.loads(
        await tools["select_evidence"].on_invoke_tool(
            ToolContext(context=None, tool_name="select_evidence", tool_call_id="select", tool_arguments="{}"),
            json.dumps({"source_id": view["source_id"], "start_line": 1, "end_line": 2}),
        )
    )
    before = list(calls)
    final = render_reference(output([selected["evidence_id"]]), store, draft())
    assert final.citations[0].excerpts[0].text == selected["excerpt"]
    assert calls == before == ([("native", URL), ("parallel", URL)] if recover else [("native", URL)])
    assert view["state"] == ("provider_recovered" if recover else "native_reachable")
    capture.finish()
    assert capture.session.missing_usage_calls == 0
    assert capture.session.tool_usage.actual_total_cost_usd == (0.001 if recover else 0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    [
        "source_fetch_rejected",
        "source_unavailable",
        "source_extraction_limit",
        "timeout",
        "auth",
        "target",
        "null",
        "blank",
    ],
)
async def test_source_security_and_provider_faults_keep_correct_recovery_boundary(monkeypatch, failure):
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_args: ("93.184.216.34",)
    )
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_evidence.FETCH_TIMEOUT_SECONDS", 0.01)
    calls = []

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            if failure == "timeout":
                raise TimeoutError("native acquisition timed out")
            raise SourceFetchError(failure if failure.startswith("source_") else "source_unavailable", "failed")

    class Parallel:
        async def extract_pages(self, request):
            calls.append(request)
            if failure == "auth":
                raise ToolProviderError("auth", http_status=401)
            response = (
                ExtractPagesResponse(errors=(PageExtractionError(url=URL, error_type="unavailable"),))
                if failure == "target"
                else ExtractPagesResponse(
                    pages=(
                        ExtractedPage(
                            url=URL, content=None if failure == "null" else "  " if failure == "blank" else TEXT
                        ),
                    )
                )
            )
            return SearchProviderResult(
                response, ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.001)
            )

    store = EvidenceStore()
    capture = CallCapture(
        CandidateSession("failure", date.today()), "reference", "gpt-6-luna", "openai", physical_accounting=True
    )
    sources = ReferenceSources(Native(), Parallel(), store, capture)
    if failure == "auth":
        with pytest.raises(ToolProviderError):
            await sources.acquire(URL, "writing")
        assert capture.tool_usage.actual_total_cost_usd is None
    else:
        state, doc = await sources.acquire(URL, "writing")
        assert (doc is None) == (failure in {"source_fetch_rejected", "target", "null", "blank"})
        assert state == (
            "source_fetch_rejected"
            if failure == "source_fetch_rejected"
            else "source_unavailable"
            if failure in {"target", "null", "blank"}
            else "provider_recovered"
        )
    assert len(calls) == (0 if failure == "source_fetch_rejected" else 1)


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "returned_url",
    ["https://127.0.0.1/report", "https://user:secret@example.org/report", "https://example.org/report?token=secret"],
)
async def test_parallel_cannot_register_private_or_credential_bearing_returned_url(monkeypatch, returned_url):
    def public_addresses(host, port):
        if host == "127.0.0.1":
            raise SourceFetchError("source_fetch_rejected", "non-public address")
        return ("93.184.216.34",)

    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_evidence._public_addresses", public_addresses)

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            raise SourceFetchError("source_unavailable", "403")

    class Parallel:
        async def extract_pages(self, request):
            return SearchProviderResult(
                ExtractPagesResponse(pages=(ExtractedPage(url=returned_url, content=TEXT),)),
                ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.001),
            )

    store = EvidenceStore()
    capture = CallCapture(CandidateSession("unsafe-return", date.today()), "reference", "gpt-6-luna", "openai")
    state, document = await ReferenceSources(Native(), Parallel(), store, capture).acquire(URL, "writing")
    assert state == "source_fetch_rejected" and document is None
    assert not store.sources and not store.evidence
    assert capture.tool_usage.search_tool.call_count == 1
    assert capture.tool_usage.actual_total_cost_usd == 0.001
