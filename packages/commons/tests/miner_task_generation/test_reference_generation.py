"""Protect reference lifetimes, quoted handoffs and terminal proposal agreement."""

import asyncio
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace

import httpx
import httpx2
import pytest

from harnyx_commons.errors import ToolProviderError, ToolResponseTooLargeError
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
from harnyx_commons.miner_task_generation.contracts import (
    BatchTerminalGenerationError,
    CandidateSession,
    CandidateStageError,
    QuestionDraft,
    RegisteredReferenceOutput,
    ReviewDecision,
    SourceSupport,
    StructuredQuestionDraft,
)
from harnyx_commons.miner_task_generation.reference_generation import (
    build_reference_parallel,
)
from harnyx_commons.miner_task_generation.reference_generation import (
    derive_reference as run_reference,
)
from harnyx_commons.miner_task_generation.source_fetch import PublicSourceFetcher, SourceDocument, SourceFetchError

pytestmark = pytest.mark.anyio("asyncio")


async def derive_reference(*args, **kwargs):
    """Direct stage tests explicitly own their provider; production borrows from the runner."""
    parallel = build_reference_parallel()
    try:
        return await run_reference(*args, parallel=parallel, **kwargs)
    finally:
        await parallel.aclose()


def draft():
    return QuestionDraft(
        question="Which city is named in the report?",
        answer="Paris",
        answer_components=["Paris"],
        scope="Named report",
        source_support=[SourceSupport(url="https://example.org/report", evidence="Paris")],
        revision_note="Initial supported selection",
    )


def session(task_id="reference"):
    return CandidateSession(
        task_id=task_id,
        effective_date=datetime.now(UTC).date(),
        source_support=draft().source_support,
        author_history=[{"role": "assistant", "content": "author research", "encrypted_content": "private"}],
    )


def install_stage(monkeypatch, execute):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")

    async def fetch(self, url, **kwargs):
        return SourceDocument(url, url, "text/plain", "Paris is named in the report", 28)

    async def stage(self, inputs, sources, capture):
        prompt = json.loads(inputs)
        assert prompt["quoted_author_research"] == [{"role": "assistant", "content": "author research"}]
        assert "reachability" not in prompt
        assert "encrypted_content" not in inputs
        capture.attempted += 1
        capture.account_response(
            f"resp_{capture.attempted}",
            LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, web_search_calls=0),
        )
        return await execute(sources.store, sources.fetcher, prompt)

    monkeypatch.setattr(PublicSourceFetcher, "fetch", fetch)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", stage)


def populate(store, *, answer="Paris", note=None):
    source = store.retain(
        SourceDocument(
            "https://example.org/report", "https://example.org/report", "text/plain", "Paris is named in the report", 28
        ),
        "native_reachable",
    )
    evidence = store.select(source, 1, 1)["evidence_id"]
    return RegisteredReferenceOutput(
        answer_text=f"{answer} [[{evidence}]]", structured_answer_json=None, note=note, citation_evidence_ids=[evidence]
    )


@pytest.mark.parametrize("manual_capture", [False, True])
async def test_reference_full_requests_require_manual_opt_in_without_changing_physical_costs(
    monkeypatch, manual_capture
):
    from harnyx_commons.miner_task_generation.agent_runner import record_openai_request

    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    captures = []

    async def writer(self, inputs, sources, capture):
        captures.append(capture)
        for number in range(2):
            await record_openai_request(
                httpx2.Request("POST", "https://api.example.org/responses", json={"input": inputs})
            )
            capture.account_response(
                f"resp_{number}",
                LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, web_search_calls=0),
            )
        return populate(sources.store)

    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", writer)
    observed = session()
    result = await derive_reference(
        draft(),
        None,
        observed,
        monotonic() + 5,
        client=None,
        slots=asyncio.Semaphore(1),
        fetcher=PublicSourceFetcher(),
        capture_full_requests=manual_capture,
    )
    assert result.reference_answer.text == "Paris [[1]]"
    assert len(captures) == 1 and len(captures[0].requests) == (2 if manual_capture else 0)
    assert observed.attempted_calls == observed.tool_usage.llm.call_count == 2
    assert observed.missing_usage_calls == 0 and observed.tool_usage.actual_total_cost_usd is not None


@pytest.mark.parametrize(
    ("proposal_description", "description", "value", "expected_correction"),
    [
        ("Paris", "Paris [[evidence:1]]", True, False),
        ("Paris", "[[evidence:1]] Paris", True, False),
        ("Paris", "Paris [[evidence:1]]  ", True, False),
        ("Paris", "Paris[[evidence:1]]", True, False),
        ("Paris", "[[evidence:1]] [[evidence:2]] Paris", True, False),
        ("Paris", "Paris [[evidence:1]] [[evidence:2]]", True, False),
        ("Paris", "London [[evidence:1]]", True, True),
        ("Paris", "Paris [[evidence:1]]", 1, True),
        ("Paris", "Paris [[evidence:1]]", False, True),
        ("Paris", "Paris  ", True, True),
        ("Paris\nReport", "Paris\n[[evidence:1]] Report", True, False),
        ("Paris\nReport", "Paris [[evidence:1]]\nReport", True, False),
        ("Paris\r\nReport", "Paris\r\n[[evidence:1]] Report", True, False),
        ("Paris\r\nReport", "Paris [[evidence:1]]\r\nReport", True, False),
        ("Paris\nReport", "London\n[[evidence:1]] Report", True, True),
        ("Paris\nReport", "Paris [[evidence:1]] Report", True, True),
        ("Paris\nReport", "Paris\n Report", True, True),
        ("Items:\n  Alpha", "Items:\n  [[evidence:1]] Alpha", True, False),
        ("Items:\r\n  Alpha", "Items:\r\n  [[evidence:1]] Alpha", True, False),
        ("Items:\n\tAlpha", "Items:\n\t[[evidence:1]] Alpha", True, False),
        ("Items:\n  Alpha", "Items:\n  [[evidence:1]] [[evidence:2]] Alpha", True, False),
        ("Items:\n  Alpha", "Items:\n [[evidence:1]] Alpha", True, True),
        ("Items:\n  Alpha", "Items:\n   [[evidence:1]] Alpha", True, True),
        ("Items:\n  Alpha", "Items:\n  [[evidence:1]] Beta", True, True),
    ],
)
async def test_structured_citations_do_not_hide_value_changes_or_reject_unchanged_facts(
    monkeypatch, proposal_description, description, value, expected_correction
):
    proposal = {"items": [{"description": proposal_description, "eligible": True}], "value": 88}
    structured = StructuredQuestionDraft(
        **draft().model_dump(),
        output_schema_json=json.dumps({"type": "object"}),
        structured_answer_json=json.dumps(proposal),
    )

    async def execute(store, _fetcher, _prompt):
        selected = populate(store)
        second_id = store.select("source:1", 1, 1)["evidence_id"]
        return RegisteredReferenceOutput(
            answer_text=None,
            structured_answer_json=json.dumps(
                {"items": [{"description": description, "eligible": value}], "value": 88}
            ),
            note=None,
            citation_evidence_ids=[*selected.citation_evidence_ids, second_id],
        )

    install_stage(monkeypatch, execute)
    result = await derive_reference(
        structured,
        None,
        session(),
        monotonic() + 5,
        client=None,
        slots=asyncio.Semaphore(1),
        fetcher=PublicSourceFetcher(),
    )
    assert result.factual_correction is expected_correction
    assert result.reference_answer is not None
    rendered = json.loads(result.reference_answer.text)
    assert rendered["items"][0]["description"] == description.replace("[[evidence:1]]", "[[1]]").replace(
        "[[evidence:2]]", "[[2]]"
    )
    assert rendered["items"][0]["eligible"] == value


@pytest.mark.parametrize("failure", ["model", "429", "503", "transport"])
async def test_transient_reference_retry_uses_fresh_store_and_retains_prior_costs(monkeypatch, failure):
    stores = []

    async def execute(store, fetcher, prompt):
        stores.append(store)
        result = populate(store)
        if len(stores) == 1:
            if failure == "model":
                raise CandidateStageError("transient_provider", "reference", "interrupted")
            if failure == "transport":
                raise ToolProviderError("Parallel interrupted") from httpx.ReadError("connection lost")
            raise ToolProviderError("Parallel interrupted", http_status=int(failure))
        return result

    install_stage(monkeypatch, execute)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *_args: 0)
    observed = session()
    result = await derive_reference(
        draft(), None, observed, monotonic() + 5, client=None, slots=asyncio.Semaphore(1), fetcher=PublicSourceFetcher()
    )
    assert result.reference_answer.text == "Paris [[1]]"
    assert len(stores) == 2 and stores[0] is not stores[1]
    assert observed.attempted_calls == observed.tool_usage.llm.call_count == 2
    assert observed.missing_usage_calls == 0


@pytest.mark.parametrize("status", [None, 400, 401, 403, 404])
async def test_parallel_contract_and_auth_faults_do_not_retry_the_reference(monkeypatch, status):
    """Provider/schema/auth faults must not be retried as temporary outages."""
    calls = []

    async def execute(store, _fetcher, _prompt):
        calls.append(store)
        raise ToolProviderError("invalid provider response", http_status=status) from ValueError("schema")

    install_stage(monkeypatch, execute)
    observed = session()
    expected = BatchTerminalGenerationError if status in {401, 403, 404} else ToolProviderError
    with pytest.raises(expected):
        await derive_reference(
            draft(),
            None,
            observed,
            monotonic() + 5,
            client=None,
            slots=asyncio.Semaphore(1),
            fetcher=PublicSourceFetcher(),
        )
    assert len(calls) == observed.attempted_calls == 1


async def test_read_parallel_retry_exhaustion_retains_unknown_cost_and_rechecks_source(monkeypatch):
    """The real adapter's exhausted 429 must reach a fresh reference attempt."""
    from harnyx_commons.miner_task_generation import reference_generation
    from harnyx_commons.tools.parallel import ParallelClient

    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *_args: 0)
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_args: ("93.184.216.34",)
    )
    requests = []
    native_reads = []
    observations = []
    monkeypatch.setattr(
        reference_generation, "update_generation_best_effort", lambda generation, **values: observations.append(values)
    )

    async def no_backoff(self, attempt):
        pass

    async def unavailable(self, url, *, timeout_seconds=None):
        native_reads.append(url)
        raise SourceFetchError("source_unavailable", "403")

    def respond(request):
        requests.append(request)
        if len(requests) <= 2:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "extract_id": "extract_retry",
                "session_id": "source_retry",
                "results": [{"url": "https://example.org/report", "full_content": "Paris"}],
                "errors": [],
            },
        )

    async def writer(self, inputs, sources, capture):
        assert "reachability" not in json.loads(inputs)
        capture.attempted += 1
        capture.account_response(
            f"read_retry_{capture.attempted}", LlmUsage(prompt_tokens=10, completion_tokens=10, web_search_calls=0)
        )
        state, document = await sources.acquire("https://example.org/report", "writing")
        assert state == "provider_recovered" and document is not None
        return populate(sources.store)

    monkeypatch.setattr(ParallelClient, "_sleep", no_backoff)
    monkeypatch.setattr(PublicSourceFetcher, "fetch", unavailable)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", writer)
    async with httpx.AsyncClient(base_url="https://parallel.example", transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(
            reference_generation, "ParallelClient", lambda **kwargs: ParallelClient(client=http, **kwargs)
        )
        observed = session()
        result = await derive_reference(
            draft(),
            None,
            observed,
            monotonic() + 5,
            client=None,
            slots=asyncio.Semaphore(1),
            fetcher=PublicSourceFetcher(),
        )
    assert result.reference_answer.text == "Paris [[1]]"
    assert len(native_reads) == 2 and len(requests) == 3
    assert observed.tool_usage.search_tool.call_count == 2
    assert observed.tool_usage.actual_total_cost_usd is None
    assert observed.stage_summaries[-1].available_cost_usd > 0.001
    model_observation = next(item for item in observations if "usage_details" in item)
    assert model_observation["cost_details"]["total"] > 0
    assert model_observation["metadata"]["missing_usage_calls"] == 0


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "retry",
        "retry_missing",
        "retry_source_unknown",
        "source_unknown",
        "terminal",
        "cancelled",
        "missing",
        "unpriced",
    ],
)
async def test_langfuse_retains_each_attempts_model_usage_cost_and_missing_receipts(monkeypatch, outcome):
    """Failure/cancellation receipts stay visible without counting earlier retries or source spend again."""
    from harnyx_commons.miner_task_generation import reference_generation

    generations = []
    model_costs = []

    @contextmanager
    def generation(**kwargs):
        updates = []
        generations.append(updates)
        yield SimpleNamespace(update=lambda **values: updates.append(values))

    async def unused(*args):
        raise AssertionError("replaced by the receipt-aware writer below")

    install_stage(monkeypatch, unused)
    monkeypatch.setattr(reference_generation, "start_llm_generation", generation)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *_args: 0)
    if outcome == "unpriced":
        monkeypatch.setattr(
            "harnyx_commons.miner_task_generation.agent_runner.settled_generation_llm_cost", lambda **_: None
        )

    async def writer(self, inputs, sources, capture):
        attempt = len(generations)
        before = capture.available_model_cost_usd
        capture.attempted += 1
        capture.account_response(
            f"observed_{attempt}",
            LlmUsage(
                prompt_tokens=attempt * 10,
                completion_tokens=attempt * 10,
                total_tokens=attempt * 20,
                web_search_calls=0,
            ),
        )
        model_costs.append(capture.available_model_cost_usd - before)
        await sources._provider_call("search_sources", "report", lambda: paid_source_result())
        if outcome == "missing" or (outcome == "retry_missing" and attempt == 1):
            capture.attempted += 1  # one real attempted response has no usage receipt
        if outcome in {"retry", "retry_missing", "retry_source_unknown"} and attempt == 1:
            raise ToolProviderError("rate limited", http_status=429)
        if outcome == "terminal":
            raise CandidateStageError("contract_invalid", "reference", "bad output")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return populate(sources.store)

    async def paid_source_result():
        from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

        return SearchProviderResult(
            None,
            ProviderBillingMetadata(
                "parallel",
                "request_body",
                actual_cost_usd=None if outcome in {"source_unknown", "retry_source_unknown"} else 0.001,
            ),
        )

    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", writer)
    observed = session()
    kwargs = dict(client=None, slots=asyncio.Semaphore(1), fetcher=PublicSourceFetcher())
    if outcome in {"terminal", "cancelled"}:
        with pytest.raises(CandidateStageError if outcome == "terminal" else asyncio.CancelledError):
            await derive_reference(draft(), None, observed, monotonic() + 5, **kwargs)
    else:
        await derive_reference(draft(), None, observed, monotonic() + 5, **kwargs)
    assert len(generations) == (2 if outcome in {"retry", "retry_missing", "retry_source_unknown"} else 1)
    for attempt, updates in enumerate(generations, 1):
        observation = next(update for update in updates if "usage_details" in update)
        assert observation["usage_details"] == {
            "input": attempt * 10,
            "output": attempt * 10,
            "total": attempt * 20,
            "reasoning": 0,
        }
        missing = int(outcome == "missing" or (outcome == "retry_missing" and attempt == 1))
        assert observation["metadata"]["missing_usage_calls"] == missing
        source_cost = 0 if outcome in {"source_unknown", "retry_source_unknown"} else 0.001
        assert observation["metadata"]["available_cost_usd"] == pytest.approx(model_costs[attempt - 1] + source_cost)
        if missing or outcome == "unpriced":
            assert "cost_details" not in observation
            assert observation["metadata"]["model_actual_cost_usd"] is None
        else:
            assert observation["cost_details"]["total"] == pytest.approx(model_costs[attempt - 1])
    assert observed.tool_usage.llm.call_count == len(generations)
    assert observed.missing_usage_calls == int(outcome in {"missing", "retry_missing"})
    if observed.missing_usage_calls or outcome in {"source_unknown", "retry_source_unknown", "unpriced"}:
        assert observed.tool_usage.actual_total_cost_usd is None


async def test_source_reads_and_writer_share_one_attempt_deadline(monkeypatch):
    """Source reads must not give the writer a fresh budget or escape cancellation."""
    events = []
    timeouts = []
    now = monotonic()
    started = now
    original_timeout_at = asyncio.timeout_at

    def timeout_at(deadline):
        context = original_timeout_at(deadline)
        timeouts.append(context)
        return context

    monkeypatch.setattr(asyncio, "timeout_at", timeout_at)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.monotonic", lambda: now)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.monotonic", lambda: now)

    async def execute(store, fetcher, prompt):
        nonlocal now
        events.append("writer")
        await fetcher.fetch("https://example.org/report")
        assert len(timeouts) == 1 and timeouts[0].when() == started + 60
        now = started + 80
        timeouts[0].reschedule(asyncio.get_running_loop().time() - 1)
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cancelled")

    install_stage(monkeypatch, execute)

    async def slow_fetch(self, url, **kwargs):
        nonlocal now
        now += 35
        return SourceDocument(url, url, "text/plain", "Paris", 5)

    monkeypatch.setattr(PublicSourceFetcher, "fetch", slow_fetch)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.REFERENCE_TIMEOUT_SECONDS", 60)
    observed = session()
    with pytest.raises(TimeoutError):
        await derive_reference(
            draft(),
            None,
            observed,
            started + 80,
            client=None,
            slots=asyncio.Semaphore(1),
            fetcher=PublicSourceFetcher(),
        )
    assert events == ["writer", "cancelled"]
    assert observed.attempted_calls == 1


async def test_synchronous_projection_cannot_publish_after_deadline(monkeypatch):
    """Async timeout callbacks cannot reject a projection that never yields."""
    from harnyx_commons.miner_task_generation import reference_generation

    async def execute(store, _fetcher, _prompt):
        return populate(store)

    install_stage(monkeypatch, execute)
    now = monotonic()
    original = reference_generation.render_reference

    def late_projection(*args):
        nonlocal now
        result = original(*args)
        now += 601
        return result

    monkeypatch.setattr(reference_generation, "monotonic", lambda: now)
    monkeypatch.setattr(reference_generation, "render_reference", late_projection)
    observed = session()
    with pytest.raises(TimeoutError, match="during finalization"):
        await derive_reference(
            draft(), None, observed, now + 600, client=None, slots=asyncio.Semaphore(1), fetcher=PublicSourceFetcher()
        )
    assert observed.tool_usage.llm.call_count == 2  # transient retry retains each completed receipt


@pytest.mark.parametrize("configuration", ["missing_key", "invalid_concurrency", "invalid_constructor"])
async def test_shared_reference_setup_fault_is_batch_terminal_before_any_source_call(monkeypatch, configuration):
    from harnyx_commons.miner_task_generation import reference_generation

    monkeypatch.setenv("PARALLEL_API_KEY", "" if configuration == "missing_key" else "offline-test")
    if configuration == "invalid_concurrency":
        monkeypatch.setenv("PARALLEL_MAX_CONCURRENT", "invalid-integer")
    if configuration == "invalid_constructor":

        def invalid_client(**kwargs):
            raise ValueError("invalid base URL")

        monkeypatch.setattr(reference_generation, "ParallelClient", invalid_client)
    observed = session()
    runner = GenerationAgentRunner(project_id=None, judge=None)
    with pytest.raises(BatchTerminalGenerationError) as error:
        await runner.derive_reference(draft(), None, observed, monotonic() + 5)
    assert error.value.failure_class == "provider_configuration" and error.value.stage == "reference"
    assert observed.attempted_calls == observed.tool_usage.search_tool.call_count == 0


async def test_reference_recovery_limits_provider_bytes_before_parsing_and_retains_unknown_cost(monkeypatch):
    """An oversized provider JSON response must not enter the reference worker unbounded."""
    from harnyx_commons.miner_task_generation import reference_generation

    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    monkeypatch.setattr(reference_generation, "MAX_RESPONSE_BYTES", 16)
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_args: ("93.184.216.34",)
    )

    async def failed_fetch(self, url, *, timeout_seconds=None):
        raise SourceFetchError("source_unavailable", "403")

    monkeypatch.setattr(PublicSourceFetcher, "fetch", failed_fetch)

    async def writer(self, inputs, sources, capture):
        await sources.acquire("https://example.org/report", "writing")
        raise AssertionError("oversized response cannot return a source")

    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", writer)
    factory = reference_generation.ParallelClient
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=b"invalid JSON larger than permitted bytes")

    async with httpx.AsyncClient(base_url="https://parallel.example", transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(reference_generation, "ParallelClient", lambda **kwargs: factory(client=http, **kwargs))
        observed = session()
        with pytest.raises(ToolResponseTooLargeError):
            await derive_reference(
                draft(),
                None,
                observed,
                monotonic() + 5,
                client=None,
                slots=asyncio.Semaphore(1),
                fetcher=PublicSourceFetcher(),
            )
    assert len(requests) == 1 and observed.attempted_calls == 0
    assert observed.tool_usage.search_tool.call_count == 1
    assert observed.tool_usage.actual_total_cost_usd is None


@pytest.mark.parametrize("operation", ["search", "extract"])
async def test_reference_parallel_keeps_private_queries_and_originals_out_of_info_logs(monkeypatch, caplog, operation):
    """Reference provider calls must retain billing and usable content without publishing raw payloads at INFO."""
    from harnyx_commons.miner_task_generation import reference_generation
    from harnyx_commons.tools.extraction_models import ExtractPagesRequest
    from harnyx_commons.tools.search_models import SearchWebSearchRequest

    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    caplog.set_level(logging.INFO, logger="harnyx_commons.tools.parallel.calls")
    private_query, private_original = "unpublished reference query", "private original source content"

    def respond(request):
        payload = (
            {
                "search_id": "search_ref",
                "results": [{"url": "https://example.org/report", "excerpts": [private_original]}],
            }
            if operation == "search"
            else {
                "extract_id": "extract_ref",
                "session_id": "session_ref",
                "results": [{"url": "https://example.org/report", "full_content": private_original}],
                "errors": [],
            }
        )
        return httpx.Response(200, json=payload)

    factory = reference_generation.ParallelClient
    async with httpx.AsyncClient(base_url="https://parallel.example", transport=httpx.MockTransport(respond)) as http:
        monkeypatch.setattr(reference_generation, "ParallelClient", lambda **kwargs: factory(client=http, **kwargs))
        parallel = build_reference_parallel()
        try:
            result = (
                await parallel.search_web(SearchWebSearchRequest(provider="parallel", search_queries=(private_query,)))
                if operation == "search"
                else await parallel.extract_pages(ExtractPagesRequest(urls=("https://example.org/report",)))
            )
        finally:
            await parallel.aclose()
    assert private_original in result.response.model_dump_json()
    assert result.billing is not None and result.billing.actual_cost_usd is not None
    record = next(record for record in caplog.records if record.msg == "parallel.request.complete")
    assert record.data["status_code"] == 200 and record.data["attempts"] == 1
    assert "json_fields" not in record.__dict__
    assert private_query not in str(record.__dict__) and private_original not in str(record.__dict__)


@pytest.mark.parametrize("judge_result", ["agree", "disagree", "excess", "malformed", "provider", "deadline"])
async def test_terminal_plain_reference_uses_pro_text_and_note_and_never_passes_failed_comparison(
    monkeypatch, judge_result
):
    """An unreported rewrite or failed comparison must not authorize publication."""
    requests = []
    timeouts = []
    original_timeout = asyncio.timeout

    def timeout(delay):
        context = original_timeout(delay)
        timeouts.append(context)
        return context

    monkeypatch.setattr(asyncio, "timeout", timeout)

    async def execute(workspace, _fetcher, _prompt):
        return populate(workspace, answer="The city of Paris", note="No other city is an answer.")

    install_stage(monkeypatch, execute)

    class Judge:
        async def invoke(self, request):
            requests.append(request)
            if judge_result == "provider":
                raise RuntimeError("provider failed")
            if judge_result == "deadline":
                timeouts[-1].reschedule(asyncio.get_running_loop().time() - 1)
                await asyncio.Event().wait()
            payload = {
                "expected_components": [{"component_id": "city", "is_correct": judge_result != "disagree"}],
                "excessive_components": [{"component_id": "extra-city"}] if judge_result == "excess" else [],
            }
            return SimpleNamespace(
                raw_text="{}" if judge_result == "malformed" else json.dumps(payload),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

    runner = GenerationAgentRunner(project_id=None, judge=Judge())
    observed = session()
    deadline = monotonic() + 5
    if judge_result in {"provider", "malformed", "deadline"}:
        error_type = {"provider": RuntimeError, "malformed": ValueError, "deadline": TimeoutError}[judge_result]
        with pytest.raises(error_type):
            await runner.derive_reference(draft(), ReviewDecision(passed=True, feedback=""), observed, deadline)
    else:
        result = await runner.derive_reference(draft(), ReviewDecision(passed=True, feedback=""), observed, deadline)
        assert result.reference_answer is not None
        assert (result.unsupported_reason is None) == (judge_result == "agree")
        if judge_result != "agree":
            assert "disagrees" in result.unsupported_reason
    await runner._reference_parallel_client.aclose()
    request = requests[0]
    assert request.model == "gemini-3.1-pro-preview" and request.reasoning_effort == "high"
    payload = json.loads(request.messages[-1].content[0].text.removeprefix("Payload:\n"))
    assert payload["reference_answer"] == "Paris"
    assert payload["miner_answer"] == "The city of Paris [[1]]"
    assert payload["miner_note"] == "No other city is an answer."
    assert observed.stage_summaries[-1].stage == "reference_assessment"


async def test_runner_shares_fetch_capacity_across_reference_attempts_and_releases_on_cancellation(monkeypatch):
    """Overlapping references must share five workers, including after cancellation."""
    from harnyx_commons.miner_task_generation import source_fetch

    active = 0
    maximum = 0
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def fetch(url, _kind):
        return source_fetch._FetchedBody(url, "text/plain", "", b"Paris")

    async def extract(*_args):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        if active == 5:
            admitted.set()
        try:
            await release.wait()
            return source_fetch._ExtractedSourcePayload(content="Paris")
        finally:
            active -= 1

    monkeypatch.setattr(source_fetch, "_fetch_isolated", fetch)
    monkeypatch.setattr(source_fetch, "_extract_isolated", extract)

    async def execute(workspace, fetcher, _prompt):
        await fetcher.fetch("https://example.org/report", document_kind="text")
        return populate(workspace)

    original_fetch = PublicSourceFetcher.fetch
    install_stage(monkeypatch, execute)
    monkeypatch.setattr(PublicSourceFetcher, "fetch", original_fetch)
    runner = GenerationAgentRunner(project_id=None, judge=None)
    tasks = [
        asyncio.create_task(runner.derive_reference(draft(), None, session(str(index)), monotonic() + 5))
        for index in range(10)
    ]
    try:
        await asyncio.wait_for(admitted.wait(), 1)
        # Allow every reference to reach admission; an independent limiter admits all ten.
        await asyncio.sleep(0.02)
        assert active == maximum == 5
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        release.set()
        results = await asyncio.gather(*tasks[1:])
        assert all(result.reference_answer for result in results)
        assert maximum == 5 and active == 0
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runner._reference_parallel_client is not None:
            await runner._reference_parallel_client.aclose()


async def test_unused_author_urls_do_not_delay_writer_or_trigger_source_calls(monkeypatch):
    """Unneeded historical URLs must not consume the writer's deadline or source budget."""
    written = []

    async def execute(store, _fetcher, prompt):
        written.append(prompt)
        return populate(store)

    install_stage(monkeypatch, execute)

    async def forbidden_fetch(*args, **kwargs):
        raise AssertionError("unread author URLs must not be checked")

    monkeypatch.setattr(PublicSourceFetcher, "fetch", forbidden_fetch)
    observed = session()
    observed.source_support.extend(
        SourceSupport(url=f"https://example.org/stale/{index}", evidence="historical memo") for index in range(100)
    )
    result = await derive_reference(
        draft(), None, observed, monotonic() + 5, client=None, slots=asyncio.Semaphore(1), fetcher=PublicSourceFetcher()
    )
    assert result.reference_answer is not None and len(written) == 1
    assert len(written[0]["source_support"]) == 101
    assert observed.tool_usage.search_tool.call_count == 0


@pytest.mark.parametrize("operation", ["search", "recovery"])
async def test_parallel_capacity_is_shared_across_candidates_and_survives_cancellation(monkeypatch, operation):
    """Independent candidate clients must not multiply provider capacity or close a peer's transport."""
    from harnyx_commons.tools.search_models import SearchWebSearchRequest

    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    monkeypatch.setenv("PARALLEL_MAX_CONCURRENT", "2")
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_: ("93.184.216.34",)
    )
    active = maximum = 0
    admitted, writers_ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    providers = []
    closes = []

    async def respond(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            admitted.set()
        try:
            await release.wait()
            payload = (
                {"search_id": "shared_search", "results": []}
                if operation == "search"
                else {
                    "extract_id": "shared_extract",
                    "session_id": "shared_session",
                    "results": [{"url": "https://example.org/report", "full_content": "Paris"}],
                    "errors": [],
                }
            )
            return httpx.Response(200, json=payload)
        finally:
            active -= 1

    http = httpx.AsyncClient(base_url="https://parallel.example", transport=httpx.MockTransport(respond))
    monkeypatch.setattr("harnyx_commons.tools.parallel.httpx.AsyncClient", lambda **kwargs: http)

    async def no_source(self, url, *, timeout_seconds=None):
        raise SourceFetchError("source_unavailable", "404")

    async def writer(self, inputs, sources, capture):
        providers.append(sources.parallel)
        if len(providers) == 6:
            writers_ready.set()
        capture.attempted += 1
        capture.account_response(
            f"shared_{capture.session.task_id}", LlmUsage(prompt_tokens=10, completion_tokens=10, web_search_calls=0)
        )
        if operation == "search":
            await sources._provider_call(
                "search_sources",
                "Paris report",
                lambda: sources.parallel.search_web(
                    SearchWebSearchRequest(provider="parallel", search_queries=("Paris",))
                ),
            )
        else:
            state, document = await sources.acquire("https://example.org/report", "writing")
            assert state == "provider_recovered" and document is not None
        return populate(sources.store)

    async def close_model():
        closes.append("model")

    async def close_judge():
        closes.append("judge")

    monkeypatch.setattr(PublicSourceFetcher, "fetch", no_source)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run", writer)
    model = SimpleNamespace(_client=SimpleNamespace(event_hooks={"request": [], "response": []}), close=close_model)
    runner = GenerationAgentRunner(project_id=None, judge=SimpleNamespace(aclose=close_judge), openai_client=model)
    tasks = [
        asyncio.create_task(runner.derive_reference(draft(), None, session(str(i)), monotonic() + 10)) for i in range(6)
    ]
    try:
        await asyncio.wait_for(writers_ready.wait(), 5)
        await asyncio.wait_for(admitted.wait(), 5)
        assert active == maximum == 2 and len({id(provider) for provider in providers}) == 1
        tasks[0].cancel()
        await asyncio.gather(tasks[0], return_exceptions=True)
        assert not http.is_closed
        release.set()
        results = await asyncio.gather(*tasks[1:])
        assert all(result.reference_answer is not None for result in results)
        await runner.derive_reference(draft(), None, session("later"), monotonic() + 5)
        assert maximum == 2 and active == 0 and not http.is_closed
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.aclose()
        await http.aclose()
    assert http.is_closed and closes == ["model", "judge"]


async def test_shared_parallel_closes_when_other_runner_cleanup_fails(monkeypatch):
    """A failed OpenAI cleanup must not leak the newly runner-owned Parallel transport."""
    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")

    async def fail_close():
        raise RuntimeError("model close failed")

    runner = GenerationAgentRunner(project_id=None, judge=None, openai_client=SimpleNamespace(close=fail_close))
    parallel = runner._reference_parallel()
    with pytest.raises(RuntimeError, match="model close failed"):
        await runner.aclose()
    assert parallel._client.is_closed
