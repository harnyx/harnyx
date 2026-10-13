"""Exercise the pinned Agents SDK tool loop and receipts through offline HTTP."""

import asyncio
import json
import logging
from datetime import date

import httpx2
import pytest
from openai import APIStatusError, AsyncOpenAI

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.miner_task_generation.agent_runner import (
    ACTIVE_CAPTURE,
    CallCapture,
    OpenAIStreamError,
    record_openai_request,
)
from harnyx_commons.miner_task_generation.contracts import CandidateSession, CandidateStageError
from harnyx_commons.miner_task_generation.reference_evidence import EvidenceStore, ReferenceSources
from harnyx_commons.miner_task_generation.reference_runner import (
    ReferenceAgentRunner,
    ReferenceFeedback,
    own_reference_response,
)
from harnyx_commons.miner_task_generation.source_fetch import SourceDocument

pytestmark = pytest.mark.anyio("asyncio")


@pytest.mark.parametrize("role,retain", [("author", True), ("reference", False), ("reference", True)])
async def test_request_capture_retention_does_not_change_accounting_or_debug_diagnostics(role, retain, caplog):
    """Production reference history must not be copied on every physical turn; manual capture stays available."""
    kwargs = {"retain_requests": retain} if role == "reference" else {}
    capture = CallCapture(
        CandidateSession("retention", date.today()),
        role,
        "gpt-6-luna",
        "openai",
        physical_accounting=role == "reference",
        **kwargs,
    )
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    token = ACTIVE_CAPTURE.set(capture)
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={})),
            event_hooks={"request": [record_openai_request]},
        ) as http:
            for n in range(3):
                await http.post(
                    "https://api.example.org/responses",
                    json={"input": "research" * (n + 1), "encrypted_content": "private"},
                )
    finally:
        ACTIVE_CAPTURE.reset(token)
    assert len(capture.requests) == (3 if retain else 0)
    assert capture.attempted == (3 if role == "reference" else 0)
    logged = [r.data["request"] for r in caplog.records if r.message == "task_generation.provider_request"]
    assert len(logged) == 3
    assert all("encrypted_content" not in request for request in logged)
    if retain:
        assert capture.requests == logged


def test_validation_feedback_keeps_model_details_out_of_warning_logs(caplog):
    """Large model-controlled invalid JSON keys remain corrective feedback and DEBUG evidence only."""
    from harnyx_commons.miner_task_generation.output_schema import strict_json

    key = "model-controlled-key-" * 2_000
    with pytest.raises(ValueError) as invalid:
        strict_json(json.dumps({key: 1})[:-1] + f',"{key}":2}}')
    capture = CallCapture(CandidateSession("logging", date.today()), "reference", "gpt-6-luna", "openai")
    caplog.set_level(logging.DEBUG, logger="harnyx_commons.llm.calls")
    feedback = json.loads(ReferenceFeedback(capture).error("final_output", str(invalid.value)))
    assert key in feedback["error"] and feedback["retry"] == 1
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert key not in json.dumps(warnings[0].data) and "error" not in warnings[0].data
    assert warnings[0].data["operation"] == "final_output" and warnings[0].data["retry"] == 1
    assert any(key in json.dumps(r.data) for r in caplog.records if r.levelno == logging.DEBUG)


def stream(number, *, tool=None, args=None, calls=None, final=None, empty=False, status="completed", compact=False):
    output = (
        [
            {
                "id": f"fc_{number}",
                "type": "function_call",
                "call_id": f"call_{number}",
                "name": tool,
                "arguments": json.dumps(args),
                "status": "completed",
            }
        ]
        if tool
        else [
            {
                "id": f"msg_{number}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(final), "annotations": []}],
            }
        ]
    )
    if empty:
        output[0]["content"] = []
    if calls is not None:
        output = [
            {
                "id": f"fc_{number}_{index}",
                "type": "function_call",
                "call_id": f"call_{number}_{index}",
                "name": name,
                "arguments": json.dumps(arguments),
                "status": "completed",
            }
            for index, (name, arguments) in enumerate(calls)
        ]
    if compact:
        output.insert(0, {"id": f"cmp_{number}", "type": "compaction", "encrypted_content": "opaque-state"})
    response = {
        "id": f"resp_{number}",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": status,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "output": output,
    }
    event = {"type": f"response.{status}", "sequence_number": 0, "response": response}
    return f"event: response.{status}\ndata: {json.dumps(event)}\n\n".encode()


@pytest.mark.parametrize("failure", [None, "missing_receipt", "bad_final", "tool_auth", "incomplete", "failed"])
@pytest.mark.parametrize("compact", [False, True])
async def test_real_sdk_counts_physical_calls_and_preserves_failed_request_costs(failure, compact):
    requests = []
    final = {
        "answer_text": "Alpha [[evidence:1]]",
        "structured_answer_json": None,
        "note": None,
        "citation_evidence_ids": ["evidence:1"],
    }

    def respond(request):
        requests.append(json.loads(request.content))
        n = len(requests)
        if compact and n >= 3:
            items = requests[-1]["input"]
            assert {"id": "cmp_2", "type": "compaction", "encrypted_content": "opaque-state"} in items
            call = next(item for item in items if item.get("call_id") == "call_2" and item["type"] == "function_call")
            result = next(
                item for item in items if item.get("call_id") == "call_2" and item["type"] == "function_call_output"
            )
            assert call["name"] == "select_evidence"
            assert "evidence:1" in result["output"]
        if n == 4 and failure == "missing_receipt":
            return httpx2.Response(500, json={"error": {"message": "interrupted", "type": "server_error"}})
        tool, args = (
            ("read_source", {"url": "https://example.org/report"})
            if n == 1
            else ("select_evidence", {"source_id": "source:1", "start_line": 1, "end_line": 1})
        )
        if failure == "tool_auth":
            tool, args = "search_sources", {"query": "report"}
        payload = (
            stream(n, tool=tool, args=args, compact=compact and n == 2)
            if n < 4
            else stream(
                n,
                final=final if failure != "bad_final" else final | {"citation_evidence_ids": []},
                status=failure if failure in {"incomplete", "failed"} else "completed",
            )
        )
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            return SourceDocument(url, url, "text/plain", "Alpha value 88", 14)

    class Parallel:
        async def search_web(self, request):
            raise ToolProviderError("auth failed", http_status=401)

    session = CandidateSession("sdk", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    store = EvidenceStore()
    sources = ReferenceSources(Native(), Parallel(), store, capture)
    slot = asyncio.Semaphore(1)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        try:
            runner = ReferenceAgentRunner(client, slot)
            if failure:
                with pytest.raises(
                    {
                        "missing_receipt": APIStatusError,
                        "bad_final": CandidateStageError,
                        "tool_auth": ToolProviderError,
                        "incomplete": OpenAIStreamError,
                        "failed": OpenAIStreamError,
                    }[failure]
                ):
                    await runner.run("fixed work order", sources, capture)
            else:
                result = await runner.run("fixed work order", sources, capture)
                assert result.answer_text == final["answer_text"]
            capture.finish()
        finally:
            ACTIVE_CAPTURE.reset(token)
    assert not slot.locked()
    count = 1 if failure == "tool_auth" else (9 if failure == "bad_final" else 4)
    assert len(requests) == session.attempted_calls == count
    assert session.missing_usage_calls == (1 if failure == "missing_receipt" else 0)
    assert session.tool_usage.llm.call_count == (3 if failure == "missing_receipt" else count)
    assert (session.tool_usage.actual_total_cost_usd is None) == (failure in {"missing_receipt", "tool_auth"})
    receipt_count = 3 if failure == "missing_receipt" else count
    # Five ordinary inputs, two cache reads, three cache writes and five outputs.
    expected_model_cost = receipt_count * (5 * 0.10 + 2 * 0.01 + 3 * 0.125 + 5 * 0.50) / 1_000_000
    if failure not in {"missing_receipt", "tool_auth"}:
        assert session.tool_usage.llm.actual_cost == pytest.approx(expected_model_cost)
    assert capture.available_cost_usd == pytest.approx(expected_model_cost)
    for request in requests:
        assert "max_output_tokens" not in request
        assert request["reasoning"]["effort"] == "high"
        assert request["context_management"] == [{"type": "compaction", "compact_threshold": 200_000}]


@pytest.mark.parametrize("cancel_during_close", [False, True])
async def test_cancellation_joins_provider_close_before_releasing_model_slot(cancel_during_close):
    """SDK cancellation detaches HTTP close; reference retains ownership through that close."""
    reading, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    closed = False

    class GatedStream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            if cancel_during_close:
                yield stream(
                    1,
                    final={
                        "answer_text": "Alpha",
                        "structured_answer_json": None,
                        "note": None,
                        "citation_evidence_ids": ["evidence:1"],
                    },
                )
                return
            reading.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            nonlocal closed
            closing.set()
            await release.wait()
            closed = True

    session = CandidateSession("cancel-stream", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    sources = ReferenceSources(None, None, EvidenceStore(), capture)
    slot = asyncio.Semaphore(1)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=GatedStream())
        ),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        task = asyncio.create_task(ReferenceAgentRunner(client, slot).run("fixed order", sources, capture))
        try:
            await asyncio.wait_for(closing.wait() if cancel_during_close else reading.wait(), 1)
            task.cancel()
            await asyncio.wait_for(closing.wait(), 1)
            await asyncio.sleep(0.01)
            assert not task.done() and slot.locked() and not closed
            task.cancel()  # a second cancellation must not abandon the owned close
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert closed and not slot.locked()
            capture.finish()
            assert session.attempted_calls == 1
            assert session.missing_usage_calls == (0 if cancel_during_close else 1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            ACTIVE_CAPTURE.reset(token)


@pytest.mark.parametrize("invalid_search", [False, True])
@pytest.mark.parametrize("invalid_final", ["schema", "empty"])
async def test_feedback_recovers_tools_and_final_output_with_reset_retained_evidence_and_compaction(
    invalid_search, invalid_final
):
    """Correctable mistakes must not discard fetched sources or SDK compaction state."""
    from harnyx_commons.miner_task_generation.contracts import QuestionDraft
    from harnyx_commons.miner_task_generation.reference_evidence import render_reference

    requests, native_reads = [], []
    final = {
        "answer_text": "Alpha [[evidence:1]]",
        "structured_answer_json": None,
        "note": None,
        "citation_evidence_ids": ["evidence:1"],
    }
    actions = [
        ("read_source", {"url": "https://example.org/report"}),
        ("read_chunk", {"source_id": "source:1", "cursor": "source:1/read/bad/0"}),
        ("select_evidence", {"source_id": "source:404", "start_line": 1, "end_line": 1}),
        ("select_evidence", {"source_id": "source:1", "start_line": 0, "end_line": 1}),
        ("search_sources", {"query": "  "}) if invalid_search else ("read_chunk", {"source_id": "source:1"}),
        ("invented_source_tool", {}),
        ("select_evidence", {"source_id": "source:1", "start_line": 1, "end_line": 1}),
    ]

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        n = len(requests)
        if n >= 3 and n <= 7:
            previous = next(
                item
                for item in payload["input"]
                if item.get("call_id") == f"call_{n - 1}" and item["type"] == "function_call_output"
            )
            message = json.loads(previous["output"])
            assert message["type"] == "reference.model_feedback" and message["retry"] == n - 2
            assert message["error"] and message["instruction"]
        if n >= 9:
            assert {"id": "cmp_7", "type": "compaction", "encrypted_content": "opaque-state"} in payload["input"]
            assert any(
                item.get("call_id") == "call_7" and item["type"] == "function_call_output" for item in payload["input"]
            )
            feedback = json.loads(payload["input"][-1]["content"])
            assert feedback["operation"] == "final_output" and feedback["retry"] == n - 8
            if invalid_final != "empty" or n != 9:
                assert any(
                    item.get("type") == "message" and item.get("id") == f"msg_{n - 1}" for item in payload["input"]
                )
        if n <= len(actions):
            tool, args = actions[n - 1]
            body = stream(n, tool=tool, args=args, compact=n == 7)
        else:
            answer = final | {"citation_evidence_ids": [] if n == 8 else ["evidence:404"]} if n < 10 else final
            body = stream(n, final=answer, empty=n == 8 and invalid_final == "empty")
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            native_reads.append(url)
            return SourceDocument(url, url, "text/plain", "Alpha value 88", 14)

    session = CandidateSession("feedback", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    sources = ReferenceSources(Native(), None, EvidenceStore(), capture)
    draft = QuestionDraft(
        question="Which value?",
        answer="Alpha",
        answer_components=["Alpha"],
        scope="report",
        source_support=[{"url": "https://example.org/report", "evidence": "Alpha"}],
        revision_note="initial",
    )
    slot = asyncio.Semaphore(1)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        try:
            result = await ReferenceAgentRunner(
                client, slot, validate_output=lambda output: render_reference(output, sources.store, draft)
            ).run("fixed work order", sources, capture)
            capture.finish()
        finally:
            ACTIVE_CAPTURE.reset(token)
    assert result.answer_text == final["answer_text"]
    assert native_reads == ["https://example.org/report"]
    assert len(requests) == session.attempted_calls == session.tool_usage.llm.call_count == 10
    assert session.missing_usage_calls == 0 and session.tool_usage.actual_total_cost_usd is not None
    assert not slot.locked()
    feedback_events = [e["response"] for e in capture.responses if e["response"]["type"] == "reference.model_feedback"]
    assert [e["retry"] for e in feedback_events] == [1, 2, 3, 4, 5, 1, 2]
    assert not any(e["exhausted"] for e in feedback_events)
    resets = [e["response"] for e in capture.responses if e["response"]["type"] == "reference.model_feedback_reset"]
    assert [(e["operation"], e["previous_retries"]) for e in resets] == [("select_evidence", 5), ("final_output", 2)]


async def test_parallel_tool_feedback_resets_in_completion_order_without_losing_evidence():
    """Concurrent tool results share one correction budget and preserve successful selections."""
    requests = []
    final = {
        "answer_text": "Alpha [[evidence:1]]",
        "structured_answer_json": None,
        "note": None,
        "citation_evidence_ids": ["evidence:1"],
    }

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        n = len(requests)
        if n == 1:
            body = stream(n, tool="read_source", args={"url": "https://example.org/report"})
        elif n == 2:
            body = stream(
                n,
                calls=[("read_chunk", {"source_id": "source:1", "cursor": "invalid"})] * 5,
            )
        elif n == 3:
            errors = [
                json.loads(item["output"])
                for item in payload["input"]
                if item.get("type") == "function_call_output" and item.get("call_id", "").startswith("call_2_")
            ]
            assert [error["retry"] for error in errors] == [1, 2, 3, 4, 5]
            body = stream(
                n,
                calls=[
                    ("select_evidence", {"source_id": "source:1", "start_line": 1, "end_line": 1}),
                    ("search_text", {"source_id": "source:1", "query": "", "cursor": None}),
                ],
            )
        else:
            results = {
                item["call_id"]: json.loads(item["output"])
                for item in payload["input"]
                if item.get("type") == "function_call_output" and item.get("call_id", "").startswith("call_3_")
            }
            assert results["call_3_0"]["evidence_id"] == "evidence:1"
            assert results["call_3_1"]["retry"] == 1
            body = stream(n, final=final)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    class Native:
        async def fetch(self, url, *, timeout_seconds=None):
            return SourceDocument(url, url, "text/plain", "Alpha value 88", 14)

    session = CandidateSession("parallel-feedback", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    sources = ReferenceSources(Native(), None, EvidenceStore(), capture)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        try:
            result = await ReferenceAgentRunner(client, asyncio.Semaphore(1)).run("fixed work order", sources, capture)
            capture.finish()
        finally:
            ACTIVE_CAPTURE.reset(token)
    assert result.answer_text == final["answer_text"]
    assert sources.store.citation("evidence:1").excerpts[0].text == "Alpha value 88"
    assert len(requests) == session.attempted_calls == session.tool_usage.llm.call_count == 4
    assert session.missing_usage_calls == 0 and session.tool_usage.actual_total_cost_usd is not None
    resets = [e["response"] for e in capture.responses if e["response"]["type"] == "reference.model_feedback_reset"]
    assert [(e["operation"], e["previous_retries"]) for e in resets] == [("select_evidence", 5), ("final_output", 1)]


@pytest.mark.parametrize("operation", ["tool", "final", "empty_final"])
async def test_sixth_consecutive_model_error_exhausts_five_retries_and_retains_every_usage_receipt(operation):
    """The initial failure gets five correction chances, without an unbounded error loop."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        n = len(requests)
        assert n <= 6
        if operation == "tool":
            body = stream(n, tool="select_evidence", args={"source_id": "source:404", "start_line": 1, "end_line": 1})
        else:
            body = stream(
                n,
                final={
                    "answer_text": "Alpha",
                    "structured_answer_json": None,
                    "note": None,
                    "citation_evidence_ids": [],
                },
                empty=operation == "empty_final",
            )
        if n > 1:
            assert "reference.model_feedback" in json.dumps(requests[-1]["input"])
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    session = CandidateSession("exhausted", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    slot = asyncio.Semaphore(1)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        try:
            with pytest.raises(CandidateStageError, match="Five model correction retries exhausted"):
                await ReferenceAgentRunner(client, slot).run(
                    "fixed", ReferenceSources(None, None, EvidenceStore(), capture), capture
                )
            capture.finish()
        finally:
            ACTIVE_CAPTURE.reset(token)
    assert len(requests) == session.attempted_calls == session.tool_usage.llm.call_count == 6
    assert session.missing_usage_calls == 0 and session.tool_usage.actual_total_cost_usd is not None
    assert not slot.locked()
    feedback = [e["response"] for e in capture.responses if e["response"]["type"] == "reference.model_feedback"]
    assert [e["retry"] for e in feedback] == [1, 2, 3, 4, 5, 5]
    assert feedback[-1]["exhausted"] and not any(e["exhausted"] for e in feedback[:-1])


async def test_unexpected_collector_error_is_not_disguised_as_a_model_mistake():
    """Runtime bugs must remain visible instead of consuming model correction retries."""
    from agents.exceptions import UserError

    class BrokenNative:
        async def fetch(self, url, *, timeout_seconds=None):
            raise ValueError("collector implementation fault")

    session = CandidateSession("broken", date.today())
    capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=stream(1, tool="read_source", args={"url": "https://example.org/report"}),
            )
        ),
        event_hooks={"request": [record_openai_request], "response": [own_reference_response]},
    ) as http:
        client = AsyncOpenAI(api_key="offline-test", http_client=http, max_retries=0)
        token = ACTIVE_CAPTURE.set(capture)
        try:
            with pytest.raises(UserError) as error:
                await ReferenceAgentRunner(client, asyncio.Semaphore(1)).run(
                    "fixed", ReferenceSources(BrokenNative(), None, EvidenceStore(), capture), capture
                )
            assert isinstance(error.value.__cause__, ValueError)
            assert str(error.value.__cause__) == "collector implementation fault"
            capture.finish()
        finally:
            ACTIVE_CAPTURE.reset(token)
    assert session.attempted_calls == session.tool_usage.llm.call_count == 1
    assert not any(e["response"]["type"] == "reference.model_feedback" for e in capture.responses)
