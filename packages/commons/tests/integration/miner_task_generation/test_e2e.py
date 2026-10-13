"""Deterministic composed lifecycle through public schemas and registered evidence rendering."""

import json
import runpy
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import httpx2
import pytest
from openai import AsyncOpenAI

from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_fast_scoring import FastJudgeAssessment
from harnyx_commons.miner_task_generation.agent_runner import CallCapture, GenerationAgentRunner
from harnyx_commons.miner_task_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.miner_task_generation.contracts import (
    AgentResult,
    CandidateSession,
    QuestionDraft,
    ReviewDecision,
    Seed,
    SeedSource,
    SourceSupport,
    StructuredQuestionDraft,
    TerminalReference,
)
from harnyx_commons.miner_task_generation.reference_evidence import EvidenceStore
from harnyx_commons.miner_task_generation.source_fetch import PublicSourceFetcher, SourceDocument, SourceFetchError
from harnyx_commons.tools.extraction_models import ExtractedPage, ExtractPagesResponse, PageExtractionError
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_models import SearchWebResult, SearchWebSearchResponse

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


def _check_reference_request_retention(monkeypatch):
    captures = []
    finish = CallCapture.finish

    def checked_finish(capture):
        if capture.role == "reference":
            captures.append(capture)
            assert capture.requests == []
        return finish(capture)

    monkeypatch.setattr(CallCapture, "finish", checked_finish)
    return captures


def _final_validation_feedback(comparison, number, payload, final):
    if comparison != "final_feedback":
        return
    if number == 3:
        final["citation_evidence_ids"] = ["evidence:404"]
    if number == 4:
        feedback = json.loads(payload["input"][-1]["content"])
        assert feedback["operation"] == "final_output" and feedback["retry"] == 1
        assert "Unregistered evidence ID" in feedback["error"]
        assert any(
            item.get("call_id") == "call_2" and item["type"] == "function_call_output" for item in payload["input"]
        )


@pytest.mark.parametrize(
    ("mode", "terminal_value", "proposal", "corrected", "comparison"),
    [
        ("plain_text", 1200, 1200, False, "agree"),
        ("plain_text", 1200, 1200, False, "compaction"),
        ("plain_text", 1200, 1200, False, "reread"),
        ("plain_text", 1200, 1200, False, "final_feedback"),
        ("plain_text", 1200, 1200, False, "paging"),
        ("plain_text", 1200, 1200, False, "paging_long_line"),
        ("plain_text", 1200, 1200, False, "replacement"),
        ("plain_text", 1200, 1200, False, "replacement_null"),
        ("plain_text", 1200, 1200, False, "replacement_blank"),
        ("plain_text", 1300, 1200, False, "disagree"),
        ("plain_text", 1200, 1200, False, "excess"),
        ("plain_text", 1200, 1200, False, "malformed"),
        ("plain_text", 1200, 1200, False, "provider"),
        ("plain_text", 1200, 1200, False, "deadline"),
        ("structured", 1200, 1200, False, None),
        ("structured", 1200, 1200, False, "prose_citation"),
        ("structured", 1300, 1200, True, "prose_citation"),
        ("structured", 1300, 1200, True, None),
        ("structured", 1, True, True, None),
        ("structured", 0, False, True, None),
    ],
)
async def test_full_candidate_lifecycle_renders_public_reference_from_registered_evidence(
    monkeypatch, mode, terminal_value, proposal, corrected, comparison
):
    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    captures = _check_reference_request_retention(monkeypatch)
    replacement = comparison in {"replacement", "replacement_null", "replacement_blank"}
    paging = comparison in {"paging", "paging_long_line"}
    native_reads = []

    async def fetch(self, url, **kwargs):
        native_reads.append(url)
        if replacement and url == "https://example.org/report":
            raise SourceFetchError("source_unavailable", "404")
        text = "Alpha value 1200"
        if paging:
            prefix = "Unrelated introduction " * 200_000
            text = prefix + ("\n" if comparison == "paging" else "") + text + "\n" + "End context " * 10
        return SourceDocument(url, url, "text/plain", text, len(text))

    monkeypatch.setattr(PublicSourceFetcher, "fetch", fetch)

    if replacement:
        monkeypatch.setattr(
            "harnyx_commons.miner_task_generation.reference_evidence._public_addresses", lambda *_: ("93.184.216.34",)
        )

        async def extract(self, request):
            return SearchProviderResult(
                ExtractPagesResponse(errors=(PageExtractionError(url=request.urls[0], error_type="unavailable"),))
                if comparison == "replacement"
                else ExtractPagesResponse(
                    pages=(
                        ExtractedPage(url=request.urls[0], content=None if comparison == "replacement_null" else " "),
                    )
                ),
                ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.001),
            )

        async def search(self, request):
            return SearchProviderResult(
                SearchWebSearchResponse(
                    data=[SearchWebResult(link="https://example.org/replacement", snippet="report")]
                ),
                ProviderBillingMetadata("parallel", "request_body", actual_cost_usd=0.005),
            )

        monkeypatch.setattr(ParallelClient, "extract_pages", extract)
        monkeypatch.setattr(ParallelClient, "search_web", search)
    schema = {
        "type": "object",
        "properties": {"value": {"type": ["integer", "boolean"]}},
        "required": ["value"],
        "additionalProperties": False,
        "allOf": [{"properties": {"value": {"type": ["integer", "boolean"]}}}],
    }
    if comparison == "prose_citation":
        schema["properties"]["description"] = {"type": "string"}
        schema["required"].append("description")

    def structured_value(value, *, cited=False):
        result = {"value": value}
        if comparison == "prose_citation":
            result["description"] = "Alpha has the named value" + (" [[evidence:1]]" if cited else "")
        return result

    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        n = len(requests)
        if comparison == "compaction" and n == 3:
            assert {"id": "cmp_2", "type": "compaction", "encrypted_content": "opaque-state"} in payload["input"]
            selection = next(
                item
                for item in payload["input"]
                if item.get("call_id") == "call_2" and item["type"] == "function_call_output"
            )
            assert "evidence:1" in selection["output"]
            assert payload["context_management"] == [{"type": "compaction", "compact_threshold": 200_000}]
        if n == 1:
            entry = next(item for item in payload["input"] if item.get("role") == "user")
            prompt = json.loads(entry["content"])
            assert "reachability" not in prompt and native_reads == []
            assert prompt["current_draft"]["question"] == "What value does Alpha have in the named report?"
            assert {"url": "https://example.org/report", "evidence": "Alpha value 1200"} in prompt["source_support"]
            assert "solver-private-trace" not in json.dumps(payload)
            assert prompt["verification"] == {"pass": True, "feedback": ""}
        tool_calls = [
            (
                "read_source",
                {"url": "https://example.org/replacement" if replacement else "https://example.org/report"},
            ),
            ("select_evidence", {"source_id": "source:1", "start_line": 1, "end_line": 1}),
        ]
        if replacement:
            tool_calls.insert(0, ("search_sources", {"query": "Alpha report replacement"}))
            tool_calls.insert(0, ("read_source", {"url": "https://example.org/report"}))
        if comparison == "reread":
            tool_calls.append(("read_source", {"url": "https://example.org/report"}))
            if n == 4:
                reread = next(
                    item
                    for item in payload["input"]
                    if item.get("call_id") == "call_3" and item["type"] == "function_call_output"
                )
                assert json.loads(reread["output"])["source_id"] == "source:1"
                assert native_reads == ["https://example.org/report"] * 2
        if paging:

            def tool_output(call_id):
                item = next(
                    item
                    for item in payload["input"]
                    if item.get("call_id") == call_id and item["type"] == "function_call_output"
                )
                return json.loads(item["output"])

            first = tool_output("call_1") if n >= 2 else None
            search = tool_output("call_3") if n >= 4 else None
            tool_calls = [
                ("read_source", {"url": "https://example.org/report"}),
                ("read_chunk", {"source_id": "source:1", "cursor": first["next_cursor"] if first else "pending"}),
                ("search_text", {"source_id": "source:1", "query": "Alpha value 1200", "cursor": None}),
                (
                    "read_chunk",
                    {"source_id": "source:1", "cursor": search["matches"][0]["cursor"] if search else "pending"},
                ),
                (
                    "select_evidence",
                    {
                        "source_id": "source:1",
                        "start_line": search["matches"][0]["start_line"] if search else 1,
                        "end_line": search["matches"][0]["end_line"] if search else 1,
                    },
                ),
            ]
            if n >= 2:
                assert first["total_characters"] > 4_000_000 and not first["text_complete"]
                assert len(first["text"]) < 16_000 and "Alpha value 1200" not in first["text"]
            if n >= 5:
                chunk = tool_output("call_4")
                assert chunk["start_line"] <= search["matches"][0]["start_line"] <= chunk["end_line"]
            assert len(request.content) < 100_000
        if n <= len(tool_calls):
            tool, args = tool_calls[n - 1]
            output = [
                {
                    "id": f"fc_{n}",
                    "type": "function_call",
                    "call_id": f"call_{n}",
                    "name": tool,
                    "arguments": json.dumps(args),
                    "status": "completed",
                }
            ]
        else:
            final = {
                "answer_text": f"{terminal_value} [[evidence:1]]" if mode == "plain_text" else None,
                "structured_answer_json": (
                    json.dumps(structured_value(terminal_value, cited=True)) if mode == "structured" else None
                ),
                "note": None,
                "citation_evidence_ids": ["evidence:1"],
            }
            _final_validation_feedback(comparison, n, payload, final)
            output = [
                {
                    "id": f"msg_{n}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(final), "annotations": []}],
                }
            ]
        if comparison == "compaction" and n == 2:
            output.insert(0, {"id": "cmp_2", "type": "compaction", "encrypted_content": "opaque-state"})
        response = {
            "id": f"resp_{n}",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "output": output,
        }
        event = {"type": "response.completed", "sequence_number": 0, "response": response}
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"event: response.completed\ndata: {json.dumps(event)}\n\n".encode(),
        )

    class Judge:
        async def invoke(self, request):
            assert comparison is not None  # Structured references need no semantic call.
            if comparison == "provider":
                raise RuntimeError("provider interrupted")
            if comparison == "deadline":
                import asyncio

                await asyncio.Event().wait()
            assessment = FastJudgeAssessment(
                expected_components=[{"component_id": "value", "is_correct": comparison != "disagree"}],
                excessive_components=[{"component_id": "extra"}] if comparison == "excess" else [],
            )
            return SimpleNamespace(
                raw_text="{}" if comparison == "malformed" else assessment.model_dump_json(),
                usage=LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

    model_http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    model_client = AsyncOpenAI(api_key="offline-test", http_client=model_http, max_retries=0)
    terminal_runner = GenerationAgentRunner(project_id=None, judge=Judge(), openai_client=model_client)

    class Runner:
        async def invoke(self, role, message, session, deadline):
            assert role != "analyst"
            if role == "author":
                payload = dict(
                    question="What value does Alpha have in the named report?",
                    answer="1200",
                    answer_components=["1200"],
                    scope="named report",
                    source_support=[SourceSupport(url="https://example.org/report", evidence="Alpha value 1200")],
                    revision_note="current",
                )
                draft = (
                    StructuredQuestionDraft(
                        **payload,
                        output_schema_json=json.dumps(schema),
                        structured_answer_json=json.dumps(structured_value(proposal)),
                    )
                    if mode == "structured"
                    else QuestionDraft(**payload)
                )
                return AgentResult(answer=draft.model_dump_json())
            if role in {"reviewer", "verifier"}:
                return AgentResult(answer=ReviewDecision(passed=True, feedback="").model_dump_json(by_alias=True))
            if role in {"solver", "grounded_solver"}:
                assert "source_support" not in message and "1200" not in message
                return AgentResult(
                    answer='{"value":900}' if mode == "structured" else "900",
                    provider_responses=[{"response": {"text": "solver-private-trace"}}],
                )
            return AgentResult(answer="Recorded solver route analysis")

        async def assess(self, *args):
            return FastJudgeAssessment(
                expected_components=[{"component_id": "value", "is_correct": False}], excessive_components=[]
            )

        async def derive_reference(self, draft, verification, session, deadline):
            return await terminal_runner.derive_reference(draft, verification, session, deadline)

    seed = Seed(
        seed_id="alpha",
        seed_question="Alpha value?",
        reference_answer="1200",
        scope="report",
        event_date="2026-10-01",
        supporting_sources=[
            SeedSource(url="https://example.org/report", evidence="Alpha value 1200", publication_date="2026-10-01")
        ],
    )
    try:
        result = await CandidatePipeline(runner=Runner()).run(
            seed,
            mode=mode,
            fast=True,
            session=CandidateSession(task_id="e2e", effective_date=datetime.now(UTC).date()),
            deadline=monotonic() + (0.1 if comparison == "deadline" else 5),
        )
    finally:
        if terminal_runner._reference_parallel_client is not None:
            await terminal_runner._reference_parallel_client.aclose()
        await model_client.close()
        await model_http.aclose()
    assert result.cycles[-1].analysis is None
    assert captures
    if comparison in {"malformed", "provider", "deadline"}:
        assert result.status == "operational_failure" and result.finalized is None
        return
    if comparison in {"disagree", "excess"}:
        assert result.status == "unresolved_reference" and result.finalized is None
        assert result.reference and "disagrees" in result.reference.unsupported_reason
        assert result.reference.reference_answer.text == f"{terminal_value} [[1]]"
        return
    if corrected:
        assert result.reference and result.reference.factual_correction
        assert result.status == "unresolved_reference" and result.finalized is None
        return
    assert result.status == "finalized" and result.finalized
    task = result.finalized.task
    assert task.query.output_schema == (schema if mode == "structured" else None)
    assert task.query.fast is True and task.reference_answer.citations
    if mode == "structured":
        expected = structured_value(terminal_value)
        if comparison == "prose_citation":
            expected["description"] += " [[1]]"
        assert json.loads(task.reference_answer.text) == expected
    else:
        assert task.reference_answer.text == "1200 [[1]]"
    assert "seed" not in task.model_dump_json() and "solver_results" not in task.model_dump_json()
    if replacement:
        assert task.reference_answer.citations[0].url == "https://example.org/replacement"
        assert len(requests) == 5
        assert native_reads == ["https://example.org/report", "https://example.org/replacement"]
    if paging:
        assert native_reads == ["https://example.org/report"] and len(requests) == 6
        assert "Alpha value 1200" in task.reference_answer.citations[0].excerpts[0].text
    if comparison == "final_feedback":
        assert native_reads == ["https://example.org/report"] and len(requests) == 4


@pytest.mark.parametrize("stale", [False, True])
async def test_manual_acceptance_capture_preserves_sources_and_receipts_on_writer_failure(monkeypatch, tmp_path, stale):
    """A failed paid acceptance must leave serializable source/selection and cost evidence for audit."""
    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    draft = QuestionDraft(
        question="What value?",
        answer="1200",
        answer_components=["1200"],
        scope="report",
        source_support=[SourceSupport(url="https://example.org/report", evidence="Alpha value 1200")],
        revision_note="original",
    )
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "mode": "plain_text",
                "effective_date": "2026-10-12",
                "draft": draft.model_dump(mode="json"),
                "author_history": [],
            }
        )
    )
    directory = tmp_path / "captured"
    monkeypatch.setenv("HARNYX_REFERENCE_HANDOFF", str(handoff))
    monkeypatch.setenv("HARNYX_REFERENCE_ACCEPTANCE_DIR", str(directory))

    async def client(self, **kwargs):
        return None

    async def failed_reference(draft, verification, session, deadline, **kwargs):
        assert kwargs["capture_full_requests"] is True
        capture = CallCapture(session, "reference", "gpt-6-luna", "openai", physical_accounting=True)
        store = EvidenceStore()
        source = store.retain(
            SourceDocument(
                "https://example.org/report", "https://example.org/report", "text/plain", "Alpha value 1200", 16
            ),
            "native_reachable",
        )
        store.select(source, 1, 1)
        if stale:
            store.observe("https://example.org/report", "source_unavailable")
        from harnyx_commons.miner_task_generation.agent_runner import ACTIVE_CAPTURE, record_openai_request

        token = ACTIVE_CAPTURE.set(capture)
        try:
            await record_openai_request(
                httpx2.Request("POST", "https://api.example.org/responses", json={"input": "quoted research"})
            )
        finally:
            ACTIVE_CAPTURE.reset(token)
        capture.event({"type": "response.created", "response": {"id": "receipt-before-failure"}})
        try:
            raise RuntimeError("writer interrupted")
        finally:
            capture.finish()

    monkeypatch.setattr(GenerationAgentRunner, "_openai", client)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.derive_reference", failed_reference)
    manual_path = Path(__file__).resolve().parents[6] / "scripts/tools/miner_task_reference_acceptance.py"
    manual = runpy.run_path(str(manual_path))
    with pytest.raises(RuntimeError, match="writer interrupted"):
        await manual["_capture_reference"](monkeypatch)
    receipt = json.loads((directory / "reference-acceptance.json").read_text())
    assert receipt["failure"]["type"] == "RuntimeError"
    assert receipt["captures"][0]["requests"] and receipt["captures"][0]["events"]
    assert receipt["sources"][0]["document"]["content"] == "Alpha value 1200"
    selected = receipt["selections"][0]["evidence:1"]
    original = receipt["sources"][0]["document"]["content"]
    assert selected["source_id"] == receipt["sources"][0]["source_id"]
    assert original[selected["excerpt_start"] : selected["excerpt_end"]] == "Alpha value 1200"
    assert receipt["stages"][0]["missing_usage_calls"] == 1
    assert receipt["stages"][0]["tool_usage"]["actual_total_cost_usd"] is None


@pytest.mark.parametrize("count,accepted", [(1, True), (2, False)])
async def test_manual_structured_acceptance_ignores_key_order_but_rejects_changed_values(
    monkeypatch, tmp_path, count, accepted
):
    """A valid structured reference must not require a paid replay solely because keys were reordered."""
    monkeypatch.setenv("PARALLEL_API_KEY", "offline-test")
    expected = '{"count":1,"enabled":true}'
    draft = StructuredQuestionDraft(
        question="What count and enabled state are reported?",
        answer=expected,
        answer_components=["1", "true"],
        scope="report",
        source_support=[SourceSupport(url="https://example.org/report", evidence="Count and enabled state")],
        revision_note="original",
        structured_answer_json=expected,
        output_schema_json=json.dumps(
            {
                "type": "object",
                "properties": {"count": {"type": "integer"}, "enabled": {"type": "boolean"}},
                "required": ["count", "enabled"],
                "additionalProperties": False,
            }
        ),
    )
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps(
            {
                "mode": "structured",
                "effective_date": "2026-10-13",
                "draft": draft.model_dump(mode="json"),
                "author_history": [],
                "expected_answer": expected,
            }
        )
    )
    directory = tmp_path / "captured"
    monkeypatch.setenv("HARNYX_REFERENCE_HANDOFF", str(handoff))
    monkeypatch.setenv("HARNYX_REFERENCE_ACCEPTANCE_DIR", str(directory))

    async def client(self, **kwargs):
        return None

    async def fixed_reference(*args, **kwargs):
        return TerminalReference.model_validate(
            {
                "reference_answer": {
                    "text": json.dumps({"enabled": True, "count": count}),
                    "citations": [
                        {
                            "url": "https://example.org/report",
                            "excerpts": [
                                {
                                    "start": 0,
                                    "end": len("Count and enabled state"),
                                    "text": "Count and enabled state",
                                }
                            ],
                        }
                    ],
                },
            }
        )

    monkeypatch.setattr(GenerationAgentRunner, "_openai", client)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.derive_reference", fixed_reference)
    manual_path = Path(__file__).resolve().parents[6] / "scripts/tools/miner_task_reference_acceptance.py"
    manual = runpy.run_path(str(manual_path))
    if accepted:
        await manual["_capture_reference"](monkeypatch)
    else:
        with pytest.raises(ValueError, match="expected structured value"):
            await manual["_capture_reference"](monkeypatch)
    receipt = json.loads((directory / "reference-acceptance.json").read_text())
    assert (receipt["failure"] is None) is accepted
    assert receipt["terminal"]["reference_answer"]["text"] == json.dumps({"enabled": True, "count": count})
