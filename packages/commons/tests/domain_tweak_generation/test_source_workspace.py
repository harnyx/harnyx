import asyncio
import json
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest
from mcp import types as mcp_types

from harnyx_commons.domain_tweak_generation import ProofStep, SourceDocument, SourceFetchError, SourceWorkspace
from harnyx_commons.domain_tweak_generation import source_workspace as source_workspace_module
from harnyx_commons.domain_tweak_generation.source_workspace import SourceLink, _serialize_audit_packet


class _RecordingFetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def fetch(self, url: str, *, document_kind: str) -> SourceDocument:
        self.calls.append((url, document_kind))
        raise AssertionError("host validation must reject the request before fetch")


class _UnavailableFetcher:
    async def fetch(self, url: str, *, document_kind: str) -> SourceDocument:
        del url, document_kind
        raise SourceFetchError("source_unavailable", "upstream document is unavailable")


def test_web_search_registration_does_not_require_additive_search_count_metadata() -> None:
    """Future failure: the documented WebSearch envelope must work without optional SDK metadata."""
    workspace = SourceWorkspace()

    context = workspace.register_web_search_results(
        {
            "query": "annual report",
            "results": [
                {
                    "tool_use_id": "server-tool-1",
                    "content": [{"title": "Annual report", "url": "https://example.com/report"}],
                },
                "Search results for query `annual report`",
            ],
            "durationSeconds": 0.2,
        }
    )

    candidate = workspace.get_source_candidate("source_candidate:1")
    assert candidate.url == "https://example.com/report"
    assert "source_candidate:1" in context
    assert "Annual report" in context
    assert "https://example.com/report" not in context


def test_web_search_registration_rejects_undocumented_non_error_shape() -> None:
    workspace = SourceWorkspace()

    with pytest.raises(ValueError, match="pinned Agent SDK contract"):
        workspace.register_web_search_results({"results": [{"url": "https://example.com"}]})


@pytest.mark.anyio
async def test_fetch_tool_returns_a_stable_failure_id_for_exact_question_generation_attribution() -> None:
    """Future failure: QG must be able to select the exact fetch failure that blocks generation."""
    workspace = SourceWorkspace()
    workspace.register_web_search_results(
        {
            "query": "annual report",
            "results": [
                {
                    "tool_use_id": "server-tool-1",
                    "content": [{"title": "Annual report", "url": "https://example.com/report"}],
                }
            ],
            "durationSeconds": 0.2,
        }
    )
    server = workspace.question_generation_tools(_UnavailableFetcher()).mcp_servers[
        "question_generation_vfs"
    ]["instance"]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name="fetch_source",
            arguments={
                "source_candidate_id": "source_candidate:1",
                "document_kind": "html",
                "public_document_reason": "ordinary public annual report",
            },
        ),
    )

    response = await server.request_handlers[mcp_types.CallToolRequest](request)

    assert response.root.isError is True
    content = response.root.content
    assert len(content) == 1
    assert isinstance(content[0], mcp_types.TextContent)
    payload = json.loads(content[0].text)
    assert payload == {
        "error": "fetch_source_failed",
        "source_failure_id": "source_failure:1",
        "failure_class": "source_unavailable",
        "message": "upstream document is unavailable",
    }
    assert workspace.source_failure("source_failure:1").failure_class == "source_unavailable"


def test_unknown_source_candidate_is_rejected_before_fetch() -> None:
    workspace = SourceWorkspace()

    with pytest.raises(ValueError, match="unknown source_candidate_id"):
        workspace.get_source_candidate("source_candidate:404")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("source_candidate_id", "document_kind", "public_document_reason"),
    (
        ("source_candidate:404", "html", "ordinary public report"),
        ("source_candidate:1", "api", "ordinary public report"),
        ("source_candidate:1", "html", ""),
    ),
)
async def test_fetch_contract_errors_do_not_enter_source_failure_ledger(
    source_candidate_id: str,
    document_kind: str,
    public_document_reason: str,
) -> None:
    """Future failure: host input defects must not authorize a model-declared source failure."""
    workspace = SourceWorkspace()
    workspace.register_web_search_results(
        {
            "query": "annual report",
            "results": [
                {
                    "tool_use_id": "server-tool-1",
                    "content": [{"title": "Annual report", "url": "https://example.com/report"}],
                }
            ],
            "durationSeconds": 0.2,
            "searchCount": 1,
        }
    )
    fetcher = _RecordingFetcher()

    with pytest.raises(ValueError):
        await workspace._fetch_source(
            fetcher,  # type: ignore[arg-type]
            source_candidate_id=source_candidate_id,
            document_kind=document_kind,
            public_document_reason=public_document_reason,
        )

    assert fetcher.calls == []
    assert workspace.source_failure("source_failure:1") is None


def test_workspace_uses_stable_ids_and_attaches_table_header() -> None:
    """Future failure: models select IDs; they must never recopy evidence or lose row labels."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com",
            final_url="https://example.com",
            media_type="text/html",
            content="HEADER\tName\tValue\nROW\tA\t7\nROW\tB\t9",
            fetched_bytes=10,
        )
    )
    lines = workspace.lines(source)
    evidence = workspace.register_evidence(
        claim="A has value 7",
        start_line_id=lines[1].line_id,
        end_line_id=lines[1].line_id,
    )
    certificate = workspace.register_regex_certificate(
        source_id=source.source_id,
        start_line_id=lines[0].line_id,
        end_line_id=lines[-1].line_id,
        pattern=r"^ROW",
    )

    assert evidence.evidence_id == "E1"
    assert evidence.excerpt.startswith("HEADER\tName\tValue")
    assert certificate.match_line_ids == (lines[1].line_id, lines[2].line_id)
    assert "content" not in workspace.source_metadata()[0]


@pytest.mark.anyio
async def test_list_source_links_filters_bounded_results_and_registers_fetch_candidates() -> None:
    """Future failure: retained HTML navigation must stay bounded without becoming a crawler."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/index",
            final_url="https://example.com/index",
            media_type="text/html",
            content="Index",
            fetched_bytes=5,
            links=tuple(
                SourceLink(url=f"https://example.com/annual/{index}", text=f"Annual record {index}")
                for index in range(105)
            )
            + (SourceLink(url="https://example.com/monthly", text="Monthly record"),),
        )
    )
    server = workspace.question_generation_tools(_UnavailableFetcher()).mcp_servers[
        "question_generation_vfs"
    ]["instance"]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name="list_source_links",
            arguments={"source_id": source.source_id, "pattern": "ANNUAL"},
        ),
    )

    response = await server.request_handlers[mcp_types.CallToolRequest](request)

    content = response.root.content
    assert response.root.isError is not True
    assert len(content) == 1 and isinstance(content[0], mcp_types.TextContent)
    payload = json.loads(content[0].text)
    assert payload["returned_match_count"] == 100
    assert payload["total_match_count"] == 105
    assert payload["truncated"] is True
    assert len(payload["links"]) == 100
    assert workspace.get_source_candidate("source_candidate:100").url.endswith("/annual/99")


@pytest.mark.anyio
async def test_list_source_links_bounds_the_serialized_result_not_only_the_link_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: many individually valid links must not create an unbounded agent-tool response."""
    monkeypatch.setattr(source_workspace_module, "_MAX_LINK_RESULT_CHARACTERS", 600)
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/index",
            final_url="https://example.com/index",
            media_type="text/html",
            content="Index",
            fetched_bytes=5,
            links=tuple(
                SourceLink(
                    url=f"https://example.com/annual/{index}/" + ("u" * 80),
                    text=f"Annual record {index} " + ("t" * 80),
                )
                for index in range(20)
            ),
        )
    )
    server = workspace.question_generation_tools(_UnavailableFetcher()).mcp_servers[
        "question_generation_vfs"
    ]["instance"]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name="list_source_links",
            arguments={"source_id": source.source_id, "pattern": "annual"},
        ),
    )

    response = await server.request_handlers[mcp_types.CallToolRequest](request)

    content = response.root.content
    assert len(content) == 1 and isinstance(content[0], mcp_types.TextContent)
    payload = json.loads(content[0].text)
    assert 0 < payload["returned_match_count"] < payload["total_match_count"]
    assert payload["truncated"] is True
    assert len(content[0].text) <= 600
    for item in payload["links"]:
        assert workspace.get_source_candidate(item["source_candidate_id"]).url == item["url"]


def test_audit_tools_are_exactly_the_fixed_read_only_source_inspection_boundary() -> None:
    """Future failure: the blind audit must not gain fetch, search, registration, or mutation tools."""
    assert SourceWorkspace().audit_tools().allowed_tools == (
        "mcp__audit_vfs__list_sources",
        "mcp__audit_vfs__regex_search",
        "mcp__audit_vfs__read_lines",
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_sources", {}),
        ("regex_search", {"pattern": "Alpha", "context_lines": 1, "max_matches": 10}),
        ("read_lines", {}),
    ],
)
async def test_shared_inspection_tools_return_the_same_payload_for_author_and_audit_paths(
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    """Future failure: author and audit wrappers must not drift into different source semantics."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="HEADER\tName\tValue\nROW\tAlpha\t7\nROW\tBeta\t9",
            fetched_bytes=45,
        )
    )
    lines = workspace.lines(source)
    resolved_arguments = dict(arguments)
    if tool_name == "regex_search":
        resolved_arguments["source_id"] = source.source_id
    elif tool_name == "read_lines":
        resolved_arguments = {
            "start_line_id": lines[0].line_id,
            "end_line_id": lines[-1].line_id,
        }
    author_server = workspace.question_generation_tools(_UnavailableFetcher()).mcp_servers[
        "question_generation_vfs"
    ]["instance"]
    audit_server = workspace.audit_tools().mcp_servers["audit_vfs"]["instance"]

    async def call(server: object) -> str:
        request = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(name=tool_name, arguments=resolved_arguments),
        )
        response = await server.request_handlers[mcp_types.CallToolRequest](request)  # type: ignore[attr-defined]
        content = response.root.content
        assert len(content) == 1 and isinstance(content[0], mcp_types.TextContent)
        return content[0].text

    assert await call(author_server) == await call(audit_server)


def test_citation_offsets_preserve_raw_crlf_and_non_ascii_source_text() -> None:
    """Future failure: line views must map back to exact miner-visible raw character offsets."""
    content = "preface-" + ("x" * 90) + "\r\nROW\tCafé\t1,200\r\ntail-雪"
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content=content,
            fetched_bytes=len(content.encode()),
        )
    )
    lines = workspace.lines(source)
    evidence = workspace.register_evidence(
        claim="Café value",
        start_line_id=lines[1].line_id,
        end_line_id=lines[1].line_id,
    )

    selected = workspace.citation_slices(evidence)[0]

    assert selected.end - selected.start >= 100
    assert content[selected.start : selected.end] == source.content[selected.start : selected.end]
    assert "ROW\tCafé\t1,200\r\n" in content[selected.start : selected.end]


def test_closed_audit_packet_includes_selected_evidence_and_certificate_boundaries() -> None:
    """Future failure: the independent audit must see bounded context without receiving the source body."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="intro\nHEADER\tName\tValue\nROW\tAlpha\t1200\nROW\tBeta\t900\nfooter",
            fetched_bytes=80,
        )
    )
    lines = workspace.lines(source)
    evidence = workspace.register_evidence(
        claim="Alpha value",
        start_line_id=lines[2].line_id,
        end_line_id=lines[2].line_id,
    )
    certificate = workspace.register_regex_certificate(
        source_id=source.source_id,
        start_line_id=lines[1].line_id,
        end_line_id=lines[3].line_id,
        pattern=r"ROW\t",
    )

    packet = workspace.proof_packet(
        question="Which row is larger?",
        short_answers=("Alpha",),
        answer_text="Alpha is larger [[1]].",
        validated_citations=(),
        steps=(
            ProofStep(
                step_id="S1",
                statement="Alpha has value 1200.",
                kind="supported",
                evidence_ids=(evidence.evidence_id,),
                scan_certificate_ids=(certificate.certificate_id,),
            ),
        ),
    )

    selected = packet["selected_evidence"][0]  # type: ignore[index]
    scan = packet["scan_certificates"][0]  # type: ignore[index]
    assert selected["excerpt"] == "HEADER\tName\tValue\nROW\tAlpha\t1200"
    assert [item["text"] for item in selected["boundary_context"]] == ["intro", "ROW\tBeta\t900", "footer"]
    assert [item["text"] for item in scan["matched_lines"]] == ["ROW\tAlpha\t1200", "ROW\tBeta\t900"]
    assert "content" not in str(packet)


def test_evidence_registration_rejects_a_range_larger_than_the_read_contract() -> None:
    """Future failure: evidence registration must not become a back door for returning a source body."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="\n".join(f"row {index}" for index in range(129)),
            fetched_bytes=1,
        )
    )
    lines = workspace.lines(source)

    with pytest.raises(ValueError, match="evidence range"):
        workspace.register_evidence(
            claim="all rows",
            start_line_id=lines[0].line_id,
            end_line_id=lines[-1].line_id,
        )


def test_certificate_rejects_an_audit_packet_with_too_many_matches() -> None:
    """Future failure: a broad certificate must fail closed instead of expanding unbounded match text."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="\n".join(f"ROW {index}" for index in range(101)),
            fetched_bytes=1,
        )
    )
    lines = workspace.lines(source)

    with pytest.raises(ValueError, match="certificate match limit"):
        workspace.register_regex_certificate(
            source_id=source.source_id,
            start_line_id=lines[0].line_id,
            end_line_id=lines[-1].line_id,
            pattern=r"^ROW",
        )


def test_evidence_rejects_one_pathologically_long_line() -> None:
    """Future failure: a single long line must not bypass the bounded audit-text contract."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="x" * 32_001,
            fetched_bytes=1,
        )
    )
    line = workspace.lines(source)[0]

    with pytest.raises(ValueError, match="evidence audit text"):
        workspace.register_evidence(
            claim="long value",
            start_line_id=line.line_id,
            end_line_id=line.line_id,
        )


def test_proof_packet_trims_aggregate_evidence_inside_the_context_bound() -> None:
    """Future failure: aggregate excerpts must remain valid, bounded, and identity-complete."""
    workspace = SourceWorkspace()
    blocks = [(str(index) * 30_000) + ("\n" * 5) for index in range(5)]
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="".join(blocks),
            fetched_bytes=1,
        )
    )
    lines = workspace.lines(source)
    records = [
        workspace.register_evidence(
            claim=f"value {index}",
            start_line_id=lines[index * 5].line_id,
            end_line_id=lines[index * 5].line_id,
        )
        for index in range(5)
    ]

    packet = workspace.proof_packet(
        question="Combine the values.",
        short_answers=("answer",),
        answer_text="Combined answer.",
        validated_citations=(),
        steps=(
            ProofStep(
                step_id="S1",
                statement="The values are combined.",
                kind="supported",
                evidence_ids=tuple(record.evidence_id for record in records),
            ),
        ),
    )

    serialized = _serialize_audit_packet(packet)
    assert len(serialized) <= 128_000
    assert json.loads(serialized) == packet
    selected = packet["selected_evidence"]
    assert isinstance(selected, list)
    assert [item["evidence_id"] for item in selected] == [record.evidence_id for record in records]
    assert all("audit text truncated" in item["excerpt"] for item in selected)
    assert all(item["excerpt"].startswith(str(index)) for index, item in enumerate(selected))
    assert all(item["excerpt"].endswith(str(index)) for index, item in enumerate(selected))
    assert all(
        line["line_id"]
        for item in selected
        for line in item["boundary_context"]
    )


def test_structured_proof_packet_preserves_required_contract_while_trimming_evidence() -> None:
    """Future failure: packet budgeting must never truncate schema or structured reference semantics."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content=("a" * 30_000) + ("\n" * 5) + ("b" * 30_000) + ("\n" * 5) + ("c" * 30_000),
            fetched_bytes=90_010,
        )
    )
    lines = workspace.lines(source)
    evidence = tuple(
        workspace.register_evidence(
            claim=f"value {index}",
            start_line_id=lines[index * 5].line_id,
            end_line_id=lines[index * 5].line_id,
        )
        for index in range(3)
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    structured_answer = {"value": "z" * 50_000}

    packet = workspace.proof_packet(
        question="Return field value.",
        short_answers=("answer",),
        answer_text=None,
        validated_citations=(),
        steps=(
            ProofStep(
                step_id="S1",
                statement="Combine the records.",
                kind="supported",
                evidence_ids=tuple(item.evidence_id for item in evidence),
            ),
        ),
        response_mode="structured",
        output_schema=schema,
        structured_answer=structured_answer,
    )

    assert len(_serialize_audit_packet(packet)) <= 128_000
    assert packet["output_schema"] == schema
    assert packet["structured_answer"] == structured_answer
    assert "audit text truncated" in str(packet["selected_evidence"])


def test_structured_proof_packet_preserves_irreducible_public_envelope_over_target() -> None:
    """Future failure: the ordinary packet target must not silently narrow exact public semantics."""
    workspace = SourceWorkspace()

    packet = workspace.proof_packet(
        question="q" * 60_000,
        short_answers=("answer",),
        answer_text=None,
        validated_citations=(),
        steps=(),
        response_mode="structured",
        output_schema={"type": "object"},
        structured_answer={"value": "z" * 70_000},
    )

    assert len(_serialize_audit_packet(packet)) > 128_000
    assert packet["question"] == "q" * 60_000
    assert packet["structured_answer"] == {"value": "z" * 70_000}


def test_proof_packet_minimum_uses_the_smaller_serialized_text_value() -> None:
    """Future failure: escaping must not make a retained original larger than the truncation marker."""
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="\\" * 20,
            fetched_bytes=20,
        )
    )
    line = workspace.lines(source)[0]
    evidence = workspace.register_evidence(
        claim="escaped value",
        start_line_id=line.line_id,
        end_line_id=line.line_id,
    )
    steps = (
        ProofStep(
            step_id="S1",
            statement="The escaped value is selected.",
            kind="supported",
            evidence_ids=(evidence.evidence_id,),
        ),
    )
    ordinary = workspace.proof_packet(
        question="",
        short_answers=("answer",),
        answer_text="answer",
        validated_citations=(),
        steps=steps,
    )
    question = "Q" * (128_001 - len(_serialize_audit_packet(ordinary)))

    packet = workspace.proof_packet(
        question=question,
        short_answers=("answer",),
        answer_text="answer",
        validated_citations=(),
        steps=steps,
    )

    assert len(_serialize_audit_packet(packet)) <= 128_000
    selected = packet["selected_evidence"]
    assert isinstance(selected, list)
    assert selected[0]["excerpt"] == "[... audit text truncated ...]"


def test_similarity_search_keeps_near_limit_source_working_memory_bounded() -> None:
    """Future failure: a legal source must not build a full-token corpus that can OOM concurrent candidates."""
    workspace = SourceWorkspace()
    content = (("needle filler value\n" * 250_000) + "needle target")[0:4_999_999]
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content=content,
            fetched_bytes=len(content),
        )
    )

    tracemalloc.start()
    try:
        result = workspace._similarity_result(source, "needle target", 3)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result["chunks"]
    assert peak_bytes < 32 * 1024 * 1024


@pytest.mark.anyio
async def test_similarity_search_limits_workers_and_cancels_queued_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: cancelled refill must not leave every queued source scan running in background."""
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(source_workspace_module, "_SIMILARITY_EXECUTOR", executor)
    workspace = SourceWorkspace()
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.com/report",
            final_url="https://example.com/report",
            media_type="text/plain",
            content="needle",
            fetched_bytes=6,
        )
    )
    lock = threading.Lock()
    two_started = threading.Event()
    release = threading.Event()
    invocation_count = 0
    active_count = 0
    peak_active_count = 0

    def blocking_result(
        _source: object,
        _query: str,
        _top_k: int,
    ) -> dict[str, object]:
        nonlocal invocation_count, active_count, peak_active_count
        with lock:
            invocation_count += 1
            active_count += 1
            peak_active_count = max(peak_active_count, active_count)
            if active_count == 2:
                two_started.set()
        release.wait()
        with lock:
            active_count -= 1
        return {"chunks": []}

    monkeypatch.setattr(workspace, "_similarity_result", blocking_result)
    tasks = [asyncio.create_task(workspace._run_similarity_result(source, "needle", 1)) for _ in range(10)]
    try:
        assert await asyncio.to_thread(two_started.wait, 2.0)
        for task in tasks[2:]:
            task.cancel()
        await asyncio.gather(*tasks[2:], return_exceptions=True)
        release.set()
        await asyncio.gather(*tasks[:2])
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        executor.shutdown(wait=True, cancel_futures=True)

    assert invocation_count == 2
    assert peak_active_count == 2
