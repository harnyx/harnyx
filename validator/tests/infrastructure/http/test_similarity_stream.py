from __future__ import annotations

import asyncio
import json

import pytest
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

from harnyx_commons.llm.judge_observability import emit_judge_event
from harnyx_validator.infrastructure.http import similarity_stream as stream_module
from harnyx_validator.infrastructure.http.similarity_stream import SimilarityStreamingResponse


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [200, 502])
@pytest.mark.parametrize("evidence_size", [20, 2000])
async def test_private_evidence_cannot_prevent_terminal_delivery(monkeypatch, status_code, evidence_size) -> None:
    monkeypatch.setattr(stream_module, "SIMILARITY_TERMINAL_MAX_BYTES", 1000)

    async def execute():
        return JSONResponse(
            {"reasoning": "complete judgment", "provider_evidence": [{"payload": "x" * evidence_size}]},
            status_code=status_code,
        )

    response = SimilarityStreamingResponse(execute)
    await response._produce()
    frames = [part async for part in response._body()]
    terminal = json.loads(frames[-1].split(b"data: ", 1)[1])
    payload = terminal["result" if status_code == 200 else "error"]
    assert payload["reasoning"] == "complete judgment"
    assert ("provider_evidence" in payload) == (evidence_size == 20)


@pytest.mark.anyio
async def test_history_limits_are_explicit_and_do_not_truncate_judgment(monkeypatch) -> None:
    monkeypatch.setattr(stream_module, "SIMILARITY_DIAGNOSTICS_MAX_BYTES", 2500)

    async def execute():
        for index in range(70):
            emit_judge_event("finished", {
                "attempt_id": str(index), "model": "m" * 500, "outcome": "failed",
                "timeout_phase": "total", "elapsed_ms": 900000.0,
            })
        return JSONResponse({"reasoning": "complete judgment"})

    response = SimilarityStreamingResponse(execute)
    await response._produce()
    frames = [part async for part in response._body()]
    result = json.loads(frames[-1].split(b"data: ", 1)[1])
    assert result["result"]["reasoning"] == "complete judgment"
    diagnostics = result["diagnostics"]
    assert diagnostics["omitted_attempts"] + len(diagnostics["attempts"]) == 70
    assert diagnostics["omitted_attempts"] > 6
    assert diagnostics["attempts"][-1]["attempt_id"] == "69"
    assert "model" in diagnostics["attempts"][-1]["truncated_fields"]


@pytest.mark.anyio
async def test_blocked_asgi_send_cancels_and_joins_judge(monkeypatch) -> None:
    monkeypatch.setattr(stream_module, "_SEND_TIMEOUT_SECONDS", 0.02)
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def execute():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()
        return JSONResponse({})

    async def send(message):
        if message["type"] == "http.response.body":
            await started.wait()
            await asyncio.Event().wait()

    async def receive():
        await asyncio.Event().wait()

    response = SimilarityStreamingResponse(execute)
    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert cleaned_up.is_set()


@pytest.mark.anyio
async def test_concurrent_streams_keep_diagnostic_context_separate() -> None:
    async def execute(name):
        emit_judge_event("started", {"attempt_id": name, "model": name})
        await asyncio.sleep(0)
        emit_judge_event("finished", {"attempt_id": name, "outcome": "response_received"})
        return JSONResponse({"name": name})

    left = SimilarityStreamingResponse(lambda: execute("left"))
    right = SimilarityStreamingResponse(lambda: execute("right"))
    await asyncio.gather(left._produce(), right._produce())
    assert [attempt.model for attempt in left._diagnostics().attempts] == ["left"]
    assert [attempt.model for attempt in right._diagnostics().attempts] == ["right"]


@pytest.mark.anyio
async def test_blocked_terminal_send_has_bounded_lifetime(monkeypatch) -> None:
    monkeypatch.setattr(stream_module, "_SEND_TIMEOUT_SECONDS", 0.02)
    terminal_send = asyncio.Event()

    async def execute():
        return JSONResponse({"reasoning": "complete"})

    async def send(message):
        if b"event: result" in message.get("body", b""):
            terminal_send.set()
            await asyncio.Event().wait()

    async def receive():
        await asyncio.Event().wait()

    response = SimilarityStreamingResponse(execute)
    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(
            response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send), 1
        )
    assert terminal_send.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize("text", ["😀" * 256, "\x00" * 256])
async def test_encoded_diagnostic_size_cannot_interrupt_judgment(text) -> None:
    response = SimilarityStreamingResponse(lambda: asyncio.sleep(0, result=JSONResponse({"verdict": "novel"})))
    response._observe("started", dict.fromkeys([
        "model", "provider", "upstream_provider", "candidate_model",
        "attempt_id", "similarity_invocation_id", "outcome", "exception_type",
    ], text))
    body = response._body()
    status = await anext(body)
    assert len(status) < stream_module.SIMILARITY_STATUS_MAX_BYTES
    assert json.loads(status.split(b"data: ", 1)[1])["status"]["truncated_fields"]
    await response._produce()
    terminal = await anext(body)
    assert json.loads(terminal.split(b"data: ", 1)[1])["result"] == {"verdict": "novel"}
    await body.aclose()
