"""Protect real reference-adapter lifetimes and terminal proposal agreement."""

import asyncio
import json
from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace

import pytest

from harnyx_commons.domain.tool_usage_accounting import tool_usage_from_llm_usage
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
from harnyx_commons.miner_task_generation.contracts import (
    CandidateSession,
    CandidateStageError,
    QuestionDraft,
    ReferenceProof,
    ReviewDecision,
    SourceSupport,
    StageRunResult,
)
from harnyx_commons.miner_task_generation.reference_generation import derive_reference
from harnyx_commons.miner_task_generation.source_fetch import PublicSourceFetcher
from harnyx_commons.miner_task_generation.source_workspace import SourceDocument, SourceWorkspace

pytestmark = pytest.mark.anyio("asyncio")


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
        task_id=task_id, effective_date=datetime.now(UTC).date(), source_support=draft().source_support
    )


def install_stage(monkeypatch, execute):
    """Mock provider I/O, preserving workspace construction, tools and host validation."""
    tools = {}
    original = SourceWorkspace.reference_tools

    def reference_tools(workspace, fetcher):
        result = original(workspace, fetcher)
        tools[id(result)] = (workspace, fetcher)
        return result

    async def stage(_runner, **kwargs):
        workspace, fetcher = tools[id(kwargs["tool_set"])]
        proof = await execute(workspace, fetcher, json.loads(kwargs["prompt"]))
        assert not kwargs["output_validator"](proof)
        return StageRunResult(proof, 1, usage())

    monkeypatch.setattr(SourceWorkspace, "reference_tools", reference_tools)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_runner.ReferenceAgentRunner.run_stage", stage)


def populate(workspace, *, answer="Paris", note=None, records=1):
    source = workspace.store(
        SourceDocument(
            requested_url="https://example.org/report",
            final_url="https://example.org/report",
            media_type="text/plain",
            content="Paris is named in the report",
            fetched_bytes=28,
        )
    )
    line = workspace.lines(source)[0]
    for index in range(records):
        evidence = workspace.register_evidence(
            claim=f"Report entry {index}", start_line_id=line.line_id, end_line_id=line.line_id
        )
    return ReferenceProof(
        status="finalized",
        answer_text=f"{answer} [[1]]",
        note=note,
        citation_evidence_ids=(evidence.evidence_id,),
        answers=({"answer_id": "A1"},),
        proof_steps=(
            {"step_id": "S1", "statement": "Paris", "kind": "supported", "evidence_ids": (evidence.evidence_id,)},
        ),
    )


def usage():
    return tool_usage_from_llm_usage(
        LlmUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        model="claude-opus-5",
        provider="vertex",
        pricing_date=datetime.now(UTC).date(),
    )


async def test_retry_discards_abandoned_evidence_and_certificates_but_keeps_usage(monkeypatch):
    """A legal first-attempt budget must not consume the fresh retry's capacity."""
    attempts = []

    async def execute(workspace, _fetcher, prompt):
        assert not workspace.evidence and not workspace.certificates
        assert prompt["source_candidates"][0]["source_candidate_id"]
        attempts.append(workspace)
        proof = populate(workspace, records=32)
        source = workspace.sources[0]
        line = workspace.lines(source)[0]
        for _ in range(16):
            workspace.register_regex_certificate(
                source_id=source.source_id, start_line_id=line.line_id, end_line_id=line.line_id, pattern="Paris"
            )
        if len(attempts) == 1:
            raise CandidateStageError(
                "transient_provider",
                "reference",
                "interrupted after tools",
                tool_usage=usage(),
            )
        return proof

    install_stage(monkeypatch, execute)
    monkeypatch.setattr("harnyx_commons.miner_task_generation.agent_runner.backoff_ms", lambda *_args: 0)
    observed = session()
    result = await derive_reference(
        draft(), None, observed, monotonic() + 5, project_id=None, fetcher=PublicSourceFetcher()
    )
    assert result.reference_answer and result.reference_answer.text == "Paris [[1]]"
    assert len(attempts) == 2 and attempts[0] is not attempts[1]
    assert observed.attempted_calls == 2
    assert observed.missing_usage_calls == 0
    assert observed.tool_usage.llm.call_count == 2
    assert observed.tool_usage.llm.prompt_tokens == 20


@pytest.mark.parametrize("judge_result", ["agree", "disagree", "excess", "malformed", "provider", "deadline"])
async def test_terminal_plain_reference_uses_pro_text_and_note_and_never_passes_failed_comparison(
    monkeypatch, judge_result
):
    """An unreported rewrite or failed comparison must not authorize publication."""
    requests = []

    async def execute(workspace, _fetcher, _prompt):
        return populate(workspace, answer="The city of Paris", note="No other city is an answer.")

    install_stage(monkeypatch, execute)

    class Judge:
        async def invoke(self, request):
            requests.append(request)
            if judge_result == "provider":
                raise RuntimeError("provider failed")
            if judge_result == "deadline":
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
    deadline = monotonic() + (0.05 if judge_result == "deadline" else 5)
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

    install_stage(monkeypatch, execute)
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
