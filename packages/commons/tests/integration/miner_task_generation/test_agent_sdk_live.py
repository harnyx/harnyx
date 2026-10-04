from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from time import monotonic
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.providers.vertex.credentials import cleanup_credentials_file, prepare_credentials
from harnyx_commons.miner_task_generation import (
    PublicSourceFetcher,
)
from harnyx_commons.miner_task_generation.agent_runner import GenerationAgentRunner
from harnyx_commons.miner_task_generation.contracts import CandidateSession, ReviewDecision
from harnyx_commons.miner_task_generation.output_schema import format_assessment, strict_json
from harnyx_commons.miner_task_generation.reference_runner import ReferenceAgentRunner
from harnyx_commons.miner_task_generation.solver_runner import solve
from harnyx_commons.miner_task_generation.source_workspace import SourceWorkspace

pytestmark = [pytest.mark.integration, pytest.mark.expensive]


class _SearchCaptureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_candidate_id: str = Field(pattern=r"^source_candidate:\d+$")


@pytest.mark.anyio
async def test_agent_sdk_live_captures_native_web_search_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertex = VertexSettings()
    project_id = vertex.gcp_project_id
    credentials_b64 = vertex.gcp_sa_credential_b64_value
    assert project_id, "GCP_PROJECT_ID must be configured"
    assert credentials_b64, "Vertex credentials must be configured"
    _, credentials_file = prepare_credentials(None, credentials_b64)
    assert credentials_file
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", credentials_file)
    try:
        runner = ReferenceAgentRunner(project_id=project_id, region="global")
        workspace = SourceWorkspace()
        result = await runner.run_stage(
            stage="question_generation",
            system_prompt=(
                "Call WebSearch exactly once for the official Python documentation. "
                "After the host reports registered source candidates, return the first source_candidate_id."
            ),
            prompt="Find the official Python documentation and return its host-registered source candidate ID.",
            output_model=_SearchCaptureResult,
            timeout_seconds=180,
            web_search=True,
            tool_set=workspace.question_generation_tools(PublicSourceFetcher()),
        )
    finally:
        cleanup_credentials_file(credentials_file)

    assert isinstance(result.output, _SearchCaptureResult)
    candidate = workspace.get_source_candidate(result.output.source_candidate_id)
    assert candidate.url.startswith(("https://", "http://"))
    assert isinstance(candidate.title, str)


@pytest.mark.anyio
async def test_openai_reviewer_returns_complete_typed_verdict_and_exposed_evidence(caplog):
    """Bounded stage conformance, without a seed or improvement lifecycle."""
    assert os.environ.get("OPENAI_API_KEY"), "OpenAI credentials must be configured"
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")
    runner = GenerationAgentRunner(project_id=None, judge=AsyncMock(spec=LlmProviderPort))
    session = CandidateSession(task_id="live-reviewer", effective_date=datetime.now(UTC).date())
    packet = {
        "public_contract": {"question": "What language is documented at docs.python.org?"},
        "question": "What language is documented at docs.python.org?",
        "candidate_draft": {
            "question": "What language is documented at docs.python.org?",
            "answer": "Python",
            "answer_components": ["Python"],
            "scope": "official documentation",
            "source_support": [{"url": "https://docs.python.org/3/", "evidence": "Python documentation"}],
            "revision_note": "SDK contract probe",
        },
        "shared_context": {
            "seed_origin": {},
            "previous_question": None,
            "previous_draft": None,
            "solver_analysis": None,
        },
        "author_trace": [],
        "author_input_archive": "SDK contract probe",
    }
    try:
        result = await runner.invoke("reviewer", json.dumps(packet), session, monotonic() + 180)
        ReviewDecision.model_validate_json(result.answer)
        assert any(chunk["response"].get("type") == "response.completed" for chunk in result.provider_responses)
        assert result.request["message"] == json.dumps(packet)
        assert session.tool_usage.llm.call_count > 0 and session.attempted_calls > 0
        requests = [
            record.data["request"] for record in caplog.records if record.message == "task_generation.provider_request"
        ]
        assert requests
        assert requests[0]["model"] == "gpt-6-luna"
        assert requests[0]["reasoning"]["effort"] == "high"
        assert requests[0]["reasoning"]["summary"] == "detailed"
        assert any(tool["type"].startswith("web_search") for tool in requests[0]["tools"])
        assert requests[0]["text"]["format"]["type"] == "json_schema"
    finally:
        await runner.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("role", ["solver", "grounded_solver"])
async def test_vertex_solver_public_schema_packet_and_complete_response(role, monkeypatch, caplog):
    """Exercise each SDK boundary once; never run author improvements."""
    vertex = VertexSettings()
    caplog.set_level("DEBUG", logger="harnyx_commons.llm.calls")
    assert vertex.gcp_project_id and vertex.gcp_sa_credential_b64_value
    _, credentials_file = prepare_credentials(None, vertex.gcp_sa_credential_b64_value)
    assert credentials_file
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", credentials_file)
    schema = {
        "type": "object",
        "properties": {"language": {"type": "string"}},
        "required": ["language"],
        "additionalProperties": False,
    }
    packet = json.dumps(
        {"question": "What programming language is documented at docs.python.org?", "output_schema": schema}
    )
    session = CandidateSession(task_id=f"live-{role}", mode="structured", effective_date=datetime.now(UTC).date())
    try:
        result = await solve(role, packet, session, monotonic() + 180, project_id=vertex.gcp_project_id)
        assert result.request["message"] == packet and result.provider_responses
        assert session.tool_usage.llm.call_count > 0
        if role == "grounded_solver":
            assert result.request["instruction"] == ""
        requests = [
            record.data["request"]
            for record in caplog.records
            if record.message == "task_generation.provider_request"
            and record.data["role"] == role
            and "worker_order" not in record.data
        ]
        assert requests
        assert any(
            part.get("text") == packet for content in requests[0]["contents"] for part in content.get("parts", [])
        )
        assert not requests[0]["config"].get("response_schema")
        assert not requests[0]["config"].get("response_mime_type")
        if role == "grounded_solver":
            assert "system_instruction" not in requests[0]["config"]
        assert format_assessment(result.answer, schema)["valid"], result.answer
        assert isinstance(strict_json(result.answer)["language"], str)
    finally:
        cleanup_credentials_file(credentials_file)
