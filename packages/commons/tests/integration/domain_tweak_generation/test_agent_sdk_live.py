from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAgentRunner,
    PublicSourceFetcher,
)
from harnyx_commons.domain_tweak_generation.source_workspace import SourceWorkspace
from harnyx_commons.llm.providers.vertex.credentials import cleanup_credentials_file, prepare_credentials

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
        runner = DomainTweakAgentRunner(project_id=project_id, region="global")
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
    assert candidate.title
