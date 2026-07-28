from __future__ import annotations

import asyncio

import pytest

from harnyx_commons.clients import PLATFORM
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio("asyncio"),
    pytest.mark.flaky(reruns=1),
]

VERTEX_MAAS_LIVE_MODEL = "openai/gpt-oss-120b"


async def test_vertex_openai_maas_completion_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = LlmProviderAdapter(
        provider_name="vertex",
        delegate=VertexLlmProvider(
            project=project,
            location=location,
            timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
            credentials_path=None,
            service_account_b64=credentials_b64 or "",
        ),
    )
    try:
        request = LlmRequest(
            provider="vertex",
            model=VERTEX_MAAS_LIVE_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            'What is 7 times 8? Reply with only "56".'
                        ),
                    ),
                ),
            ),
            temperature=0.0,
            max_output_tokens=64,
        )

        attempts = 8
        response = None
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await provider.invoke(request)
                break
            except LlmRetryExhaustedError as exc:
                last_error = exc
                if str(exc) != "empty_output" or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(1)
    finally:
        await provider.aclose()

    if response is None and last_error is not None:
        raise last_error
    assert response is not None
    assert response.raw_text, "Vertex MaaS OpenAI response should include text output"
    assert "56" in response.raw_text
    assert response.usage.reasoning_tokens is None
