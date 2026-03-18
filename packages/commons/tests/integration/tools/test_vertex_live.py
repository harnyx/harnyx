from __future__ import annotations

import pytest
from pydantic import BaseModel

from harnyx_commons.clients import PLATFORM
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider
from harnyx_commons.llm.schema import GroundedLlmRequest, LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


async def test_vertex_maas_completion_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.vertex_maas_gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "VERTEX_MAAS_GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    try:
        request = LlmRequest(
            provider="vertex",
            model="publishers/openai/models/gpt-oss-20b-maas",
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            "Respond with a short sentence describing the Harnyx validator runtime."
                        ),
                    ),
                ),
            ),
            temperature=0.2,
            max_output_tokens=256,
        )

        response = await provider.invoke(request)
        assert response.raw_text, "Vertex response should include text output"
    finally:
        await provider.aclose()


async def test_vertex_multimodal_image_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    try:
        request = LlmRequest(
            provider="vertex",
            model="gemini-2.5-flash-lite",
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text("What is in this image? Reply in one sentence."),
                        LlmMessageContentPart.input_image_url(
                            "gs://generativeai-downloads/images/scones.jpg",
                            mime_type="image/jpeg",
                        ),
                    ),
                ),
            ),
            temperature=0.2,
            max_output_tokens=256,
        )

        response = await provider.invoke(request)
        assert response.raw_text, "Vertex multimodal response should include text output"
    finally:
        await provider.aclose()


async def test_vertex_reasoning_effort_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    try:
        request = LlmRequest(
            provider="vertex",
            model="gemini-3-flash-preview",
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            "Give one concise sentence about why explicit retry telemetry matters."
                        ),
                    ),
                ),
            ),
            temperature=0.2,
            max_output_tokens=256,
            reasoning_effort="low",
        )
        response = await provider.invoke(request)
        assert response.raw_text, "Vertex reasoning response should include text output"
    finally:
        await provider.aclose()


async def test_vertex_grounded_search_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    try:
        request = GroundedLlmRequest(
            provider="vertex",
            model="gemini-2.0-flash",
            messages=(
                LlmMessage(
                    role="system",
                    content=(
                        LlmMessageContentPart.input_text(
                            "Use Google Search to cite one current affairs headline "
                            "and respond in text."
                        ),
                    ),
                ),
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("Share one current headline with a citation."),),
                ),
            ),
            temperature=0.2,
            max_output_tokens=256,
        )

        response = await provider.invoke(request)
        assert response.raw_text, "Vertex grounded response should include text output"
    finally:
        await provider.aclose()


async def test_vertex_claude_web_search_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    claude_model = LlmSettings().reference_llm_model or "claude-haiku-4-5@20251001"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    try:
        request = GroundedLlmRequest(
            provider="vertex",
            model=claude_model,
            messages=(
                LlmMessage(
                    role="system",
                    content=(
                        LlmMessageContentPart.input_text(
                            "Use web_search to fetch one current headline and reply in text with a citation."
                        ),
                    ),
                ),
                LlmMessage(
                    role="user",
                    content=(LlmMessageContentPart.input_text("Share one headline and cite the source."),),
                ),
            ),
            temperature=1.0,
            max_output_tokens=3072,
            reasoning_effort="2048",
        )

        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text, "Claude web_search response should include text output"
    assert response.usage.web_search_calls is not None


async def test_vertex_json_mode_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )
    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-flash-lite",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Return JSON only with key 'ping' and value 'pong'."),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        output_mode="json_object",
    )

    try:
        response = await provider.invoke(request)
        assert response.raw_text and "pong" in response.raw_text.lower()
    finally:
        await provider.aclose()


async def test_vertex_structured_output_live() -> None:
    vertex = VertexSettings()
    project = vertex.gcp_project_id
    location = vertex.gcp_location
    credentials_b64 = vertex.gcp_sa_credential_b64_value

    assert project, "GCP_PROJECT_ID must be configured"
    assert location, "GCP_LOCATION must be configured"
    assert credentials_b64, "Vertex credentials must be configured"

    provider = VertexLlmProvider(
        project=project,
        location=location,
        timeout=float(vertex.vertex_timeout_seconds or PLATFORM.timeout_seconds),
        credentials_path=None,
        service_account_b64=credentials_b64 or "",
    )

    class StructuredAnswer(BaseModel):
        answer: str

    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-flash-lite",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Respond only with JSON {"answer": "ok"}.'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=128,
        output_mode="structured",
        output_schema=StructuredAnswer,
    )

    try:
        response = await provider.invoke(request)
        assert response.raw_text and "ok" in response.raw_text.lower()
    finally:
        await provider.aclose()
