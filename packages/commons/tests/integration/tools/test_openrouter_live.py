from __future__ import annotations

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_commons.tools.invocation_clients import build_tool_llm_provider

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


@pytest.mark.parametrize(
    ("model", "provider_options_json"),
    (
        ("openai/gpt-oss-20b", '{"openai/gpt-oss-20b":{"require_parameters":true}}'),
        ("openai/gpt-oss-120b", '{"openai/gpt-oss-120b":{"require_parameters":true}}'),
    ),
)
async def test_chutes_tool_route_invokes_openrouter_gpt_oss_live(model: str, provider_options_json: str) -> None:
    settings = LlmSettings(
        TOOL_LLM_PROVIDER="chutes",
        OPENROUTER_MODEL_PROVIDER_OPTIONS_JSON=provider_options_json,
    )
    assert settings.openrouter_api_key_value, "OPENROUTER_API_KEY must be configured"

    registry = build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=45.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )
    provider = build_tool_llm_provider(settings, registry)
    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=256,
        timeout_seconds=180.0,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await registry.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    raw_usage = response.metadata["raw_response"]["usage"]
    assert isinstance(raw_usage["cost"], (int, float))
    assert raw_usage["cost"] >= 0.0
