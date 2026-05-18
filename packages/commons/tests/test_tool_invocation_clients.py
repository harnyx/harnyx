from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools import invocation_clients
from harnyx_commons.tools.invocation_clients import build_tool_invocation_clients

GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
GEMMA_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run-turbo"
QWEN36_MODEL = "Qwen/Qwen3.6-27B-TEE"
QWEN36_ROUTE_TARGET = "custom-openai-compatible:qwen36-cloud-run"
OPENROUTER_ROUTED_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", QWEN36_MODEL)


class _FakeLlmProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="resp-1",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(),
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


class _FakeLlmRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, _FakeLlmProvider] = {}

    @property
    def requests_by_provider(self) -> dict[str, list[LlmRequest]]:
        return {provider_name: provider.requests for provider_name, provider in self._providers.items()}

    def resolve(self, name: str) -> _FakeLlmProvider:
        provider = self._providers.get(name)
        if provider is None:
            provider = _FakeLlmProvider()
            self._providers[name] = provider
        return provider


def _llm_settings() -> LlmSettings:
    return _llm_settings_with_tool_overrides(
        {
            GEMMA_MODEL: GEMMA_ROUTE_TARGET,
            QWEN36_MODEL: QWEN36_ROUTE_TARGET,
        }
    )


def _llm_settings_without_qwen36_override() -> LlmSettings:
    return _llm_settings_with_tool_overrides({GEMMA_MODEL: GEMMA_ROUTE_TARGET})


def _llm_settings_with_tool_overrides(tool_overrides: dict[str, str]) -> LlmSettings:
    return LlmSettings.model_construct(
        search_provider=None,
        tool_llm_provider="chutes",
        chutes_api_key=SecretStr("test-key"),
        llm_model_provider_overrides_json=json.dumps({"tool": tool_overrides}),
        openai_compatible_endpoints_json=json.dumps(
            [
                {
                    "id": "gemma4-cloud-run-turbo",
                    "base_url": "https://gemma.example.run.app/v1",
                    "auth": {"type": "none"},
                },
                {
                    "id": "qwen36-cloud-run",
                    "base_url": "https://qwen.example.run.app/v1",
                    "auth": {"type": "none"},
                },
            ]
        ),
    )


def _gemma_tool_request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=GEMMA_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def _qwen36_tool_request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=QWEN36_MODEL,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def _openrouter_tool_request(*, model: str) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def test_tool_invocation_clients_do_not_resolve_tool_provider_until_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            calls.append(name)
            return f"provider:{name}"

    monkeypatch.setattr(
        invocation_clients,
        "build_cached_llm_provider_registry",
        lambda **_: _FakeRegistry(),
    )

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.search_client is None
    assert clients.tool_llm_provider is not None
    assert calls == []


def test_tool_invocation_clients_can_require_search_provider() -> None:
    with pytest.raises(RuntimeError, match="SEARCH_PROVIDER must be configured"):
        build_tool_invocation_clients(
            llm_settings=_llm_settings(),
            bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
            vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
            require_search=True,
        )


@pytest.mark.anyio("asyncio")
async def test_tool_invocation_clients_route_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_gemma_tool_request())

    assert registry.requests_by_provider["custom-openai-compatible:gemma4-cloud-run-turbo"][0].provider == (
        "custom-openai-compatible:gemma4-cloud-run-turbo"
    )


@pytest.mark.anyio("asyncio")
async def test_tool_invocation_clients_route_qwen36_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)

    clients = build_tool_invocation_clients(
        llm_settings=_llm_settings(),
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_qwen36_tool_request())

    assert registry.requests_by_provider[QWEN36_ROUTE_TARGET][0].provider == QWEN36_ROUTE_TARGET


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("model", OPENROUTER_ROUTED_MODELS)
async def test_tool_invocation_clients_route_chutes_selected_openrouter_model_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(invocation_clients, "build_cached_llm_provider_registry", lambda **_: registry)
    llm_settings = _llm_settings_without_qwen36_override() if model == QWEN36_MODEL else _llm_settings()

    clients = build_tool_invocation_clients(
        llm_settings=llm_settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
    )

    assert clients.tool_llm_provider is not None
    await clients.tool_llm_provider.invoke(_openrouter_tool_request(model=model))

    assert registry.requests_by_provider["openrouter"][0].provider == "openrouter"


@pytest.mark.parametrize(
    "llm_settings",
    [
        LlmSettings.model_construct(tool_llm_provider="bedrock"),
        LlmSettings.model_construct(
            tool_llm_provider="chutes",
            llm_model_provider_overrides_json=json.dumps({"tool": {"sample-tool-model": "bedrock"}}),
        ),
    ],
)
def test_tool_invocation_clients_reject_bedrock_tool_routes(llm_settings: LlmSettings) -> None:
    with pytest.raises(ValueError, match="TOOL_LLM_PROVIDER='bedrock' is not supported"):
        build_tool_invocation_clients(
            llm_settings=llm_settings,
            bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
            vertex_settings=VertexSettings.model_construct(gcp_project_id="project", gcp_location="us-central1"),
        )
