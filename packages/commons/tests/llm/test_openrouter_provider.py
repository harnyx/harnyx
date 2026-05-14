from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig, OpenRouterModelProviderOptions
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_ENDPOINT_ID,
    OpenRouterLlmProvider,
    build_openrouter_chat_provider,
)
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")


class _FakeOpenAiProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []
        self.closed = False

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
                ),
            ),
            usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            metadata={"raw_response": {"id": "resp-1"}},
        )

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
async def test_openrouter_provider_requires_key_only_when_openrouter_model_is_invoked(model: str) -> None:
    factory_calls: list[str] = []
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(""),
        model_provider_options={},
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY must be configured"):
        await provider.invoke(_request(model=model))

    assert factory_calls == []


async def test_openrouter_provider_rejects_unsupported_model_before_key_lookup() -> None:
    factory_calls: list[str] = []
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(""),
        model_provider_options={},
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(ValueError, match="does not support model"):
        await provider.invoke(_request(model="deepseek-ai/DeepSeek-V3.2-TEE"))

    assert factory_calls == []


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
async def test_openrouter_provider_merges_per_model_provider_options(model: str) -> None:
    fake_provider = _FakeOpenAiProvider()
    fake_client = _FakeClient()
    seen_api_keys: list[str] = []

    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(" test-openrouter-key "),
        model_provider_options={
            model: OpenRouterModelProviderOptions(
                order=("Cerebras", "Groq"),
                require_parameters=True,
            )
        },
        openrouter_chat_provider_factory=lambda api_key: _fake_provider_factory(
            api_key,
            seen_api_keys,
            fake_provider,
            fake_client,
        ),
    )

    response = await provider.invoke(
        _request(
            model=model,
            extra={"provider": {"existing": "value"}, "metadata": {"trace": "test"}},
        )
    )
    await provider.aclose()

    assert seen_api_keys == ["test-openrouter-key"]
    assert fake_provider.requests[0].provider == "openrouter"
    assert fake_provider.requests[0].extra == {
        "provider": {"existing": "value", "order": ["Cerebras", "Groq"], "require_parameters": True},
        "metadata": {"trace": "test"},
    }
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert fake_provider.closed is True
    assert fake_client.closed is True


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
async def test_openrouter_provider_omits_provider_options_when_model_has_no_config(model: str) -> None:
    fake_provider = _FakeOpenAiProvider()
    fake_client = _FakeClient()
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        model_provider_options={},
        openrouter_chat_provider_factory=lambda api_key: _fake_provider_factory(
            api_key,
            [],
            fake_provider,
            fake_client,
        ),
    )

    await provider.invoke(_request(model=model))

    assert fake_provider.requests[0].extra is None


def test_openrouter_provider_rejects_options_for_models_it_does_not_own() -> None:
    with pytest.raises(ValueError, match="unsupported models: unknown-model"):
        OpenRouterLlmProvider(
            openrouter_api_key=SecretStr("test-openrouter-key"),
            model_provider_options={"unknown-model": OpenRouterModelProviderOptions(require_parameters=True)},
        )


def test_build_openrouter_chat_provider_rejects_blank_key() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY must be configured"):
        build_openrouter_chat_provider(" ")


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
async def test_openrouter_provider_serializes_openrouter_request_contract(model: str) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["json"] = json.loads(request.content.decode("utf-8"))
        body = "\n\n".join(
            (
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
                (
                    'data: {"id":"resp-1","choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.0042}}'
                ),
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(200, text=body, request=request, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": "Bearer test-openrouter-key"},
        transport=httpx.MockTransport(handler),
    )
    endpoint = OpenAiCompatibleEndpointConfig.model_validate(
        {
            "id": OPENROUTER_ENDPOINT_ID,
            "base_url": OPENROUTER_BASE_URL,
            "auth": {"type": "none"},
        }
    )
    openai_provider = OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client)
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        model_provider_options={
            model: OpenRouterModelProviderOptions(
                order=("Cerebras",),
                require_parameters=True,
            )
        },
        openrouter_chat_provider_factory=lambda _: (openai_provider, client),
    )

    response = await provider.invoke(_request(model=model))
    await provider.aclose()

    assert captured["url"] == f"{OPENROUTER_BASE_URL}/chat/completions"
    assert captured["authorization"] == "Bearer test-openrouter-key"
    assert captured["json"]["model"] == model
    assert captured["json"]["provider"] == {"order": ["Cerebras"], "require_parameters": True}
    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 2
    assert response.metadata is not None
    assert response.metadata["raw_response"]["usage"]["cost"] == pytest.approx(0.0042)


def _fake_factory(
    api_key: str,
    factory_calls: list[str],
) -> tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]:
    factory_calls.append(api_key)
    return cast(OpenAiCompatibleLlmProvider, _FakeOpenAiProvider()), cast(httpx.AsyncClient, _FakeClient())


def _fake_provider_factory(
    api_key: str,
    seen_api_keys: list[str],
    provider: _FakeOpenAiProvider,
    client: _FakeClient,
) -> tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]:
    seen_api_keys.append(api_key)
    return cast(OpenAiCompatibleLlmProvider, provider), cast(httpx.AsyncClient, client)


def _request(
    *,
    model: str,
    extra: dict[str, Any] | None = None,
) -> LlmRequest:
    return LlmRequest(
        provider="openrouter",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        extra=extra,
    )
