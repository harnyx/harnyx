from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig, OpenRouterModelProviderOptions
from harnyx_commons.llm.provider import LlmProviderConfigurationError
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_ENDPOINT_ID,
    OPENROUTER_INTERNAL_SUPPORTED_MODELS,
    OPENROUTER_INTERNAL_TO_NATIVE_MODEL,
    OPENROUTER_NATIVE_SUPPORTED_MODELS,
    OPENROUTER_SUPPORTED_MODELS,
    OpenRouterLlmProvider,
    build_openrouter_chat_provider,
)
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmThinkingConfig,
    LlmUsage,
)
from harnyx_commons.llm.tool_models import (
    MINER_SELECTED_LLM_PROVIDER_MODELS,
)

pytestmark = pytest.mark.anyio("asyncio")

OPENROUTER_TEST_MODELS = OPENROUTER_SUPPORTED_MODELS


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


def test_openrouter_supported_models_come_from_miner_selected_provider_contract() -> None:
    assert OPENROUTER_NATIVE_SUPPORTED_MODELS == MINER_SELECTED_LLM_PROVIDER_MODELS[OPENROUTER_PROVIDER]
    assert set(OPENROUTER_INTERNAL_SUPPORTED_MODELS) == set(OPENROUTER_INTERNAL_TO_NATIVE_MODEL)
    assert set(OPENROUTER_NATIVE_SUPPORTED_MODELS) <= set(OPENROUTER_SUPPORTED_MODELS)
    assert set(OPENROUTER_INTERNAL_SUPPORTED_MODELS) <= set(OPENROUTER_SUPPORTED_MODELS)


def test_openrouter_internal_routes_rewrite_only_internal_canonical_models() -> None:
    assert OPENROUTER_INTERNAL_TO_NATIVE_MODEL == {
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "zai-org/GLM-5-TEE": "z-ai/glm-5",
        "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
        "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    }


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_requires_key_only_when_openrouter_model_is_invoked(model: str) -> None:
    factory_calls: list[str] = []
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(""),
        model_provider_options={},
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(LlmProviderConfigurationError, match="OPENROUTER_API_KEY must be configured"):
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
        await provider.invoke(_request(model="unsupported-model"))

    assert factory_calls == []


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
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


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
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


async def test_openrouter_provider_preserves_request_retry_policy_for_delegate() -> None:
    fake_provider = _FakeOpenAiProvider()
    fake_client = _FakeClient()
    retry_policy = RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0.0)
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

    await provider.invoke(_request(model=OPENROUTER_NATIVE_SUPPORTED_MODELS[0], retry_policy=retry_policy))

    assert fake_provider.requests[0].retry_policy == retry_policy


def test_openrouter_provider_rejects_options_for_models_it_does_not_own() -> None:
    with pytest.raises(ValueError, match="unsupported models: unknown-model"):
        OpenRouterLlmProvider(
            openrouter_api_key=SecretStr("test-openrouter-key"),
            model_provider_options={"unknown-model": OpenRouterModelProviderOptions(require_parameters=True)},
        )


def test_build_openrouter_chat_provider_rejects_blank_key() -> None:
    with pytest.raises(LlmProviderConfigurationError, match="OPENROUTER_API_KEY must be configured"):
        build_openrouter_chat_provider(" ")


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
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

    expected_payload_model = OPENROUTER_INTERNAL_TO_NATIVE_MODEL.get(model, model)
    assert captured["url"] == f"{OPENROUTER_BASE_URL}/chat/completions"
    assert captured["authorization"] == "Bearer test-openrouter-key"
    assert captured["json"]["model"] == expected_payload_model
    assert captured["json"]["provider"] == {"order": ["Cerebras"], "require_parameters": True}
    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 2
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert response.metadata["raw_response"]["usage"]["cost"] == pytest.approx(0.0042)


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
@pytest.mark.parametrize(
    ("thinking", "expected_reasoning"),
    (
        (LlmThinkingConfig(enabled=True), {"enabled": True}),
        (LlmThinkingConfig(enabled=True, effort="high"), {"enabled": True, "effort": "high"}),
        (LlmThinkingConfig(enabled=True, budget=2048), {"enabled": True, "max_tokens": 2048}),
        (LlmThinkingConfig(enabled=False), {"effort": "none"}),
    ),
)
async def test_openrouter_provider_serializes_thinking_as_reasoning(
    model: str,
    thinking: LlmThinkingConfig,
    expected_reasoning: dict[str, object],
) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        body = "\n\n".join(
            (
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
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

    await provider.invoke(_request(model=model, thinking=thinking))
    await provider.aclose()

    assert captured["json"]["reasoning"] == expected_reasoning
    assert captured["json"]["provider"] == {"order": ["Cerebras"], "require_parameters": True}


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
@pytest.mark.parametrize("thinking", (None, LlmThinkingConfig(enabled=True)))
async def test_openrouter_provider_rejects_non_object_request_reasoning_extra(
    model: str,
    thinking: LlmThinkingConfig | None,
) -> None:
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        model_provider_options={},
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, []),
    )

    with pytest.raises(ValueError, match="OpenRouter request extra.reasoning must be an object"):
        await provider.invoke(
            _request(
                model=model,
                extra={"reasoning": "invalid"},
                thinking=thinking,
            )
        )


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_merges_request_reasoning_extra_with_typed_thinking(model: str) -> None:
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

    await provider.invoke(
        _request(
            model=model,
            extra={"reasoning": {"exclude": True, "effort": "low"}},
            thinking=LlmThinkingConfig(enabled=True, effort="high"),
        )
    )

    assert fake_provider.requests[0].extra == {
        "reasoning": {"exclude": True, "effort": "high", "enabled": True}
    }


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
    thinking: LlmThinkingConfig | None = None,
    retry_policy: RetryPolicy | None = None,
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
        thinking=thinking,
        retry_policy=retry_policy,
    )
