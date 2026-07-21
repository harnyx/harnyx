from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
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
    OpenRouterEmbeddingClient,
    OpenRouterLlmProvider,
    _openrouter_response_metadata,
    _openrouter_routing_evidence,
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
    def __init__(
        self,
        *,
        raw_response: dict[str, object] | None = None,
        actual_cost_evidence: dict[str, object] | None = None,
    ) -> None:
        self.requests: list[LlmRequest] = []
        self.closed = False
        self.raw_response = raw_response if raw_response is not None else {"id": "resp-1"}
        self.actual_cost_evidence = actual_cost_evidence

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        metadata: dict[str, object] = {"raw_response": self.raw_response}
        if self.actual_cost_evidence is not None:
            metadata["actual_cost_evidence"] = self.actual_cost_evidence
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
            metadata=metadata,
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
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(LlmProviderConfigurationError, match="OPENROUTER_API_KEY must be configured"):
        await provider.invoke(_request(model=model))

    assert factory_calls == []


async def test_openrouter_provider_rejects_unsupported_model_before_key_lookup() -> None:
    factory_calls: list[str] = []
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(""),
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(ValueError, match="does not support model"):
        await provider.invoke(_request(model="unsupported-model"))

    assert factory_calls == []


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_omits_extra_when_request_has_no_extra(model: str) -> None:
    fake_provider = _FakeOpenAiProvider()
    fake_client = _FakeClient()
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
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
        openrouter_chat_provider_factory=lambda api_key: _fake_provider_factory(
            api_key,
            [],
            fake_provider,
            fake_client,
        ),
    )

    await provider.invoke(_request(model=OPENROUTER_NATIVE_SUPPORTED_MODELS[0], retry_policy=retry_policy))

    assert fake_provider.requests[0].retry_policy == retry_policy


def test_build_openrouter_chat_provider_rejects_blank_key() -> None:
    with pytest.raises(LlmProviderConfigurationError, match="OPENROUTER_API_KEY must be configured"):
        build_openrouter_chat_provider(" ")


async def test_build_openrouter_chat_provider_enables_router_metadata() -> None:
    provider, client = build_openrouter_chat_provider("test-key")

    assert client.headers["X-OpenRouter-Metadata"] == "enabled"

    await provider.aclose()
    await client.aclose()


def test_openrouter_routing_evidence_requires_selected_endpoint_for_upstream_attribution() -> None:
    evidence = _openrouter_routing_evidence(
        {
            "id": "resp-cached",
            "provider": "OpenAI",
            "model": "openai/gpt-oss-20b",
        }
    )

    assert evidence == {
        "provider_request_id": "resp-cached",
    }


async def test_openrouter_provider_keeps_success_when_router_metadata_is_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_provider = _FakeOpenAiProvider(
        raw_response={
            "id": "resp-1",
            "openrouter_metadata": {
                "endpoints": {
                    "available": [
                        {
                            "model": "openai/gpt-oss-20b",
                            "selected": True,
                        }
                    ]
                }
            },
        },
        actual_cost_evidence={"settlement_source": "provider_returned"},
    )
    fake_client = _FakeClient()
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openrouter_chat_provider_factory=lambda api_key: _fake_provider_factory(
            api_key,
            [],
            fake_provider,
            fake_client,
        ),
    )

    with caplog.at_level("WARNING", logger="harnyx_commons.llm.providers.openrouter"):
        response = await provider.invoke(_request(model=OPENROUTER_NATIVE_SUPPORTED_MODELS[0]))

    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["actual_cost_evidence"] == {
        "settlement_source": "provider_returned",
        "provider_request_id": "resp-1",
    }
    assert "OpenRouter router metadata is malformed" in caplog.messages


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_serializes_openrouter_request_contract(model: str) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["router_metadata"] = request.headers.get("X-OpenRouter-Metadata")
        captured["json"] = json.loads(request.content.decode("utf-8"))
        body = "\n\n".join(
            (
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
                (
                    'data: {"id":"resp-1","choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.0042},'
                    '"openrouter_metadata":{"endpoints":{"available":['
                    '{"provider":"Cerebras","model":"openai/gpt-oss-20b","selected":true}]}}}'
                ),
                "data: [DONE]",
                "",
            )
        )
        return httpx.Response(200, text=body, request=request, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={
            "Authorization": "Bearer test-openrouter-key",
            "X-OpenRouter-Metadata": "enabled",
        },
        transport=httpx.MockTransport(handler),
    )
    endpoint = OpenAiCompatibleEndpointConfig.model_validate(
        {
            "id": OPENROUTER_ENDPOINT_ID,
            "base_url": OPENROUTER_BASE_URL,
            "auth": {"type": "none"},
        }
    )
    openai_provider = OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=client,
        response_metadata_extractor=_openrouter_response_metadata,
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openrouter_chat_provider_factory=lambda _: (openai_provider, client),
    )

    response = await provider.invoke(_request(model=model, extra={"provider": {"only": ["cerebras"]}}))
    await provider.aclose()

    expected_payload_model = OPENROUTER_INTERNAL_TO_NATIVE_MODEL.get(model, model)
    assert captured["url"] == f"{OPENROUTER_BASE_URL}/chat/completions"
    assert captured["authorization"] == "Bearer test-openrouter-key"
    assert captured["router_metadata"] == "enabled"
    assert captured["json"]["model"] == expected_payload_model
    assert captured["json"]["provider"] == {"only": ["cerebras"]}
    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 2
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert response.metadata["raw_response"]["usage"]["cost"] == pytest.approx(0.0042)
    assert response.metadata["actual_cost_provider"] == "openrouter"
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0042)
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"
    assert response.metadata["actual_cost_evidence"]["upstream_provider"] == "Cerebras"
    assert response.metadata["actual_cost_evidence"]["upstream_model"] == "openai/gpt-oss-20b"
    assert response.metadata["actual_cost_evidence"]["provider_request_id"] == "resp-1"


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_preserves_nested_reasoning_usage(model: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = "\n\n".join(
            (
                'data: {"id":"resp-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
                (
                    'data: {"id":"resp-1","choices":[{"index":0,"delta":{},'
                    '"finish_reason":"stop"}],'
                    '"usage":{"prompt_tokens":3,"completion_tokens":6,'
                    '"completion_tokens_details":{"reasoning_tokens":4},"total_tokens":9}}'
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
        openrouter_chat_provider_factory=lambda _: (openai_provider, client),
    )

    response = await provider.invoke(_request(model=model))
    await provider.aclose()

    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.usage.total_tokens == 9


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
async def test_openrouter_provider_serializes_request_provider_only_extra(model: str) -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
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
        openrouter_chat_provider_factory=lambda _: (openai_provider, client),
    )

    await provider.invoke(_request(model=model, extra={"provider": {"only": ["cerebras"]}}))
    await provider.aclose()

    assert captured["json"]["provider"] == {"only": ["cerebras"]}


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
        openrouter_chat_provider_factory=lambda _: (openai_provider, client),
    )

    await provider.invoke(_request(model=model, thinking=thinking))
    await provider.aclose()

    assert captured["json"]["reasoning"] == expected_reasoning
    assert "provider" not in captured["json"]


@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
@pytest.mark.parametrize("thinking", (None, LlmThinkingConfig(enabled=True)))
async def test_openrouter_provider_rejects_non_object_request_reasoning_extra(
    model: str,
    thinking: LlmThinkingConfig | None,
) -> None:
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
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


async def test_openrouter_embedding_client_posts_embeddings_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.read().decode())
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "id": "gen-emb-1",
                "model": "Qwen/Qwen3-Embedding-8B",
                "usage": {
                    "prompt_tokens": 5,
                    "total_tokens": 5,
                    "cost": 0.00012,
                    "cost_details": {"upstream_inference_cost": 0.0001},
                },
            },
        )

    client = OpenRouterEmbeddingClient(
        model="qwen/qwen3-embedding-8b",
        api_key="test-key",
        client=httpx.AsyncClient(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler)),
        dimensions=3,
    )

    response = await client.embed_many(
        ("hello",),
        extra={"provider": {"only": ["nebius"], "allow_fallbacks": False}},
        timeout_seconds=190.0,
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/embeddings"
    assert captured["json"] == {
        "model": "qwen/qwen3-embedding-8b",
        "input": "hello",
        "dimensions": 3,
        "provider": {"only": ["nebius"], "allow_fallbacks": False},
    }
    assert captured["timeout"] == {
        "connect": 190.0,
        "read": 190.0,
        "write": 190.0,
        "pool": 190.0,
    }
    assert response.vectors == ((0.1, 0.2, 0.3),)
    assert response.usage is not None
    assert response.usage.prompt_tokens == 5
    assert response.usage.cost == pytest.approx(0.00012)
    assert response.usage.cost_details == {"upstream_inference_cost": 0.0001}
    assert response.id == "gen-emb-1"
    assert response.model == "Qwen/Qwen3-Embedding-8B"


async def test_openrouter_embedding_client_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="does not support model"):
        OpenRouterEmbeddingClient(model="unsupported", api_key="test-key")


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
