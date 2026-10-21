from __future__ import annotations

import asyncio
import json
from dataclasses import replace
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
    LlmTool,
    LlmUsage,
)
from harnyx_commons.llm.tool_models import (
    MINER_SELECTED_LLM_PROVIDER_MODELS,
)

pytestmark = pytest.mark.anyio("asyncio")

OPENROUTER_NATIVE_TEST_MODEL = OPENROUTER_NATIVE_SUPPORTED_MODELS[0]
OPENROUTER_INTERNAL_TEST_MODEL = OPENROUTER_INTERNAL_SUPPORTED_MODELS[0]
OPENROUTER_ROUTE_TEST_MODELS = (
    OPENROUTER_NATIVE_TEST_MODEL,
    OPENROUTER_INTERNAL_TEST_MODEL,
)


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
    assert OPENROUTER_INTERNAL_TO_NATIVE_MODEL == {
        "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
        "deepseek-ai/DeepSeek-V4-Flash-0731-TEE": "deepseek/deepseek-v4-flash-0731",
        "zai-org/GLM-5-TEE": "z-ai/glm-5",
        "Qwen/Qwen3.6-27B-TEE": "qwen/qwen3.6-27b",
        "google/gemma-4-31B-turbo-TEE": "google/gemma-4-31b-it",
    }
    assert set(OPENROUTER_INTERNAL_SUPPORTED_MODELS) == set(OPENROUTER_INTERNAL_TO_NATIVE_MODEL)
    assert set(OPENROUTER_NATIVE_SUPPORTED_MODELS) <= set(OPENROUTER_SUPPORTED_MODELS)
    assert set(OPENROUTER_INTERNAL_SUPPORTED_MODELS) <= set(OPENROUTER_SUPPORTED_MODELS)


async def test_openrouter_provider_requires_key_only_when_openrouter_model_is_invoked() -> None:
    factory_calls: list[str] = []
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr(""),
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, factory_calls),
    )

    with pytest.raises(LlmProviderConfigurationError, match="OPENROUTER_API_KEY must be configured"):
        await provider.invoke(_request(model=OPENROUTER_NATIVE_TEST_MODEL))

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


async def test_openrouter_provider_allows_explicit_additional_research_model() -> None:
    fake_provider = _FakeOpenAiProvider()
    fake_client = _FakeClient()
    model = "qwen/qwen3.8-27b"
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openrouter_chat_provider_factory=lambda api_key: _fake_provider_factory(
            api_key,
            [],
            fake_provider,
            fake_client,
        ),
        additional_supported_models=(model,),
    )

    await provider.invoke(_request(model=model))

    assert fake_provider.requests[0].model == model


async def test_openrouter_provider_omits_extra_when_request_has_no_extra() -> None:
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

    await provider.invoke(_request(model=OPENROUTER_NATIVE_TEST_MODEL))

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


@pytest.mark.parametrize("model", OPENROUTER_ROUTE_TEST_MODELS)
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


async def test_openrouter_provider_preserves_nested_reasoning_usage() -> None:
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

    response = await provider.invoke(_request(model=OPENROUTER_NATIVE_TEST_MODEL))
    await provider.aclose()

    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.usage.total_tokens == 9


async def test_openrouter_provider_serializes_request_provider_only_extra() -> None:
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

    await provider.invoke(
        _request(
            model=OPENROUTER_NATIVE_TEST_MODEL,
            extra={"provider": {"only": ["cerebras"]}},
        )
    )
    await provider.aclose()

    assert captured["json"]["provider"] == {"only": ["cerebras"]}


@pytest.mark.parametrize(
    ("thinking", "reasoning_effort", "expected_reasoning"),
    (
        (LlmThinkingConfig(enabled=True), None, {"enabled": True}),
        (LlmThinkingConfig(enabled=True, effort="high"), None, {"enabled": True, "effort": "high"}),
        (LlmThinkingConfig(enabled=True, budget=2048), None, {"enabled": True, "max_tokens": 2048}),
        (LlmThinkingConfig(enabled=False), None, {"effort": "none"}),
        (None, "high", {"enabled": True, "effort": "high"}),
        (LlmThinkingConfig(enabled=False), "high", {"effort": "none"}),
    ),
)
async def test_openrouter_provider_serializes_resolved_thinking_as_reasoning(
    thinking: LlmThinkingConfig | None,
    reasoning_effort: str | None,
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

    await provider.invoke(
        _request(
            model=OPENROUTER_NATIVE_TEST_MODEL,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
    )
    await provider.aclose()

    assert captured["json"]["reasoning"] == expected_reasoning
    assert "provider" not in captured["json"]


@pytest.mark.parametrize("thinking", (None, LlmThinkingConfig(enabled=True)))
async def test_openrouter_provider_rejects_non_object_request_reasoning_extra(
    thinking: LlmThinkingConfig | None,
) -> None:
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openrouter_chat_provider_factory=lambda api_key: _fake_factory(api_key, []),
    )

    with pytest.raises(ValueError, match="OpenRouter request extra.reasoning must be an object"):
        await provider.invoke(
            _request(
                model=OPENROUTER_NATIVE_TEST_MODEL,
                extra={"reasoning": "invalid"},
                thinking=thinking,
            )
        )


@pytest.mark.parametrize(
    ("thinking", "reasoning_effort"),
    (
        (LlmThinkingConfig(enabled=True, effort="high"), None),
        (None, "high"),
    ),
)
async def test_openrouter_provider_merges_request_reasoning_extra_with_resolved_thinking(
    thinking: LlmThinkingConfig | None,
    reasoning_effort: str | None,
) -> None:
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
            model=OPENROUTER_NATIVE_TEST_MODEL,
            extra={"reasoning": {"exclude": True, "effort": "low"}},
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )
    )

    assert fake_provider.requests[0].extra == {"reasoning": {"exclude": True, "effort": "high", "enabled": True}}


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
        api_key=SecretStr("test-key"),
        client=httpx.AsyncClient(base_url=OPENROUTER_BASE_URL, transport=httpx.MockTransport(handler)),
        dimensions=3,
    )
    assert "test-key" not in repr(client)

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
        OpenRouterEmbeddingClient(model="unsupported", api_key=SecretStr("test-key"))


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
    reasoning_effort: str | None = None,
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
        reasoning_effort=reasoning_effort,
        retry_policy=retry_policy,
    )


@pytest.mark.parametrize("returned_cost", [None, True, 0.0042])
async def test_openrouter_cached_input_settlement(returned_cost: object) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        chunk = {
            "id": "resp-cache",
            "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens": 6,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 1006,
                "cost": returned_cost,
            },
        }
        return httpx.Response(
            200,
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            request=request,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        endpoint = OpenAiCompatibleEndpointConfig.model_validate(
            {"id": OPENROUTER_ENDPOINT_ID, "base_url": OPENROUTER_BASE_URL, "auth": {"type": "none"}}
        )
        chat = OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client)
        provider = OpenRouterLlmProvider(
            openrouter_api_key=SecretStr("test-key"),
            openrouter_chat_provider_factory=lambda key: (chat, client),
        )
        try:
            response = await provider.invoke(_request(model="meta/muse-spark-open"))
        finally:
            await provider.aclose()

    assert response.usage.prompt_cached_tokens == 800
    assert response.usage.completion_tokens == 2
    assert response.usage.reasoning_tokens == 4
    assert response.metadata is not None
    assert response.metadata["actual_cost_usd"] == pytest.approx(0.0042 if returned_cost == 0.0042 else 0.0003955)


async def test_decision_native_transport_and_client_ownership() -> None:
    import json

    from harnyx_commons.llm.providers.openrouter import OpenRouterDecisionClient

    captured: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "cloudflare/clef-flash",
                "answers": {"b": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 12, "output_tokens": 1, "cost": 0.002},
            },
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.ai/api/alpha/", transport=httpx.MockTransport(handle)
    ) as http:
        client = OpenRouterDecisionClient(model="cloudflare/clef-flash", api_key=SecretStr("private-key"), client=http)
        result = await client.query("state", {"b": {"type": "boolean", "instructions": "?"}})
        assert str(captured[0].url) == "https://openrouter.ai/api/alpha/decisions"
        assert captured[0].headers["Authorization"] == "Bearer private-key"
        assert json.loads(captured[0].content)["questions"]["b"]["type"] == "noul"
        assert result.raw_payload["usage"] == {"input_tokens": 12, "output_tokens": 1, "cost": 0.002}
        await client.aclose()
        assert not http.is_closed
    owned = OpenRouterDecisionClient(model="cloudflare/clef-flash", api_key=SecretStr("private-key"))
    await owned.aclose()
    assert owned.client is not None and owned.client.is_closed


class _MinerChatDelegate:
    def __init__(self, *, serving_provider: str = "Inception", rejected: bool = False, byok: bool = False) -> None:
        self.serving_provider = serving_provider
        self.rejected = rejected
        self.byok = byok
        self.calls = 0

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        from harnyx_commons.llm.provider import LlmRetryExhaustedError

        self.calls += 1
        if request.on_provider_dispatch is not None:
            request.on_provider_dispatch()
        response = LlmResponse(
            id="promo",
            choices=(),
            usage=LlmUsage(prompt_tokens=10, completion_tokens=5),
            metadata={
                "raw_response": {
                    "model": "inception/mercury-3",
                    "provider": self.serving_provider,
                    "usage": {"cost": 0.004, "is_byok": self.byok, "cost_details": {"upstream_inference_cost": 0.002}},
                }
            },
        )
        if self.rejected:
            raise LlmRetryExhaustedError("rejected output", response=response, attempts=1)
        return response

    async def aclose(self) -> None:
        pass


def _miner_pricing_catalog(*, discount: float = 0.5, unsupported_other: bool = False, conditional: bool = False):
    from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingCatalog

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "inception/mercury-3",
                        "endpoints": [
                            {
                                "provider_name": "Inception",
                                "tag": "inception",
                                "model_id": "inception/mercury-3",
                                "pricing": {
                                    "prompt": "0.1",
                                    "completion": "0.35",
                                    "discount": discount,
                                    **(
                                        {
                                            "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.2"}],
                                            "request": "0.001",
                                            "web_search": "0.01",
                                            "input_cache_read": "0.01",
                                        }
                                        if conditional
                                        else {}
                                    ),
                                },
                            }
                        ]
                        + (
                            [
                                {
                                    "provider_name": "Other",
                                    "tag": "other",
                                    "model_id": "inception/mercury-3",
                                    "pricing": {"prompt": "0.1", "completion": "0.35", "discount": None},
                                }
                            ]
                            if unsupported_other
                            else []
                        ),
                    }
                },
            )
        return httpx.Response(200, json={"data": [{"id": "inception/mercury-3"}]})

    return OpenRouterPricingCatalog(transport=httpx.MockTransport(serve))


def _miner_promo_request() -> LlmRequest:
    return LlmRequest(
        provider="openrouter",
        model="inception/mercury-3",
        messages=(),
        temperature=0.0,
        max_output_tokens=None,
        use_case="tool_runtime_invoker",
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0.0),
    )


@pytest.mark.parametrize("byok,expected_actual,expected_reference", [(False, 0.004, 0.008), (True, 0.006, 0.008)])
async def test_miner_promotion_separates_inference_discount_from_actual_and_router_charge(
    byok: bool,
    expected_actual: float,
    expected_reference: float,
) -> None:
    delegate = _MinerChatDelegate(byok=byok)
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    response = await provider.invoke(_miner_promo_request())
    assert response.metadata["actual_cost_usd"] == pytest.approx(expected_actual)
    assert response.metadata["reference_cost_usd"] == pytest.approx(expected_reference)
    assert delegate.calls == 1
    await provider.aclose()


@pytest.mark.parametrize(
    "discount,byok,charge,prompt_tokens,expected_actual,expected_reference",
    [
        (0.5, False, 0.004, 10, 0.004, 0.008),
        (0.5, True, 0.004, 10, 0.006, 0.008),
        (0, False, 0.014, 10, 0.014, 0.014),
        (0, False, 0.024, 272001, 0.024, 0.024),
        (0, True, 0.001, 272001, 0.003, 0.003),
        (0, True, 0, 10, 0.002, 0.002),
    ],
)
async def test_successful_miner_stream_keeps_accounting_through_runtime_settlement(
    discount: float,
    byok: bool,
    charge: float,
    prompt_tokens: int,
    expected_actual: float,
    expected_reference: float,
) -> None:
    """A successful paid answer must retain billing facts through the real retry adapter and runtime reader."""
    from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
    from harnyx_commons.tools.executor import ToolInvocationOutput
    from harnyx_commons.tools.provider_billing import OpenRouterAccounting
    from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker

    frame = {
        "id": "mercury-canary",
        "model": "inception/mercury-3",
        "provider": "Inception",
        "choices": [{"index": 0, "delta": {"content": "Mercury 3 canary OK."}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": 8},
            "completion_tokens": 5,
            "total_tokens": prompt_tokens + 5,
            "cost": charge,
            "is_byok": byok,
            "cost_details": {"upstream_inference_cost": 0.002},
        },
    }
    body = f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
    delegate = OpenAiCompatibleLlmProvider(
        endpoint=OpenAiCompatibleEndpointConfig.model_validate(
            {"id": "openrouter", "base_url": OPENROUTER_BASE_URL, "auth": {"type": "none"}}
        ),
        client=client,
        response_metadata_extractor=_openrouter_response_metadata,
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=discount, conditional=discount == 0),
        openrouter_chat_provider_factory=lambda _: (delegate, client),
    )
    invoker = RuntimeToolInvoker(InMemoryReceiptLog(), llm_provider=provider, llm_provider_name="openrouter")
    try:
        output = await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": "inception/mercury-3",
                "messages": [{"role": "user", "content": "Reply exactly: Mercury 3 canary OK."}],
            },
        )
        assert isinstance(output, ToolInvocationOutput)
        fact = OpenRouterAccounting.from_fields(
            actual_cost_usd=output.actual_cost_usd,
            reference_cost_usd=output.reference_cost_usd,
            evidence=output.actual_cost_evidence,
        )
        assert fact.inference == "attempted"
        assert fact.actual_cost_usd == pytest.approx(expected_actual)
        assert fact.reference_cost_usd == pytest.approx(expected_reference)
        assert fact.llm_usage == LlmUsage(
            prompt_tokens=prompt_tokens, prompt_cached_tokens=8, completion_tokens=5, total_tokens=prompt_tokens + 5
        )
        assert output.public_payload["choices"][0]["message"]["content"][0]["text"] == "Mercury 3 canary OK."
    finally:
        await provider.aclose()


@pytest.mark.parametrize("dispatched", [False, True])
async def test_miner_provider_error_keeps_dispatch_evidence_without_a_response(dispatched: bool) -> None:
    """A typed delegate error must preserve known dispatch instead of bypassing billing facts."""
    from harnyx_commons.errors import ToolProviderError
    from harnyx_commons.tools.provider_billing import OpenRouterAccounting

    class Delegate:
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            if dispatched:
                request.on_provider_dispatch()
            raise ToolProviderError("delegate rejected request")

        async def aclose(self) -> None:
            pass

    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(),
        openrouter_chat_provider_factory=lambda _: (Delegate(), _FakeClient()),
    )
    try:
        with pytest.raises(ToolProviderError) as caught:
            await provider.invoke(_miner_promo_request())
        fact = OpenRouterAccounting.from_billing(caught.value.billing)
        assert fact.inference == ("attempted" if dispatched else "not_dispatched")
        assert fact.actual_cost_usd == (None if dispatched else 0.0)
        assert fact.reference_cost_unavailable is dispatched
    finally:
        await provider.aclose()


@pytest.mark.parametrize("serving_provider,unavailable", [("Inception", False), ("Unknown", True)])
@pytest.mark.parametrize("discount", [0, 0.5])
async def test_rejected_billable_promotion_keeps_both_amounts_or_explicit_unknown(
    serving_provider: str,
    unavailable: bool,
    discount: float,
) -> None:
    from harnyx_commons.errors import ToolProviderError

    delegate = _MinerChatDelegate(serving_provider=serving_provider, rejected=True)
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=discount, conditional=discount == 0),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    with pytest.raises(ToolProviderError) as caught:
        await provider.invoke(_miner_promo_request())
    assert caught.value.billing.actual_cost_usd == pytest.approx(0.004)
    assert caught.value.billing.reference_cost_unavailable is unavailable
    assert caught.value.billing.reference_cost_usd == (None if unavailable else pytest.approx(0.004 / (1 - discount)))
    await provider.aclose()


async def test_unpriceable_promotion_rejects_before_paid_dispatch() -> None:
    from harnyx_commons.errors import ToolProviderError

    delegate = _MinerChatDelegate()
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=1),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    with pytest.raises(ToolProviderError) as caught:
        await provider.invoke(_miner_promo_request())
    assert delegate.calls == 0
    assert caught.value.billing.reference_cost_usd == 0
    assert caught.value.billing.actual_cost_usd == 0
    assert not caught.value.billing.reference_cost_unavailable
    await provider.aclose()


@pytest.mark.parametrize("cost", [None, "invalid", -0.01, "NaN", "Infinity"])
@pytest.mark.parametrize("byok", [False, True])
async def test_zero_discount_conditional_route_requires_valid_original_charge(cost: object, byok: bool) -> None:
    """Relaxed route eligibility must not make missing or malformed billing a free successful answer."""
    from harnyx_commons.errors import ToolProviderError

    delegate = _FakeOpenAiProvider(
        raw_response={
            "model": "inception/mercury-3",
            "provider": "Inception",
            "usage": {
                "cost": 0.001 if byok else cost,
                "is_byok": byok,
                "cost_details": {"upstream_inference_cost": cost},
            },
        }
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=0, conditional=True),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    try:
        with pytest.raises(ToolProviderError) as caught:
            await provider.invoke(_miner_promo_request())
        assert len(delegate.requests) == 1
        assert caught.value.billing.actual_cost_usd is None
        assert caught.value.billing.reference_cost_unavailable
        assert caught.value.billing.reference_cost_usd is None
    finally:
        await provider.aclose()


async def test_zero_discount_conditional_route_rejects_wrong_model_and_keeps_charge() -> None:
    """A paid completion for another model must retain its charge without becoming comparable."""
    from harnyx_commons.errors import ToolProviderError

    delegate = _FakeOpenAiProvider(
        raw_response={"model": "other/model", "provider": "Inception", "usage": {"cost": 0.014}}
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=0, conditional=True),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    try:
        with pytest.raises(ToolProviderError) as caught:
            await provider.invoke(_miner_promo_request())
        assert caught.value.billing.actual_cost_usd == pytest.approx(0.014)
        assert caught.value.billing.reference_cost_unavailable
        assert "does not match" in caught.value.billing.provider_usage["reference_unavailable_reason"]
    finally:
        await provider.aclose()


async def test_zero_discount_conditional_calls_settle_after_concurrent_budget_exhaustion() -> None:
    """A second admitted paid completion must still reach receipts and both totals after exhaustion."""
    from datetime import UTC, datetime, timedelta
    from uuid import uuid4

    from harnyx_commons.domain.session import Session, SessionStatus
    from harnyx_commons.errors import BudgetExceededError
    from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
    from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
    from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
    from harnyx_commons.tools.dto import ToolInvocationRequest
    from harnyx_commons.tools.executor import ToolExecutor
    from harnyx_commons.tools.receipt_accounting import usage_from_receipts
    from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker
    from harnyx_commons.tools.usage_tracker import UsageTracker

    releases = (asyncio.Event(), asyncio.Event())
    both_dispatched = asyncio.Event()

    class ConcurrentDelegate(_MinerChatDelegate):
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = await super().invoke(request)
            index = self.calls - 1
            if self.calls == 2:
                both_dispatched.set()
            await releases[index].wait()
            return response

    delegate = ConcurrentDelegate(byok=True)
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=0, conditional=True),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    now = datetime.now(UTC)
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=now,
        expires_at=now + timedelta(minutes=1),
        budget_usd=0.005,
        hard_limit_usd=0.005,
    )
    sessions, receipts, tokens = InMemorySessionRegistry(), InMemoryReceiptLog(), InMemoryTokenRegistry()
    sessions.create(session)
    token = uuid4().hex
    tokens.register(session.session_id, token)
    invoker = RuntimeToolInvoker(receipts, llm_provider_resolver=lambda _provider, _context: provider)
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        tool_invoker=invoker,
        token_registry=tokens,
        clock=lambda: now,
    )
    request = ToolInvocationRequest(
        session.session_id,
        token,
        "llm_chat",
        (),
        {
            "provider": "openrouter",
            "model": "inception/mercury-3",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    tasks = [asyncio.create_task(executor.execute(request)) for _ in range(2)]
    try:
        await asyncio.wait_for(both_dispatched.wait(), timeout=5)
        releases[0].set()
        with pytest.raises(BudgetExceededError):
            await tasks[0]
        assert sessions.get(session.session_id).status is SessionStatus.EXHAUSTED
        releases[1].set()
        await tasks[1]
        settled = receipts.for_session(session.session_id)
        usage = sessions.get(session.session_id).usage
        assert len(settled) == 2
        assert usage == usage_from_receipts(settled)
        assert usage.total_cost_usd == pytest.approx(0.012)
        assert usage.actual_total_cost_usd == pytest.approx(0.012)
        assert usage.llm_usage_totals["openrouter"]["inception/mercury-3"].call_count == 2
        assert sessions.get(session.session_id).status is SessionStatus.EXHAUSTED
        with pytest.raises(RuntimeError, match="not active"):
            await executor.execute(request)
        assert delegate.calls == 2
    finally:
        for release in releases:
            release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await provider.aclose()


@pytest.mark.parametrize(
    "only,serving_provider,expected_calls",
    [(None, "Inception", 0), (["inception"], "Inception", 1), (["inception"], "Other", 1)],
)
@pytest.mark.parametrize("discount", [0, 0.5])
async def test_miner_pricing_honors_provider_only_and_checks_actual_serving_provider(
    only: list[str] | None, serving_provider: str, expected_calls: int, discount: float
) -> None:
    from harnyx_commons.errors import ToolProviderError

    delegate = _MinerChatDelegate(serving_provider=serving_provider)
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(discount=discount, unsupported_other=True, conditional=discount == 0),
        openrouter_chat_provider_factory=lambda _: (delegate, _FakeClient()),
    )
    request = replace(
        _miner_promo_request(), extra={"provider": {"allow_fallbacks": False, **({"only": only} if only else {})}}
    )
    if only and serving_provider == "Inception":
        response = await provider.invoke(request)
        assert response.metadata["reference_cost_usd"] == pytest.approx(0.004 / (1 - discount))
    else:
        with pytest.raises(ToolProviderError) as caught:
            await provider.invoke(request)
        assert caught.value.billing.reference_cost_unavailable is bool(expected_calls)
    assert delegate.calls == expected_calls
    await provider.aclose()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_billing_frame_survives_broken_or_cancelled_stream(cancelled: bool) -> None:
    import asyncio

    from harnyx_commons.errors import ToolInvocationCancelledError, ToolProviderError

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'data: {"id":"promo-stream","model":"inception/mercury-3","provider":"Inception",'
                b'"choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n'
            )
            yield (
                b'data: {"id":"promo-stream","usage":{"cost":0.004,"is_byok":false,"prompt_tokens":10,'
                b'"completion_tokens":5,"total_tokens":15},"choices":[]}\n\n'
            )
            if cancelled:
                raise asyncio.CancelledError()
            raise httpx.ReadError("stream interrupted after billing")

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=InterruptedStream())))
    delegate = OpenAiCompatibleLlmProvider(
        endpoint=OpenAiCompatibleEndpointConfig.model_validate(
            {
                "id": "openrouter",
                "base_url": OPENROUTER_BASE_URL,
                "auth": {"type": "none"},
            }
        ),
        client=client,
        response_metadata_extractor=_openrouter_response_metadata,
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(),
        openrouter_chat_provider_factory=lambda _: (delegate, client),
    )
    try:
        with pytest.raises(ToolInvocationCancelledError if cancelled else ToolProviderError) as caught:
            await provider.invoke(_miner_promo_request())
        assert caught.value.billing.actual_cost_usd == pytest.approx(0.004)
        assert caught.value.billing.reference_cost_usd == pytest.approx(0.008)
        assert caught.value.billing.llm_usage.prompt_tokens == 10
    finally:
        await provider.aclose()


def test_ambiguous_selected_providers_never_invent_a_reference_cost() -> None:
    from harnyx_commons.llm.providers.openrouter import _miner_openrouter_billing
    from harnyx_commons.llm.providers.openrouter_pricing import OpenRouterPricingSnapshot, PricingEndpoint

    snapshot = OpenRouterPricingSnapshot(
        "inception/mercury-3",
        (
            PricingEndpoint.model_validate(
                {
                    "provider_name": "Inception",
                    "tag": "inception",
                    "model_id": "inception/mercury-3",
                    "pricing": {"prompt": "0.1", "completion": "0.35", "discount": 0.5},
                }
            ),
        ),
        "now",
    )
    response = LlmResponse(
        id="ambiguous",
        choices=(),
        usage=LlmUsage(),
        metadata={
            "raw_response": {
                "model": "inception/mercury-3",
                "usage": {"cost": 0.004},
                "openrouter_metadata": {
                    "endpoints": {
                        "available": [
                            {"provider": "Inception", "selected": True},
                            {"provider": "Other", "selected": True},
                        ]
                    }
                },
            }
        },
    )
    billing = _miner_openrouter_billing(response, snapshot=snapshot, requested_model=snapshot.model)
    assert billing.actual_cost_usd == pytest.approx(0.004)
    assert billing.reference_cost_unavailable
    assert billing.reference_cost_usd is None


@pytest.mark.parametrize(
    "cost,byok,upstream,count,aggregate,expected_reference",
    [
        (0.0001234567891234, False, 0.0, 1, 0.000123456789, 0.0002469135782468),
        (0.0001, True, 0.0002, 1, 0.0003, 0.0005),
        (0.004, False, 0.0, 2, 0.008, None),
        (0.0, False, 0.0, 2, 0.0, None),
    ],
)
async def test_reference_settlement_uses_response_count_instead_of_rounded_amount_equality(
    cost: float,
    byok: bool,
    upstream: float,
    count: int,
    aggregate: float,
    expected_reference: float | None,
) -> None:
    """Future failure: rounding rejects one paid response or zero hides multiple responses."""
    from harnyx_commons.llm.providers.openrouter import _miner_openrouter_billing

    snapshot = await _miner_pricing_catalog().snapshot("inception/mercury-3")
    response = LlmResponse(
        id="rounded-charge",
        choices=(),
        usage=LlmUsage(),
        metadata={
            "billable_response_count": count,
            "actual_cost_usd_total": aggregate,
            "raw_response": {
                "model": snapshot.model,
                "provider": "Inception",
                "usage": {"cost": cost, "is_byok": byok, "cost_details": {"upstream_inference_cost": upstream}},
            },
        },
    )
    billing = _miner_openrouter_billing(response, snapshot=snapshot, requested_model=snapshot.model)
    assert billing.actual_cost_usd == pytest.approx(aggregate if count > 1 else cost + (upstream if byok else 0))
    assert billing.reference_cost_unavailable is (expected_reference is None)
    assert billing.reference_cost_usd == (None if expected_reference is None else pytest.approx(expected_reference))


async def test_continuous_usage_does_not_finalize_partial_streamed_tool_arguments() -> None:
    """Future failure: observing billable usage aborts valid incremental function JSON."""
    frames = [
        {
            "id": "streamed-tool",
            "model": "inception/mercury-3",
            "provider": "Inception",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"query":'},
                            }
                        ]
                    },
                }
            ],
            "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
        {
            "id": "streamed-tool",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"harnyx"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"cost": 0.004, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
    delegate = OpenAiCompatibleLlmProvider(
        endpoint=OpenAiCompatibleEndpointConfig.model_validate(
            {"id": "openrouter", "base_url": OPENROUTER_BASE_URL, "auth": {"type": "none"}}
        ),
        client=client,
        response_metadata_extractor=_openrouter_response_metadata,
    )
    provider = OpenRouterLlmProvider(
        openrouter_api_key=SecretStr("test"),
        pricing_catalog=_miner_pricing_catalog(),
        openrouter_chat_provider_factory=lambda _: (delegate, client),
    )
    request = replace(
        _miner_promo_request(),
        tools=(LlmTool(type="function", function={"name": "search", "parameters": {"type": "object"}}),),
    )
    try:
        response = await provider.invoke(request)
        assert response.tool_calls[0].arguments == {"query": "harnyx"}
        assert response.metadata["actual_cost_usd"] == pytest.approx(0.004)
        assert response.metadata["reference_cost_usd"] == pytest.approx(0.008)
    finally:
        await provider.aclose()
