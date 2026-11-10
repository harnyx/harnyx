"""Protect native Responses contracts used by internal Luna workflows."""

from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from harnyx_commons.llm.provider import LlmProviderError
from harnyx_commons.llm.providers.openrouter_responses import OpenRouterResponsesLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmTool,
    OpenRouterWebSearchEngine,
)

pytestmark = pytest.mark.anyio("asyncio")


def response(output, status="completed"):
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "model": "openai/gpt-6-luna",
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 5,
            "total_tokens": 17,
            "cost": 0.001,
            "server_tool_use_details": {"web_search_requests": 1},
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def message(text):
    return {
        "type": "message",
        "id": "msg-test",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class Answer(BaseModel):
    result: str


@pytest.mark.parametrize("engine", [None, *OpenRouterWebSearchEngine])
async def test_grounded_discovery_preserves_sources_and_caller_engine_choice(engine):
    cited_url = "https://source.example/cited"
    search_url = "https://source.example/search-result"
    citation_title = "" if engine == OpenRouterWebSearchEngine.EXA else "Cited"
    answer = message("Discovery with a cited source")
    answer["content"][0]["annotations"] = [
        {"type": "url_citation", "url": cited_url, "title": citation_title, "start_index": 0, "end_index": 9}
    ]
    hosted_search = engine == OpenRouterWebSearchEngine.EXA
    action = {"type": "search", "query": "evidence"}
    if hosted_search:
        action["sources"] = [{"type": "url", "url": search_url, "title": "Search result"}]

    def handler(request):
        body = json.loads(request.content)
        expected_tool = {"type": "openrouter:web_search"}
        if engine is not None:
            expected_tool["parameters"] = {"engine": engine}
        assert body["tools"] == [expected_tool]
        # This endpoint rejects unsupported includes even when search is enabled.
        assert set(body["include"]) <= {"reasoning.encrypted_content"}
        return httpx.Response(
            200,
            json=response(
                [
                    {
                        "type": "openrouter:web_search" if hosted_search else "web_search_call",
                        "id": "ws-test",
                        "status": "completed",
                        "action": action,
                    },
                    answer,
                ]
            ),
        )

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenRouterResponsesLlmProvider(client=sdk)
    try:
        result = await provider.invoke(
            GroundedLlmRequest(
                provider="openrouter",
                model="openai/gpt-6-luna",
                temperature=None,
                max_output_tokens=None,
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Find evidence"),)),),
                web_search_engine=engine,
            )
        )
        assert result.raw_text == "Discovery with a cited source"
        assert result.metadata["sources"] == (
            [{"url": search_url, "title": "Search result"}] if hosted_search else []
        ) + [{"url": cited_url, "title": citation_title or cited_url}]
    finally:
        await provider.aclose()


async def test_structured_judge_uses_native_responses_and_records_usage():
    bodies = []

    def handler(request):
        assert request.url.host == "openrouter.ai"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.path == "/api/v1/responses"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=response([message('{"result":"ok"}')]))

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenRouterResponsesLlmProvider(client=sdk)
    try:
        result = await provider.invoke(
            LlmRequest(
                provider="openrouter",
                model="openai/gpt-6-luna",
                temperature=None,
                max_output_tokens=None,
                messages=(
                    LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Judge the JSON result"),)),
                ),
                reasoning_effort="high",
                output_mode="structured",
                output_schema=Answer,
            )
        )
        assert result.raw_text == '{"result":"ok"}'
        assert result.finish_reason == "stop" and result.usage.prompt_tokens == 12
        assert result.usage.reasoning_tokens == 1
        assert bodies[0]["reasoning"]["effort"] == "high" and bodies[0]["store"] is False
        assert bodies[0]["text"]["format"]["schema"]["additionalProperties"] is False
    finally:
        await provider.aclose()
    assert sdk.is_closed()


async def test_context_search_keeps_auth_host_side_and_replays_native_reasoning():
    bodies, callbacks = [], []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json=response(
                    [
                        {
                            "type": "openrouter:web_search",
                            "id": "ws-test",
                            "status": "completed",
                            "action": {"type": "search", "query": "prior post"},
                        },
                        {"type": "reasoning", "id": "rs-test", "summary": [], "encrypted_content": "opaque-state"},
                        {
                            "type": "function_call",
                            "id": "fc-test",
                            "call_id": "call-test",
                            "name": "search_context",
                            "arguments": '{"query":"earlier post"}',
                            "status": "completed",
                        },
                    ]
                ),
            )
        return httpx.Response(200, json=response([message("Retrieved evidence")]))

    def retrieve(request):
        callbacks.append(request)
        assert request.headers["X-Api-Key"] == "context-secret"
        assert json.loads(request.content) == {"query": "earlier post"}
        return httpx.Response(200, json=[{"snippet": "Prior content", "uri": "https://example.com/post"}])

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenRouterResponsesLlmProvider(
        client=sdk, retrieval_client=httpx.AsyncClient(transport=httpx.MockTransport(retrieve))
    )
    try:
        result = await provider.invoke(
            GroundedLlmRequest(
                provider="openrouter",
                model="openai/gpt-6-luna",
                temperature=None,
                max_output_tokens=None,
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Find prior content"),)),),
                tools=(
                    LlmTool(
                        type="provider_native",
                        config={
                            "retrieval": {
                                "external_api": {
                                    "api_spec": "SIMPLE_SEARCH",
                                    "endpoint": "https://platform.example/context",
                                    "auth_config": {
                                        "api_key_config": {"name": "X-Api-Key", "api_key_string": "context-secret"}
                                    },
                                }
                            }
                        },
                    ),
                ),
                timeout=10,
            )
        )
        assert result.raw_text == "Retrieved evidence" and result.usage.prompt_tokens == 24
        assert result.usage.web_search_calls == 2
        assert result.metadata["billable_response_count"] == 2
        assert result.metadata["actual_cost_usd"] == 0.002
        assert result.metadata["actual_cost_provider"] == "openrouter"
        assert result.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"
        assert len(callbacks) == 1
        assert "context-secret" not in json.dumps(bodies)
        assert any(item.get("encrypted_content") == "opaque-state" for item in bodies[1]["input"])
        assert any(
            item.get("type") == "function_call_output" and item["call_id"] == "call-test" for item in bodies[1]["input"]
        )
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "arguments",
    ["not json", "[]", "{}", '{"query":null}', '{"query":17}', '{"query":"valid","extra":true}'],
)
async def test_invalid_search_arguments_cannot_dispatch_retrieval_and_keep_usage(arguments):
    requests, callbacks = [], []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=response(
                [{"type": "function_call", "call_id": "search", "name": "search_context", "arguments": arguments}]
            ),
        )

    def retrieve(request):
        callbacks.append(request)
        raise AssertionError("Invalid search arguments reached retrieval")

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenRouterResponsesLlmProvider(
        client=sdk, retrieval_client=httpx.AsyncClient(transport=httpx.MockTransport(retrieve))
    )
    try:
        with pytest.raises(LlmProviderError) as failure:
            await provider.invoke(
                GroundedLlmRequest(
                    provider="openrouter",
                    model="openai/gpt-6-luna",
                    temperature=None,
                    max_output_tokens=None,
                    messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Search"),)),),
                    tools=(
                        LlmTool(
                            type="provider_native",
                            config={
                                "retrieval": {
                                    "external_api": {
                                        "api_spec": "SIMPLE_SEARCH",
                                        "endpoint": "https://platform.example/context",
                                        "auth_config": {
                                            "api_key_config": {"name": "X-Api-Key", "api_key_string": "secret"}
                                        },
                                    }
                                }
                            },
                        ),
                    ),
                    timeout=10,
                )
            )
        assert len(requests) == 1 and not callbacks
        assert failure.value.response is not None
        assert failure.value.response.usage.prompt_tokens == 12
        assert failure.value.response.metadata["billable_response_count"] == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_incomplete_outputs_cannot_be_accepted_as_answers(status):
    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=response([message("Partial")], status)),
            )
        ),
    )
    provider = OpenRouterResponsesLlmProvider(client=sdk)
    try:
        with pytest.raises(LlmProviderError):
            await provider.invoke(
                LlmRequest(
                    provider="openrouter",
                    model="openai/gpt-6-luna",
                    temperature=None,
                    max_output_tokens=None,
                    messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Answer"),)),),
                    retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
                )
            )
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    ("code", "error_type"),
    [("server_error", None), ("server_error", "provider_overloaded"), ("permission_denied", "permission_denied")],
)
async def test_failed_response_reports_upstream_error_and_preserves_billing_without_retry(code, error_type):
    payload = response([], "failed")
    payload["error"] = {
        "code": code,
        "message": "Upstream provider is overloaded",
        "metadata": {"provider_name": "OpenAI"},
    }
    if error_type is not None:
        payload["error_type"] = error_type
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider = OpenRouterResponsesLlmProvider(client=sdk)
    try:
        with pytest.raises(LlmProviderError) as failure:
            await provider.invoke(
                LlmRequest(
                    provider="openrouter",
                    model="openai/gpt-6-luna",
                    temperature=None,
                    max_output_tokens=None,
                    messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Answer"),)),),
                    retry_policy=RetryPolicy(attempts=3, initial_ms=0, max_ms=0, jitter=0),
                )
            )
        assert str(failure.value) == (
            f"OpenRouter response failed: {error_type or code}: Upstream provider is overloaded"
        )
        assert len(requests) == 1
        result = failure.value.response
        assert result is not None
        assert result.metadata["raw_response"]["error"] == payload["error"]
        assert result.metadata["billable_response_count"] == 1
        assert result.metadata["actual_cost_usd_total"] == 0.001
        assert result.usage.prompt_tokens == 12
    finally:
        await provider.aclose()


async def test_missing_openrouter_cost_receipt_preserves_usage_without_claiming_actual_cost():
    payload = response([message("Complete answer")])
    payload["usage"].pop("cost")
    sdk = AsyncOpenAI(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
        ),
    )
    provider = OpenRouterResponsesLlmProvider(client=sdk)
    try:
        result = await provider.invoke(
            LlmRequest(
                provider="openrouter",
                model="openai/gpt-6-luna",
                temperature=None,
                max_output_tokens=None,
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Answer"),)),),
            )
        )
        assert result.raw_text == "Complete answer" and result.usage.prompt_tokens == 12
        assert result.metadata.get("actual_cost_usd") is None
        assert result.metadata["cost_incomplete_reason"] == "missing_openrouter_billing_receipt"
    finally:
        await provider.aclose()
