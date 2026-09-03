from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings, OpenAiCompatibleEndpointConfig
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.providers.openrouter import OpenRouterLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_commons.llm.timeout import LlmAttemptTimeoutError
from harnyx_miner_sdk.llm import Timeout

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


@pytest.mark.parametrize("stream_options", (None, {}, {"include_usage": False, "continuous_usage_stats": False}))
async def test_custom_openai_compatible_provider_contract_against_local_server(
    stream_options: dict[str, bool] | None,
) -> None:
    seen_payloads: list[dict[str, object]] = []
    server, base_url = _start_openai_compatible_server(seen_payloads)
    settings = LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=(
            f'[{{"id":"gemma4-cloud-run-turbo","base_url":"{base_url}/v1","auth":{{"type":"none"}}}}]'
        ),
        LLM_MODEL_PROVIDER_OVERRIDES_JSON=(
            '{"tool":{"google/gemma-4-31B-turbo-TEE":"custom-openai-compatible:gemma4-cloud-run-turbo"}}'
        ),
    )
    registry = build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64="",
        ),
    )
    provider = build_routed_llm_provider(
        surface="tool",
        default_provider="chutes",
        llm_settings=settings,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )

    try:
        response = await provider.invoke(
            LlmRequest(
                provider="chutes",
                model="google/gemma-4-31B-turbo-TEE",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
                    ),
                ),
                temperature=0.0,
                max_output_tokens=16,
                extra={"stream_options": stream_options},
            )
        )
    finally:
        await registry.aclose()
        server.should_exit = True

    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "custom-openai-compatible:gemma4-cloud-run-turbo"
    assert response.metadata["effective_model"] == "google/gemma-4-31B-turbo-TEE"
    assert seen_payloads
    assert seen_payloads[0]["model"] == "nvidia/Gemma-4-31B-IT-NVFP4"
    assert seen_payloads[0]["stream"] is True
    assert seen_payloads[0]["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


async def test_qwen36_custom_openai_compatible_provider_contract_against_local_server() -> None:
    seen_payloads: list[dict[str, object]] = []
    server, base_url = _start_openai_compatible_server(seen_payloads)
    settings = LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=(
            f'[{{"id":"qwen36-cloud-run","base_url":"{base_url}/v1","auth":{{"type":"none"}}}}]'
        ),
        LLM_MODEL_PROVIDER_OVERRIDES_JSON=(
            '{"tool":{"Qwen/Qwen3.6-27B-TEE":"custom-openai-compatible:qwen36-cloud-run"}}'
        ),
    )
    registry = build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64="",
        ),
    )
    provider = build_routed_llm_provider(
        surface="tool",
        default_provider="chutes",
        llm_settings=settings,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )

    try:
        response = await provider.invoke(
            LlmRequest(
                provider="chutes",
                model="Qwen/Qwen3.6-27B-TEE",
                messages=(
                    LlmMessage(
                        role="user",
                        content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
                    ),
                ),
                temperature=0.0,
                max_output_tokens=16,
            )
        )
    finally:
        await registry.aclose()
        server.should_exit = True

    assert response.raw_text == "ok"
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "custom-openai-compatible:qwen36-cloud-run"
    assert response.metadata["effective_model"] == "Qwen/Qwen3.6-27B-TEE"
    assert seen_payloads
    assert seen_payloads[0]["model"] == "Qwen/Qwen3.6-27B-FP8"
    assert seen_payloads[0]["stream"] is True
    assert seen_payloads[0]["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


@pytest.mark.parametrize("provider_name", ["chutes", "custom", "openrouter"])
@pytest.mark.parametrize("scenario", ["prefill", "inactivity", "total", "slow_prefill"])
async def test_attempt_deadlines_close_real_http_streams(provider_name: str, scenario: str) -> None:
    await _assert_http_stream_deadline(provider_name, scenario)


@pytest.mark.parametrize("provider_name", ["custom", "openrouter"])
@pytest.mark.parametrize("scenario", ["encrypted_total", "encrypted_inactivity"])
async def test_encrypted_reasoning_controls_real_http_deadlines(provider_name: str, scenario: str) -> None:
    await _assert_http_stream_deadline(provider_name, scenario)


async def _assert_http_stream_deadline(provider_name: str, scenario: str) -> None:
    disconnected = threading.Event()
    server, base_url = _start_openai_compatible_server([], scenario=scenario, disconnected=disconnected)
    client = httpx.AsyncClient(base_url=base_url)
    if provider_name == "chutes":
        provider = ChutesLlmProvider(base_url=base_url, api_key="test-key", client=client)
    else:
        endpoint = OpenAiCompatibleEndpointConfig.model_validate(
            {"id": "local-timeout", "base_url": base_url + "/v1", "auth": {"type": "none"}}
        )
        delegate = OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client)
        provider = (
            delegate
            if provider_name == "custom"
            else OpenRouterLlmProvider(
                openrouter_api_key=SecretStr("test-key"),
                openrouter_chat_provider_factory=lambda model: (delegate, client),
            )
        )
    policy = (
        Timeout(1, prefill=0.5, inactivity=0.05)
        if scenario == "slow_prefill"
        else Timeout(0.4, prefill=0.15, inactivity=0.15)
    )
    request = LlmRequest(
        provider="chutes",
        model="openai/gpt-oss-20b",
        messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("hello"),)),),
        temperature=None,
        max_output_tokens=None,
        timeout=policy,
        retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
    )
    try:
        if scenario == "slow_prefill":
            response = await provider.invoke(request)
            assert response.raw_text == "ok"
        else:
            with pytest.raises(LlmRetryExhaustedError) as exc:
                await provider.invoke(request)
            assert isinstance(exc.value.__cause__, LlmAttemptTimeoutError)
            assert exc.value.__cause__.phase == scenario.removeprefix("encrypted_")
        assert await asyncio.to_thread(disconnected.wait, 2)
    finally:
        await provider.aclose()
        await client.aclose()
        server.should_exit = True


def _start_openai_compatible_server(
    seen_payloads: list[dict[str, object]],
    *,
    scenario: str | None = None,
    disconnected: threading.Event | None = None,
) -> tuple[uvicorn.Server, str]:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> StreamingResponse:
        payload = await request.json()
        seen_payloads.append(dict(payload))
        chunks = _deadline_chunks(scenario, disconnected) if scenario else _sse_chunks()
        return StreamingResponse(chunks, media_type="text/event-stream")

    port = _find_free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            log_config=None,
            timeout_graceful_shutdown=1,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("OpenAI-compatible test server exited before startup")
        if time.monotonic() >= deadline:
            raise RuntimeError("OpenAI-compatible test server did not start")
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


async def _deadline_chunks(scenario: str, disconnected: threading.Event | None) -> AsyncIterator[str]:
    try:
        if scenario == "slow_prefill":
            await asyncio.sleep(0.15)
            for chunk in _sse_chunks():
                yield chunk
            return
        if scenario in {"encrypted_total", "encrypted_inactivity"}:
            yield (
                'data: {"choices":[{"delta":{"reasoning_details":'
                '[{"type":"reasoning.encrypted","data":"opaque"}]}}]}\n\n'
            )
            while True:
                data = "opaque" if scenario == "encrypted_total" else ""
                yield (
                    'data: {"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.encrypted","data":"'
                    + data
                    + '"}]}}]}\n\n'
                )
                await asyncio.sleep(0.02)
        if scenario != "prefill":
            yield 'data: {"choices":[{"index":0,"delta":{"reasoning":"thinking"}}]}\n\n'
        while True:
            if scenario == "total":
                yield 'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
            else:
                yield ': heartbeat\n\ndata: {"choices":[{"delta":{"reasoning_details":[]}}]}\n\n'
            await asyncio.sleep(0.02)
    finally:
        if disconnected is not None:
            disconnected.set()


def _sse_chunks() -> Iterator[str]:
    yield 'data: {"id":"chatcmpl-local","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
    yield (
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
    )
    yield "data: [DONE]\n\n"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
