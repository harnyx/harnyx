from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

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


def _start_openai_compatible_server(seen_payloads: list[dict[str, object]]) -> tuple[uvicorn.Server, str]:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> StreamingResponse:
        payload = await request.json()
        seen_payloads.append(dict(payload))
        return StreamingResponse(_sse_chunks(), media_type="text/event-stream")

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
