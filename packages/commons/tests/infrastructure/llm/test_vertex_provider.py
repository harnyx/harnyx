from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import BaseModel

from harnyx_commons.clients import CHUTES
from harnyx_commons.llm.provider_types import normalize_reasoning_effort
from harnyx_commons.llm.providers.vertex.codec import (
    build_choices,
    build_vertex_maas_chat_payload,
    normalize_messages,
    parse_vertex_maas_chat_response,
    resolve_thinking_config,
    vertex_maas_openai_chat_model_name,
)
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider
from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmRequest,
    LlmResponse,
    LlmTool,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")


class FakeUsage:
    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_token_count = prompt
        self.cached_content_token_count = None
        self.candidates_token_count = completion
        self.thoughts_token_count = None
        self.total_token_count = total


class FakeResponse:
    def __init__(self) -> None:
        self.text = "ok"
        self.response_id = "fake-response-id"
        self.usage_metadata = FakeUsage(12, 5, 17)
        self.candidates = [self._candidate()]

    @staticmethod
    def _candidate() -> Any:
        class _FunctionCall:
            id = None
            name = "lookup"
            args = {"query": "harnyx"}

        class _Part:
            text = "ok"
            function_call = _FunctionCall()
            thought = False
            thought_signature = None

        class _Content:
            parts = [_Part()]

        class _Candidate:
            content = _Content()
            finish_reason = None
            grounding_metadata = None

        return _Candidate()

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"text": self.text}


@pytest.fixture(autouse=True)
def anthropic_clients(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    created: list[Any] = []

    class _FailingMessages:
        async def create(self, **kwargs: Any) -> Any:
            raise AssertionError(f"unexpected AsyncAnthropicVertex.messages.create call: {kwargs!r}")

    class _FakeAsyncAnthropicVertex:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.messages = _FailingMessages()
            self.closed = False
            created.append(self)

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "harnyx_commons.llm.providers.vertex.provider.AsyncAnthropicVertex",
        _FakeAsyncAnthropicVertex,
    )
    return created


def _patch_google_client(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    response_factory: Callable[[], Any] = FakeResponse,
) -> None:
    class _FakeAsyncModels:
        async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            latest = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            captured["model_call"] = latest
            return response_factory()

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.models = _FakeAsyncModels()

        async def aclose(self) -> None:
            captured["google_async_closed"] = True

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = _FakeAsyncClient()

        def close(self) -> None:
            captured["google_sync_closed"] = True

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.genai.Client", _FakeClient)


def _patch_vertex_maas_http_client(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    *,
    response_payload: dict[str, Any] | None = None,
) -> None:
    payload = response_payload or {
        "id": "chatcmpl-vertex",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "56",
                    "reasoning_content": "I need to multiply 7 by 8.",
                    "tool_calls": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "reasoning_tokens": 5,
            "total_tokens": 15,
        },
    }

    class _FakeHttpResponse:
        def __init__(self, json_payload: dict[str, Any]) -> None:
            self._json_payload = json_payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._json_payload

    class _FakeAsyncHttpClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["http_client_kwargs"] = kwargs

        async def post(self, url: str, **kwargs: Any) -> _FakeHttpResponse:
            captured["http_call"] = {"url": url, **kwargs}
            return _FakeHttpResponse(payload)

        async def aclose(self) -> None:
            captured["http_closed"] = True

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.httpx.AsyncClient", _FakeAsyncHttpClient)


def _async_return(value: Any) -> Callable[[], Any]:
    async def _inner() -> Any:
        return value

    return _inner


async def test_vertex_provider_invokes_generative_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("stay concise"),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hi"),),
            ),
        ),
        temperature=0.3,
        max_output_tokens=256,
        output_mode="json_object",
        tools=(
            LlmTool(
                type="function",
                function={
                    "name": "lookup",
                    "description": "Lookup info",
                    "parameters": {"type": "object", "properties": {}},
                },
            ),
        ),
        tool_choice="required",
    )

    response = await provider.invoke(request)

    client_kwargs = captured["client_kwargs"]
    assert client_kwargs["project"] == "demo-project"
    assert client_kwargs["location"] == "us-central1"
    http_options = client_kwargs["http_options"]
    assert http_options.api_version == "v1beta1"
    assert client_kwargs["credentials"] is None

    model_call = captured["model_call"]
    assert model_call["model"] == "publishers/openai/models/gpt-oss-20b-maas"
    assert model_call["contents"][0].role == "user"
    config = model_call["config"]
    assert config.system_instruction == "stay concise"
    assert config.temperature == pytest.approx(0.3)
    assert config.max_output_tokens == 256
    assert config.response_mime_type == "application/json"
    assert config.tools and len(config.tools) == 1
    assert config.tool_config.function_calling_config.mode.name == "ANY"
    assert config.thinking_config is None

    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 17
    tool_calls = response.tool_calls
    assert tool_calls[0].name == "lookup"
    assert response.metadata is not None
    raw_response = response.metadata["raw_response"]
    assert isinstance(raw_response, dict)
    assert raw_response["text"] == "ok"


async def test_vertex_maas_gpt_oss_routes_to_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )
    monkeypatch.setattr(provider, "_vertex_maas_access_token", _async_return("access-token"))

    request = LlmRequest(
        provider="vertex-maas",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("stay concise"),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("What is 7 times 8?"),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
        reasoning_effort="high",
    )

    response = await provider.invoke(request)

    assert "model_call" not in captured
    http_call = captured["http_call"]
    assert http_call["url"].endswith("/endpoints/openapi/chat/completions")
    assert http_call["headers"]["Authorization"] == "Bearer access-token"
    payload = http_call["json"]
    assert payload["model"] == "openai/gpt-oss-120b-maas"
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 64
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert response.raw_text == "56"
    assert response.choices[0].message.reasoning == "I need to multiply 7 by 8."
    assert response.usage.reasoning_tokens == 5


async def test_vertex_provider_keeps_vertex_gpt_oss_on_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )
    monkeypatch.setattr(provider, "_vertex_maas_access_token", _async_return("access-token"))

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=32,
    )

    await provider.invoke(request)

    assert "model_call" in captured
    assert "http_call" not in captured


async def test_vertex_provider_raw_response_metadata_is_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    thought_signature = "ZmFrZS10aG91Z2h0LXNpZw=="

    class _RawResponseWithThoughtSignature(FakeResponse):
        def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
            if mode == "json":
                return {
                    "text": self.text,
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "thought_signature": thought_signature,
                                    }
                                ]
                            }
                        }
                    ],
                }
            return {
                "text": self.text,
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "thought_signature": b"\xff\xfe",
                                }
                            ]
                        }
                    }
                ],
            }

    _patch_google_client(
        monkeypatch,
        {},
        response_factory=_RawResponseWithThoughtSignature,
    )

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="gemini-2.5-pro",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        reasoning_effort="high",
    )

    response = await provider.invoke(request)

    assert response.metadata is not None
    raw_response = response.metadata["raw_response"]
    assert isinstance(raw_response, dict)
    signature_value = raw_response["candidates"][0]["content"]["parts"][0]["thought_signature"]
    assert isinstance(signature_value, str)
    assert signature_value == thought_signature
    payload_signature = response.payload["metadata"]["raw_response"]["candidates"][0]["content"]["parts"][0][
        "thought_signature"
    ]
    assert payload_signature == thought_signature


async def test_vertex_provider_normalizes_assistant_and_tool_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("user-turn"),),
            ),
            LlmMessage(
                role="assistant",
                content=(LlmMessageContentPart.input_text("assistant-turn"),),
            ),
            LlmMessage(
                role="tool",
                content=(LlmMessageContentPart.input_text("tool-turn"),),
            ),
        ),
        temperature=None,
        max_output_tokens=128,
        output_mode="text",
    )

    await provider.invoke(request)

    contents = captured["model_call"]["contents"]
    assert [entry.role for entry in contents] == ["user", "model", "user"]


def test_vertex_codec_fails_fast_on_unknown_request_role() -> None:
    with pytest.raises(ValueError, match="unsupported Vertex request role: 'critic'"):
        normalize_messages(
            (
                LlmMessage(
                    role=cast(Any, "critic"),
                    content=(LlmMessageContentPart.input_text("invalid-role"),),
                ),
            )
        )


def test_vertex_resolve_thinking_config_returns_none_when_effort_is_null() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort=None)
    assert config is None


def test_vertex_resolve_thinking_config_returns_none_when_effort_is_zero() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort="0")
    assert config is None


def test_normalize_reasoning_effort_rejects_non_positive_budgets() -> None:
    assert normalize_reasoning_effort(None) is None
    assert normalize_reasoning_effort("  ") is None
    assert normalize_reasoning_effort("0") is None
    assert normalize_reasoning_effort("-1") is None
    assert normalize_reasoning_effort("high") == "high"
    assert normalize_reasoning_effort(" 256 ") == "256"


def test_vertex_resolve_thinking_config_sets_level_with_include_thoughts() -> None:
    config = resolve_thinking_config(model="gemini-2.5-pro", reasoning_effort="high")
    assert config is not None
    assert config.include_thoughts is True
    assert config.thinking_level is not None


def test_vertex_codec_build_choices_separates_thought_text_from_assistant_output() -> None:
    class _ThoughtPart:
        text = "deliberation"
        function_call = None
        thought = True
        thought_signature = "sig-1"

    class _AssistantPart:
        text = "final answer"
        function_call = None
        thought = False
        thought_signature = None

    class _Content:
        parts = [_ThoughtPart(), _AssistantPart()]

    class _Candidate:
        content = _Content()
        finish_reason = None
        grounding_metadata = None

    class _Response:
        candidates = [_Candidate()]

    choices = build_choices(_Response())
    message = choices[0].message

    assert tuple(part.text for part in message.content) == ("final answer",)
    assert message.reasoning == "deliberation"


def test_vertex_codec_build_choices_preserves_signature_only_text_as_output() -> None:
    class _ThoughtPart:
        text = "deliberation"
        function_call = None
        thought = False
        thought_signature = "sig-2"

    class _AssistantPart:
        text = "final answer"
        function_call = None
        thought = False
        thought_signature = None

    class _Content:
        parts = [_ThoughtPart(), _AssistantPart()]

    class _Candidate:
        content = _Content()
        finish_reason = None
        grounding_metadata = None

    class _Response:
        candidates = [_Candidate()]

    choices = build_choices(_Response())
    message = choices[0].message

    assert tuple(part.text for part in message.content) == ("deliberation", "final answer")
    assert message.reasoning is None


class _StructuredPairwisePreference(BaseModel):
    preferred_position: str


def test_vertex_maas_chat_payload_supports_structured_output() -> None:
    request = LlmRequest(
        provider="vertex-maas",
        model="publishers/openai/models/gpt-oss-120b-maas",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("Return JSON."),),
            ),
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("Choose first or second."),),
            ),
        ),
        output_mode="structured",
        output_schema=_StructuredPairwisePreference,
        temperature=None,
        max_output_tokens=64,
        reasoning_effort="high",
    )

    payload = build_vertex_maas_chat_payload(request)

    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "_StructuredPairwisePreference"
    assert payload["reasoning_effort"] == "high"
    assert "temperature" not in payload


def test_parse_vertex_maas_chat_response_maps_reasoning_tool_calls_and_usage() -> None:
    payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": '{"preferred_position":"first"}',
                        }
                    ],
                    "reasoning_content": [{"text": "I should prefer the first answer."}],
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "paris"},
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 7,
            "reasoning_tokens": 3,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }

    response = parse_vertex_maas_chat_response(payload)

    assert response.id == "chatcmpl-123"
    assert response.raw_text == '{"preferred_position":"first"}'
    assert response.choices[0].message.reasoning == "I should prefer the first answer."
    assert response.choices[0].message.tool_calls[0].arguments == '{"query": "paris"}'
    assert response.tool_calls[0].name == "lookup"
    assert response.tool_calls[0].arguments == {"query": "paris"}
    assert response.usage.prompt_tokens == 14
    assert response.usage.prompt_cached_tokens == 2
    assert response.usage.completion_tokens == 7
    assert response.usage.reasoning_tokens == 3
    assert response.usage.total_tokens == 24


def test_vertex_maas_openai_chat_model_name_strips_publisher_prefix() -> None:
    assert (
        vertex_maas_openai_chat_model_name("publishers/openai/models/gpt-oss-120b-maas")
        == "openai/gpt-oss-120b-maas"
    )
    assert vertex_maas_openai_chat_model_name("openai/gpt-oss-120b-maas") == "openai/gpt-oss-120b-maas"


def test_vertex_verify_response_still_rejects_reasoning_only_output() -> None:
    response = LlmResponse(
        id="reasoning-only",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(),
                    tool_calls=None,
                    reasoning="I reasoned but produced no final answer.",
                ),
                finish_reason="stop",
            ),
        ),
        usage=LlmUsage(reasoning_tokens=5),
        finish_reason="stop",
    )

    assert VertexLlmProvider._verify_response(response) == (False, True, "empty_output")


async def test_vertex_provider_routes_claude_models_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {"vertex_calls": 0, "anthropic_calls": 0}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    async def fake_call_claude(request: Any) -> LlmResponse:
        captured["anthropic_calls"] += 1
        captured["anthropic_model"] = request.model
        return LlmResponse(
            id="claude-response",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(),
            finish_reason="stop",
        )

    monkeypatch.setattr(provider, "_call_claude_anthropic", fake_call_claude)

    request = LlmRequest(
        provider="vertex",
        model="/anthropic/models/claude-sonnet-4-5@20250929",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
    )

    response = await provider.invoke(request)
    assert response.raw_text == "ok"

    assert captured["anthropic_calls"] == 1
    assert "model_call" not in captured


async def test_vertex_provider_aclose_closes_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_clients: list[Any],
) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)
    _patch_vertex_maas_http_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    await provider.aclose()

    assert captured["google_async_closed"] is True
    assert captured["google_sync_closed"] is True
    assert captured["http_closed"] is True
    assert len(anthropic_clients) == 1
    assert anthropic_clients[0].closed is True


def test_vertex_provider_writes_base64_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}

    class FakeCredentials:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        @classmethod
        def from_service_account_info(
            cls,
            info: dict[str, Any],
            scopes: tuple[str, ...],
        ) -> FakeCredentials:
            captured["creds_info"] = info
            captured["creds_scopes"] = scopes
            return cls("info")

        @classmethod
        def from_service_account_file(
            cls,
            path: str,
            scopes: tuple[str, ...],
        ) -> FakeCredentials:
            captured["creds_path"] = path
            captured["creds_scopes"] = scopes
            return cls("file")

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.aio = self._AsyncClient()

        def close(self) -> None:
            captured["google_sync_closed"] = True

        class _AsyncClient:
            def __init__(self) -> None:
                self.models = self._Models()

            async def aclose(self) -> None:
                return None

            class _Models:
                async def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                    return FakeResponse()

    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.credentials.ServiceAccountCredentials", FakeCredentials)
    monkeypatch.setattr("harnyx_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

    service_account_payload = json.dumps({
        "type": "service_account",
        "client_email": "vertex@test-project.iam.gserviceaccount.com",
        "private_key_id": "abc123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n",
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    encoded = base64.b64encode(service_account_payload.encode()).decode()

    VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
        service_account_b64=encoded,
    )


async def test_vertex_provider_injects_google_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.0",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("summarize"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
    )

    await provider.invoke(request)

    config = captured["model_call"]["config"]
    response_mime_type = config.response_mime_type
    assert response_mime_type is None
    assert config.tools
    tool = config.tools[0]
    assert tool.google_search is not None
    assert config.thinking_config is None


async def test_vertex_provider_includes_provider_native_grounded_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.0",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("check novelty"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        tools=(
            LlmTool(
                type="provider_native",
                config={
                    "retrieval": {
                        "external_api": {
                            "api_spec": "ELASTIC_SEARCH",
                            "endpoint": "https://elastic.example.com",
                            "api_auth": {"api_key_config": {"api_key_string": "ApiKey test"}},
                            "elastic_search_params": {
                                "index": "feed-eval-alias",
                                "search_template": "feed_eval_hybrid_v1",
                                "num_hits": 20,
                            },
                        }
                    }
                },
            ),
        ),
    )

    await provider.invoke(request)

    config = captured["model_call"]["config"]
    assert config.tools
    assert len(config.tools) == 2
    assert config.tools[0].google_search is not None
    retrieval = config.tools[1].retrieval
    assert retrieval is not None
    external_api = retrieval.external_api
    assert external_api is not None
    assert external_api.endpoint == "https://elastic.example.com"
    assert external_api.elastic_search_params is not None
    assert external_api.elastic_search_params.search_template == "feed_eval_hybrid_v1"


async def test_vertex_serializes_input_tool_result_as_function_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}
    _patch_google_client(monkeypatch, captured)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=30.0,
    )

    request = LlmRequest(
        provider="vertex",
        model="publishers/openai/models/gpt-oss-20b-maas",
        messages=(
            LlmMessage(
                role="user",
                content=(
                    LlmInputToolResultPart(
                        tool_call_id="call-1",
                        name="search_repo",
                        output_json=json.dumps({"data": [{"path": "README.md"}]}),
                    ),
                ),
            ),
        ),
        temperature=None,
        max_output_tokens=128,
        output_mode="text",
    )

    await provider.invoke(request)

    contents = captured["model_call"]["contents"]
    assert contents
    part = contents[0].parts[0]
    function_response = part.function_response
    assert function_response is not None
    assert function_response.name == "search_repo"
    assert function_response.response["tool_call_id"] == "call-1"
    assert function_response.response["data"][0]["path"] == "README.md"


def test_vertex_verify_accepts_tool_call_only_choice() -> None:
    response = LlmResponse(
        id="resp-tool-call-only",
        choices=(
            LlmChoice(
                index=0,
                message=LlmChoiceMessage(
                    role="assistant",
                    content=(LlmMessageContentPart(type="text", text=""),),
                    tool_calls=(
                        LlmMessageToolCall(
                            id="tc-1",
                            type="function",
                            name="search_repo",
                            arguments='{"query":"harnyx"}',
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=LlmUsage(),
        finish_reason="tool_calls",
    )

    ok, retryable, reason = VertexLlmProvider._verify_response(response)
    assert ok is True
    assert retryable is False
    assert reason is None
