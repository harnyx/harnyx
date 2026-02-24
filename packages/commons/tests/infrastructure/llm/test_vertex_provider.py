from __future__ import annotations

import base64
import json
from typing import Any, cast

import pytest

from caster_commons.clients import CHUTES
from caster_commons.llm.providers.vertex.codec import build_choices, normalize_messages, resolve_thinking_config
from caster_commons.llm.providers.vertex.provider import VertexLlmProvider
from caster_commons.llm.schema import (
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
            args = {"query": "caster"}

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

    def model_dump(self) -> dict[str, Any]:
        return {"text": self.text}


async def test_vertex_provider_invokes_generative_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.models = self._Models()

        class _Models:
            def __init__(self) -> None:
                self.latest: dict[str, Any] = {}

            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                self.latest = {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
                captured["model_call"] = self.latest
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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
    assert http_options.api_version == "v1"
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


async def test_vertex_provider_normalizes_assistant_and_tool_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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

    contents = captured["contents"]
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
    assert message.reasoning == {
        "thought_text_parts": ("deliberation",),
        "has_thought_signature": True,
    }


def test_vertex_codec_build_choices_treats_thought_signature_without_thought_flag_as_reasoning() -> None:
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

    assert tuple(part.text for part in message.content) == ("final answer",)
    assert message.reasoning == {
        "thought_text_parts": ("deliberation",),
        "has_thought_signature": True,
    }


async def test_vertex_provider_routes_claude_models_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {"vertex_calls": 0, "anthropic_calls": 0}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                captured["vertex_calls"] += 1
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

    provider = VertexLlmProvider(
        project="demo-project",
        location="us-central1",
        timeout=CHUTES.timeout_seconds,
    )

    def fake_call_claude(request: Any) -> LlmResponse:
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
    assert captured["vertex_calls"] == 0


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
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.credentials.ServiceAccountCredentials", FakeCredentials)
    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                captured["model"] = model
                captured["config"] = config
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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

    config = captured["config"]
    response_mime_type = config.response_mime_type
    assert response_mime_type is None
    assert config.tools
    tool = config.tools[0]
    assert tool.google_search is not None
    assert config.thinking_config is None


async def test_vertex_provider_includes_provider_native_grounded_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                captured["model"] = model
                captured["config"] = config
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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

    config = captured["config"]
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

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = self._Models()

        class _Models:
            def generate_content(self, *, model: str, contents: Any, config: Any) -> FakeResponse:
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return FakeResponse()

    monkeypatch.setattr("caster_commons.llm.providers.vertex.provider.genai.Client", FakeClient)

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

    contents = captured["contents"]
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
                            arguments='{"query":"caster"}',
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
