from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from caster_commons.clients import OPENAI
from caster_commons.llm.providers.vertex.provider import VertexLlmProvider
from caster_commons.llm.schema import (
    GroundedLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
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
        timeout=OPENAI.timeout_seconds,
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

    assert response.raw_text == "ok"
    assert response.usage.total_tokens == 17
    tool_calls = response.tool_calls
    assert tool_calls[0].name == "lookup"


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
        timeout=OPENAI.timeout_seconds,
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
