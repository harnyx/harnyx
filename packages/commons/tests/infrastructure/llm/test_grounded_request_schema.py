from __future__ import annotations

import pytest

from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmTool,
    OpenRouterWebSearchEngine,
    supports_grounded_additional_tools,
    supports_grounded_requests,
)


@pytest.mark.parametrize("provider,grounded", [("vertex", True), ("openrouter", False)])
def test_search_engine_cannot_be_silently_ignored_by_an_inapplicable_request(provider, grounded):
    request_type = GroundedLlmRequest if grounded else LlmRequest
    with pytest.raises(ValueError, match="web_search_engine requires an OpenRouter grounded request"):
        request_type(
            provider=provider,
            model="gemini-2.5-flash" if provider == "vertex" else "openai/gpt-6-luna",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            web_search_engine=OpenRouterWebSearchEngine.EXA,
        )


def _tools() -> tuple[LlmTool, ...]:
    return (
        LlmTool(
            type="provider_native",
            config={"retrieval": {"external_api": {"api_spec": "ELASTIC_SEARCH"}}},
        ),
    )


def _messages() -> tuple[LlmMessage, ...]:
    return (
        LlmMessage(
            role="user",
            content=(LlmMessageContentPart.input_text("hello"),),
        ),
    )


def test_grounded_chutes_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="grounded mode not supported for provider/model 'chutes:openai/gpt-oss-20b-TEE'",
    ):
        GroundedLlmRequest(
            provider="chutes",
            model="openai/gpt-oss-20b-TEE",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=_tools(),
        )


def test_grounded_vertex_gemini_allows_additional_tools() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.5-flash",
        messages=_messages(),
        temperature=None,
        max_output_tokens=None,
        tools=_tools(),
    )
    assert request.tools is not None


def test_grounded_vertex_gemini_publisher_path_allows_additional_tools() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="publishers/google/models/gemini-2.5-flash",
        messages=_messages(),
        temperature=None,
        max_output_tokens=None,
        tools=_tools(),
    )
    assert request.tools is not None


def test_grounded_vertex_full_resource_gemini_allows_additional_tools() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="projects/test/locations/us-central1/publishers/google/models/gemini-2.5-flash",
        messages=_messages(),
        temperature=None,
        max_output_tokens=None,
        tools=_tools(),
    )
    assert request.tools is not None


def test_grounded_vertex_claude_rejects_additional_tools() -> None:
    with pytest.raises(
        ValueError,
        match="grounded requests with additional tools are not supported for Vertex Claude models",
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="publishers/anthropic/models/claude-sonnet-4-5@20250929",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=_tools(),
        )


def test_grounded_vertex_regular_claude_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="grounded mode not supported for provider/model 'vertex:claude-3-7-sonnet'",
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="claude-3-7-sonnet",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=(),
        )


def test_grounded_vertex_claude_web_search_model_allowed_without_additional_tools() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="publishers/anthropic/models/claude-sonnet-4-5@20250929",
        messages=_messages(),
        temperature=None,
        max_output_tokens=None,
        tools=(),
    )
    assert request.grounded is True


def test_grounded_vertex_invalid_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="grounded mode not supported for provider/model 'vertex:not-a-real-model'",
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="not-a-real-model",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=(),
        )


def test_grounded_vertex_typoed_gemini_like_model_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="grounded mode not supported for provider/model 'vertex:foo/gemini-typo'",
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="foo/gemini-typo",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=(),
        )


def test_supports_grounded_additional_tools_matches_grounded_request_support_for_invalid_gemini_like_model() -> None:
    assert supports_grounded_requests(provider="vertex", model="foo/gemini-typo") is False
    assert supports_grounded_additional_tools(provider="vertex", model="foo/gemini-typo") is False


def test_grounded_vertex_malformed_gemini_publisher_path_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("grounded mode not supported for provider/model 'vertex:foo/publishers/google/models/gemini-2.5-flash'"),
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="foo/publishers/google/models/gemini-2.5-flash",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=(),
        )
