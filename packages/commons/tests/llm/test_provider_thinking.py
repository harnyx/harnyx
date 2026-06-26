from __future__ import annotations

from harnyx_commons.llm.providers.thinking import resolve_template_thinking
from harnyx_commons.llm.schema import LlmThinkingConfig


def test_resolve_template_thinking_derives_named_reasoning_effort_for_capable_model() -> None:
    resolved = resolve_template_thinking(
        canonical_model="google/gemma-4-31B-turbo-TEE",
        provider_name="chutes",
        request_thinking=None,
        reasoning_effort=" high ",
    )

    assert resolved is not None
    assert resolved.thinking == LlmThinkingConfig(enabled=True, effort="high")
    assert resolved.chat_template_kwargs() == {"enable_thinking": True}


def test_resolve_template_thinking_prefers_explicit_request_thinking() -> None:
    resolved = resolve_template_thinking(
        canonical_model="google/gemma-4-31B-turbo-TEE",
        provider_name="custom-openai-compatible:gemma4-cloud-run-turbo",
        request_thinking=LlmThinkingConfig(enabled=False),
        reasoning_effort="high",
    )

    assert resolved is not None
    assert resolved.thinking == LlmThinkingConfig(enabled=False)
    assert resolved.chat_template_kwargs() == {"enable_thinking": False}


def test_resolve_template_thinking_ignores_numeric_blank_and_unsupported_inputs() -> None:
    assert (
        resolve_template_thinking(
            canonical_model="google/gemma-4-31B-turbo-TEE",
            provider_name="chutes",
            request_thinking=None,
            reasoning_effort="2048",
        )
        is None
    )
    assert (
        resolve_template_thinking(
            canonical_model="google/gemma-4-31B-turbo-TEE",
            provider_name="chutes",
            request_thinking=None,
            reasoning_effort=" ",
        )
        is None
    )
    assert (
        resolve_template_thinking(
            canonical_model="unsupported/model",
            provider_name="chutes",
            request_thinking=LlmThinkingConfig(enabled=True),
            reasoning_effort="high",
        )
        is None
    )
