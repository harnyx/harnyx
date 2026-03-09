from __future__ import annotations

from caster_commons.llm.providers.chutes import ChutesLlmProvider
from caster_commons.llm.providers.chutes import _serialize_message as _serialize_chutes_message
from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmInputToolResultPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmUsage,
    supports_tool_result_messages,
)


def test_tool_result_loop_capabilities_are_centralized() -> None:
    assert supports_tool_result_messages(provider="chutes", model="deepseek-ai/DeepSeek-V3.1") is True
    assert supports_tool_result_messages(provider="vertex", model="gemini-2.5-flash") is True
    assert supports_tool_result_messages(provider="unknown", model="gpt-4.1-mini") is False
    assert (
        supports_tool_result_messages(
            provider="vertex",
            model="publishers/anthropic/models/claude-sonnet-4-5@20250929",
        )
        is False
    )


def test_chutes_serializes_tool_result_message_as_tool_role_message() -> None:
    payload = _serialize_chutes_message(
        LlmMessage(
            role="user",
            content=(
                LlmInputToolResultPart(
                    tool_call_id="call-2",
                    name="get_repo_file",
                    output_json='{"path":"README.md","content":"demo"}',
                ),
            ),
        )
    )

    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "call-2"
    assert payload["name"] == "get_repo_file"
    assert payload["content"] == '{"path":"README.md","content":"demo"}'


def test_chutes_verify_accepts_tool_call_only_choice() -> None:
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
                            id="tc-2",
                            type="function",
                            name="get_repo_file",
                            arguments='{"path":"README.md"}',
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=LlmUsage(),
        finish_reason="tool_calls",
    )

    ok, retryable, reason = ChutesLlmProvider._verify_response(response)
    assert ok is True
    assert retryable is False
    assert reason is None
