from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.llm import (
    LlmChoiceMessage,
    LlmInputTextPart,
    LlmMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
)
from harnyx_miner_sdk.tools.llm_chat_models import LlmChatRequest


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather for one or more cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    }
                },
                "required": ["cities"],
            },
            "strict": True,
        },
    }


def _request_payload() -> dict[str, object]:
    return {
        "provider": "openrouter",
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [_tool()],
        "tool_choice": {
            "type": "function",
            "function": {"name": "lookup_weather"},
        },
        "parallel_tool_calls": True,
        "max_output_tokens": 512,
    }


def test_recursive_function_schema_and_strict_flag_round_trip() -> None:
    request = LlmChatRequest.model_validate(_request_payload())

    assert request.model_dump(mode="json", exclude_none=True)["tools"] == [_tool()]
    normalized = request.to_tool_request()
    assert normalized.tools is not None
    assert normalized.tools[0].function == _tool()["function"]
    assert normalized.tool_choice == {
        "type": "function",
        "function": {"name": "lookup_weather"},
    }
    assert normalized.parallel_tool_calls is True


def test_parameterless_function_definition_omits_parameters() -> None:
    payload = _request_payload()
    payload["tools"] = [
        {
            "type": "function",
            "function": {"name": "current_time", "description": "Return the current time."},
        }
    ]
    payload["tool_choice"] = "auto"

    dumped = LlmChatRequest.model_validate(payload).model_dump(mode="json", exclude_none=True)

    assert dumped["tools"] == [
        {
            "type": "function",
            "function": {"name": "current_time", "description": "Return the current time."},
        }
    ]


def test_valid_parallel_tool_transcript_preserves_opaque_reasoning_details() -> None:
    reasoning_details = [
        {"type": "reasoning.encrypted", "data": "opaque", "index": 0},
        {"type": "reasoning.text", "text": "Need both cities", "signature": "sig"},
    ]
    payload = _request_payload()
    payload["messages"] = [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-paris",
                    "type": "function",
                    "name": "lookup_weather",
                    "arguments": '{"cities":[{"name":"Paris"}]}',
                },
                {
                    "id": "call-rome",
                    "type": "function",
                    "name": "lookup_weather",
                    "arguments": '{"cities":[{"name":"Rome"}]}',
                },
            ],
            "reasoning_details": reasoning_details,
        },
        {"role": "tool", "tool_call_id": "call-rome", "content": '{"temperature":21}'},
        {"role": "tool", "tool_call_id": "call-paris", "content": '{"temperature":19}'},
    ]

    request = LlmChatRequest.model_validate(payload)
    dumped = request.model_dump(mode="json", exclude_none=True)

    assert dumped["messages"][1]["content"] is None
    assert dumped["messages"][1]["reasoning_details"] == reasoning_details
    normalized = request.to_tool_request()
    assert normalized.messages[1].reasoning_details == tuple(reasoning_details)
    assert tuple(part.tool_call_id for message in normalized.messages[2:] for part in message.content) == (
        "call-rome",
        "call-paris",
    )


def test_request_preserves_whitespace_in_recursive_and_opaque_json_values() -> None:
    payload = _request_payload()
    payload["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {
                    "type": "object",
                    "properties": {" city ": {"type": "string", "const": " Paris "}},
                },
            },
        }
    ]
    payload["messages"] = [
        {
            "role": "assistant",
            "content": " Answer with surrounding whitespace. ",
            "reasoning_details": [{"type": "reasoning.text", "text": "  opaque  "}],
        }
    ]

    dumped = LlmChatRequest.model_validate(payload).model_dump(mode="json", exclude_none=True)

    assert dumped["tools"][0]["function"]["parameters"]["properties"] == {
        " city ": {"type": "string", "const": " Paris "}
    }
    assert dumped["messages"][0]["content"] == " Answer with surrounding whitespace. "
    assert dumped["messages"][0]["reasoning_details"][0]["text"] == "  opaque  "


def test_choice_message_converts_directly_to_assistant_replay_shape() -> None:
    reasoning_details = ({"type": "reasoning.encrypted", "data": "opaque"},)
    message = LlmChoiceMessage(
        role="assistant",
        content=(LlmMessageContentPart(type="text", text="Checking."),),
        tool_calls=(
            LlmMessageToolCall(
                id="call-1",
                type="function",
                name="lookup_weather",
                arguments='{"cities":[{"name":"Paris"}]}',
            ),
        ),
        reasoning_details=reasoning_details,
    )

    replay = message.to_input_message()

    assert replay.role == "assistant"
    assert replay.tool_calls == message.tool_calls
    assert replay.reasoning_details == reasoning_details


@pytest.mark.parametrize("role", ("system", "user", "tool"))
def test_normalized_non_assistant_messages_reject_assistant_state(role: str) -> None:
    with pytest.raises(ValueError, match="tool_calls"):
        LlmMessage(
            role=role,  # type: ignore[arg-type]
            content=(LlmInputTextPart(text="message"),),
            tool_calls=(
                LlmMessageToolCall(
                    id="call-1",
                    type="function",
                    name="lookup_weather",
                    arguments="{}",
                ),
            ),
        )

    with pytest.raises(ValueError, match="reasoning_details"):
        LlmMessage(
            role=role,  # type: ignore[arg-type]
            content=(LlmInputTextPart(text="message"),),
            reasoning_details=({"type": "reasoning.encrypted", "data": "opaque"},),
        )


def test_max_tokens_alias_normalizes_to_canonical_output_field() -> None:
    payload = _request_payload()
    del payload["max_output_tokens"]
    payload["max_tokens"] = 512

    request = LlmChatRequest.model_validate(payload)
    dumped = request.model_dump(mode="json", exclude_none=True)

    assert request.max_output_tokens == 512
    assert dumped["max_output_tokens"] == 512
    assert "max_tokens" not in dumped


def test_max_tokens_alias_rejects_ambiguous_dual_input() -> None:
    payload = _request_payload()
    payload["max_tokens"] = 256

    with pytest.raises(ValidationError, match="mutually exclusive"):
        LlmChatRequest.model_validate(payload)


@pytest.mark.parametrize("removed_field", ("include", "response_format"))
def test_removed_and_extra_fields_are_rejected(removed_field: str) -> None:
    payload = _request_payload()
    payload[removed_field] = [] if removed_field == "include" else 1

    with pytest.raises(ValidationError, match=removed_field):
        LlmChatRequest.model_validate(payload)


def test_named_choice_requires_one_declared_unique_function() -> None:
    undeclared = _request_payload()
    undeclared["tool_choice"] = {"type": "function", "function": {"name": "missing"}}
    with pytest.raises(ValidationError, match="declared"):
        LlmChatRequest.model_validate(undeclared)

    duplicated = _request_payload()
    duplicated["tools"] = [_tool(), deepcopy(_tool())]
    with pytest.raises(ValidationError, match="unique"):
        LlmChatRequest.model_validate(duplicated)


@pytest.mark.parametrize(
    "arguments",
    ("not-json", "[]", '"scalar"', '{"value":NaN}', '{"value":Infinity}'),
)
def test_tool_call_arguments_must_be_a_json_object(arguments: str) -> None:
    payload = _request_payload()
    payload["messages"] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "name": "lookup_weather",
                    "arguments": arguments,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
    ]

    with pytest.raises(ValidationError, match="JSON object"):
        LlmChatRequest.model_validate(payload)


def test_duplicate_call_ids_are_rejected_within_one_pending_block() -> None:
    payload = _request_payload()
    call = {
        "id": "call-1",
        "type": "function",
        "name": "lookup_weather",
        "arguments": "{}",
    }
    payload["messages"] = [
        {"role": "assistant", "content": None, "tool_calls": [call, deepcopy(call)]},
        {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
    ]

    with pytest.raises(ValidationError, match="unique"):
        LlmChatRequest.model_validate(payload)


def test_call_id_may_be_reused_after_earlier_block_is_resolved() -> None:
    payload = _request_payload()
    call = {
        "id": "reused-id",
        "type": "function",
        "name": "lookup_weather",
        "arguments": "{}",
    }
    payload["messages"] = [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "reused-id", "content": "{}"},
        {"role": "assistant", "content": None, "tool_calls": [deepcopy(call)]},
        {"role": "tool", "tool_call_id": "reused-id", "content": "{}"},
    ]

    LlmChatRequest.model_validate(payload)


@pytest.mark.parametrize(
    "messages",
    (
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "name": "lookup_weather",
                        "arguments": "{}",
                    }
                ],
            }
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "name": "lookup_weather",
                        "arguments": "{}",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "unknown", "content": "{}"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "name": "lookup_weather",
                        "arguments": "{}",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "name": "lookup_weather",
                        "arguments": "{}",
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ],
    ),
)
def test_missing_duplicate_unknown_or_interrupted_tool_results_are_rejected(
    messages: list[dict[str, object]],
) -> None:
    payload = _request_payload()
    payload["messages"] = messages

    with pytest.raises(ValidationError, match="tool"):
        LlmChatRequest.model_validate(payload)
