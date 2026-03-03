from __future__ import annotations

from caster_commons.llm.providers.chutes import ChutesLlmProvider, _parse_chutes_response_payload
from caster_commons.llm.schema import LlmResponse, LlmUsage


def test_parse_payload_skips_malformed_choice_and_keeps_valid_choice() -> None:
    payload = {
        "id": "resp_1",
        "choices": [
            {"message": {"content": "ok"}},
            None,
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    assert parsed.choices[0].message.content[0].text == "ok"


def test_all_malformed_choices_fall_back_to_retryable_empty_choices_verifier() -> None:
    payload = {
        "id": "resp_2",
        "choices": [None],
    }

    parsed = _parse_chutes_response_payload(payload)
    ok, retryable, reason = ChutesLlmProvider._verify_response(
        LlmResponse(
            id="resp_2",
            choices=parsed.choices,
            usage=LlmUsage(),
        )
    )

    assert (ok, retryable, reason) == (False, True, "empty_choices")


def test_non_array_choices_fall_back_to_retryable_empty_choices_verifier() -> None:
    payload = {
        "id": "resp_3",
        "choices": {"unexpected": "object"},
    }

    parsed = _parse_chutes_response_payload(payload)
    ok, retryable, reason = ChutesLlmProvider._verify_response(
        LlmResponse(
            id="resp_3",
            choices=parsed.choices,
            usage=LlmUsage(),
        )
    )

    assert (ok, retryable, reason) == (False, True, "empty_choices")


def test_parse_payload_ignores_malformed_tool_call_and_keeps_valid_choice() -> None:
    payload = {
        "id": "resp_4",
        "choices": [
            {
                "message": {
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "tc-valid",
                            "type": "function",
                            "function": {
                                "name": "summarize",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "tc-bad",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
            },
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    assert parsed.choices[0].message.content[0].text == "ok"
    tool_calls = parsed.choices[0].message.tool_calls or ()
    assert tuple(call.id for call in tool_calls) == ("tc-valid",)


def test_parse_payload_skips_malformed_content_fragment_and_keeps_valid_text() -> None:
    payload = {
        "id": "resp_5",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "ok"},
                        None,
                    ]
                },
            },
        ],
    }

    parsed = _parse_chutes_response_payload(payload)

    assert len(parsed.choices) == 1
    parts = parsed.choices[0].message.content
    assert len(parts) == 1
    assert parts[0].text == "ok"
