import asyncio
from dataclasses import replace

import pytest

from harnyx_commons.llm.provider import _request_snapshot
from harnyx_commons.llm.schema import (
    GroundedLlmRequest,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmTool,
)


def test_request_snapshot_redacts_nested_api_key_fields() -> None:
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
                            "auth_config": {
                                "api_key_config": {
                                    "api_key_string": "secret-auth-config-snake",
                                    "apiKeyString": "secret-auth-config-camel",
                                }
                            },
                            "api_auth": {
                                "api_key_config": {
                                    "api_key_string": "secret-legacy-snake",
                                    "apiKeyString": "secret-legacy-camel",
                                }
                            },
                        }
                    }
                },
            ),
        ),
    )

    snapshot = _request_snapshot(request)
    external_api = snapshot["tools"][0]["config"]["retrieval"]["external_api"]
    auth_config = external_api["auth_config"]["api_key_config"]
    legacy_auth_config = external_api["api_auth"]["api_key_config"]

    assert auth_config["api_key_string"] == "[REDACTED]"
    assert auth_config["apiKeyString"] == "[REDACTED]"
    assert legacy_auth_config["api_key_string"] == "[REDACTED]"
    assert legacy_auth_config["apiKeyString"] == "[REDACTED]"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("callback_field", ("on_output", "on_provider_dispatch", "postprocessor"))
async def test_request_snapshot_does_not_copy_runtime_callbacks(callback_field: str) -> None:
    """Observability must not copy live callback state and abort an LLM call."""

    class CallbackOwner:
        def __init__(self) -> None:
            self.pending = asyncio.get_running_loop().create_future()

        def callback(self, *_args: object) -> None:
            pass

    owner = CallbackOwner()
    callback = owner.callback
    request = replace(
        LlmRequest(
            provider="openrouter",
            model="openai/gpt-oss-20b",
            messages=(),
            temperature=None,
            max_output_tokens=None,
        ),
        **{callback_field: callback},
    )

    snapshot = _request_snapshot(request)

    assert callback_field not in snapshot
    assert snapshot["model"] == request.model
    assert getattr(request, callback_field) is callback
    assert not owner.pending.done()
