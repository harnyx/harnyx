from caster_commons.llm.provider import _request_snapshot
from caster_commons.llm.schema import GroundedLlmRequest, LlmMessage, LlmMessageContentPart, LlmTool


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
