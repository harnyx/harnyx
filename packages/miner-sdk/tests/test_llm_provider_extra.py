from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.llm_provider_extra import (
    AiGatewayExtra,
    OpenRouterExtra,
    OpenRouterProviderSelection,
    validate_provider_extra,
)


def test_openrouter_provider_extra_accepts_provider_only_selection() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra={"provider": {"only": ["cerebras"]}},
    )

    assert isinstance(parsed, OpenRouterExtra)
    assert parsed.to_request_extra() == {"provider": {"only": ["cerebras"]}}


def test_openrouter_provider_extra_accepts_allow_fallbacks() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra={"provider": {"only": ["cerebras"], "allow_fallbacks": False}},
    )

    assert isinstance(parsed, OpenRouterExtra)
    assert parsed.to_request_extra() == {
        "provider": {"only": ["cerebras"], "allow_fallbacks": False}
    }


def test_openrouter_provider_extra_accepts_allow_fallbacks_without_provider_only() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra={"provider": {"allow_fallbacks": False}},
    )

    assert isinstance(parsed, OpenRouterExtra)
    assert parsed.to_request_extra() == {"provider": {"allow_fallbacks": False}}


def test_openrouter_provider_extra_normalizes_provider_names_without_changing_case() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra=OpenRouterExtra(provider=OpenRouterProviderSelection(only=(" Cerebras ",))),
    )

    assert parsed is not None
    assert parsed.to_request_extra() == {"provider": {"only": ["Cerebras"]}}


def test_chutes_rejects_provider_extra() -> None:
    with pytest.raises(ValueError, match="provider_extra is not supported for provider 'chutes'"):
        validate_provider_extra(
            provider="chutes",
            provider_extra={"provider": {"only": ["cerebras"]}},
        )


def test_ai_gateway_provider_extra_accepts_provider_selection() -> None:
    parsed = validate_provider_extra(
        provider="ai_gateway",
        provider_extra={"provider": {"only": ["cerebras"]}},
    )

    assert isinstance(parsed, AiGatewayExtra)
    assert parsed.to_request_extra() == {"provider": {"only": ["cerebras"]}}


def test_ai_gateway_provider_extra_accepts_provider_options_gateway_selection() -> None:
    parsed = validate_provider_extra(
        provider="ai_gateway",
        provider_extra={"providerOptions": {"gateway": {"only": ["cerebras"]}}},
    )

    assert isinstance(parsed, AiGatewayExtra)
    assert parsed.to_request_extra() == {"providerOptions": {"gateway": {"only": ["cerebras"]}}}


def test_ai_gateway_provider_extra_accepts_matching_vercel_forms() -> None:
    parsed = validate_provider_extra(
        provider="ai_gateway",
        provider_extra={
            "provider": {"only": ["cerebras"]},
            "providerOptions": {"gateway": {"only": ["cerebras"]}},
        },
    )

    assert parsed is not None
    assert parsed.to_request_extra() == {
        "provider": {"only": ["cerebras"]},
        "providerOptions": {"gateway": {"only": ["cerebras"]}},
    }


def test_ai_gateway_provider_extra_rejects_conflicting_vercel_forms() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={
                "provider": {"only": ["cerebras"]},
                "providerOptions": {"gateway": {"only": ["groq"]}},
            },
        )


def test_ai_gateway_provider_extra_rejects_raw_provider_string() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"provider": "cerebras"},
        )


def test_ai_gateway_provider_extra_rejects_provider_options_snake_case() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"provider_options": {"gateway": {"only": ["cerebras"]}}},
        )


def test_ai_gateway_provider_extra_rejects_empty_payload() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(provider="ai_gateway", provider_extra={})


def test_openrouter_provider_extra_rejects_empty_provider_selection() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(provider="openrouter", provider_extra={"provider": {}})


def test_provider_extra_rejects_common_reasoning_field() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"reasoning": {"effort": "high"}},
        )

    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"reasoning": {"effort": "high"}},
        )


def test_provider_extra_rejects_common_thinking_field() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"thinking": {"enabled": True}},
        )

    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"thinking": {"enabled": True}},
        )


def test_openrouter_provider_extra_rejects_unapproved_provider_preferences() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"provider": {"only": ["cerebras"], "order": ["cerebras"]}},
        )


def test_ai_gateway_provider_extra_rejects_unapproved_provider_preferences() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"provider": {"only": ["cerebras"], "allow_fallbacks": False}},
        )


@pytest.mark.parametrize(
    "provider_only",
    ([], [""], ["  "], ["cerebras", 1], "cerebras"),
)
def test_openrouter_provider_extra_rejects_invalid_provider_only_values(provider_only: object) -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"provider": {"only": provider_only}},
        )


@pytest.mark.parametrize("allow_fallbacks", ["false", 0, 1])
def test_openrouter_provider_extra_rejects_invalid_allow_fallbacks(allow_fallbacks: object) -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"provider": {"only": ["cerebras"], "allow_fallbacks": allow_fallbacks}},
        )


@pytest.mark.parametrize(
    "provider_only",
    ([], [""], ["  "], ["cerebras", 1], "cerebras"),
)
def test_ai_gateway_provider_extra_rejects_invalid_provider_only_values(provider_only: object) -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"provider": {"only": provider_only}},
        )


@pytest.mark.parametrize(
    "provider_only",
    ([], [""], ["  "], ["cerebras", 1], "cerebras"),
)
def test_ai_gateway_provider_extra_rejects_invalid_provider_options_only_values(provider_only: object) -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="ai_gateway",
            provider_extra={"providerOptions": {"gateway": {"only": provider_only}}},
        )
