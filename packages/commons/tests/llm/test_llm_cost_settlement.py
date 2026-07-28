from __future__ import annotations

import pytest

from harnyx_commons.llm.cost_settlement import (
    settled_cost_from_metadata,
    settled_response_cost,
)
from harnyx_commons.llm.schema import LlmResponse, LlmUsage


def _response(
    *,
    usage: LlmUsage | None = None,
    metadata: dict[str, object] | None = None,
) -> LlmResponse:
    return LlmResponse(
        id="resp-1",
        choices=(),
        usage=usage or LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, total_tokens=3_000),
        metadata=metadata,
    )


def test_openrouter_provider_returned_cost_wins() -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": {"is_byok": False, "cost": 0.0123}}}),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.0123)
    assert cost.provider == "openrouter"
    assert cost.evidence["settlement_source"] == "provider_returned"
    assert cost.evidence["pricing_origin"] == "openrouter_usage_cost"
    assert cost.evidence["usage"] == {"is_byok": False, "cost": 0.0123}


def test_openrouter_byok_adds_usage_cost_and_upstream_inference_cost() -> None:
    cost = settled_response_cost(
        _response(
            metadata={
                "raw_response": {
                    "usage": {
                        "is_byok": True,
                        "cost": 0.01,
                        "cost_details": {"upstream_inference_cost": 0.09},
                    }
                }
            }
        ),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.10)
    assert cost.evidence["pricing_origin"] == "openrouter_usage_cost_and_upstream_inference_cost"
    assert cost.evidence["usage"] == {
        "is_byok": True,
        "cost": 0.01,
        "cost_details": {"upstream_inference_cost": 0.09},
    }


def test_openrouter_non_byok_does_not_add_returned_upstream_inference_cost() -> None:
    cost = settled_response_cost(
        _response(
            metadata={
                "raw_response": {
                    "usage": {
                        "is_byok": False,
                        "cost": 0.01,
                        "cost_details": {"upstream_inference_cost": 0.09},
                    }
                }
            }
        ),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.01)
    assert cost.evidence["usage"] == {
        "is_byok": False,
        "cost": 0.01,
        "cost_details": {"upstream_inference_cost": 0.09},
    }


def test_openrouter_explicit_non_byok_zero_cost_is_provider_returned() -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": {"is_byok": False, "cost": 0}}}),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.cost_usd == 0.0
    assert cost.evidence["settlement_source"] == "provider_returned"


def test_openrouter_missing_usage_cost_uses_static_pricing() -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": {}}}),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.001069)
    assert cost.provider == "openrouter"
    assert cost.evidence["settlement_source"] == "static_pricing"
    assert cost.evidence["pricing_origin"] == "miner_tool_llm_pricing"
    assert cost.evidence["reasoning_tokens"] is None
    assert cost.evidence["is_byok_status"] == "missing"
    assert cost.evidence["cost_status"] == "missing"
    assert cost.evidence["upstream_inference_cost_status"] == "missing"


def test_ai_gateway_provider_metadata_cost_wins() -> None:
    cost = settled_response_cost(
        _response(
            metadata={
                "raw_response": {
                    "providerMetadata": {
                        "gateway": {
                            "cost": "0.0123",
                        },
                    },
                },
            },
        ),
        provider="ai_gateway",
        model="zai/glm-5.2-fast",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.0123)
    assert cost.provider == "ai_gateway"
    assert cost.evidence["settlement_source"] == "provider_returned"
    assert cost.evidence["pricing_origin"] == "ai_gateway_provider_metadata_cost"


@pytest.mark.parametrize("raw_cost", [True, "not-a-decimal", float("nan"), -0.01])
def test_ai_gateway_malformed_provider_metadata_cost_falls_back_to_static_pricing(raw_cost: object) -> None:
    cost = settled_response_cost(
        _response(
            metadata={
                "raw_response": {
                    "providerMetadata": {
                        "gateway": {
                            "cost": raw_cost,
                        },
                    },
                },
            },
        ),
        provider="ai_gateway",
        model="openai/gpt-oss-20b",
    )

    assert cost is not None
    assert cost.evidence["settlement_source"] == "static_pricing"
    assert cost.evidence["pricing_origin"] == "miner_tool_llm_pricing"


def test_static_pricing_evidence_preserves_reported_reasoning_tokens() -> None:
    cost = settled_response_cost(
        _response(usage=LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, reasoning_tokens=7, total_tokens=3_007)),
        provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        model="nvidia/Gemma-4-31B-IT-NVFP4",
    )

    assert cost is not None
    assert cost.evidence["reasoning_tokens"] == 7


def test_static_pricing_evidence_keeps_unavailable_reasoning_tokens() -> None:
    cost = settled_response_cost(
        _response(
            usage=LlmUsage(prompt_tokens=1_000, completion_tokens=2_000, reasoning_tokens=None, total_tokens=3_000)
        ),
        provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        model="nvidia/Gemma-4-31B-IT-NVFP4",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.00089)
    assert cost.evidence["reasoning_tokens"] is None


def test_openrouter_malformed_usage_cost_falls_back_to_static_pricing() -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": {"is_byok": False, "cost": True}}}),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.evidence["settlement_source"] == "static_pricing"
    assert cost.evidence["is_byok_status"] == "valid"
    assert cost.evidence["cost_status"] == "malformed"


@pytest.mark.parametrize(
    ("usage", "expected_statuses"),
    [
        (
            {"is_byok": "true", "cost": 0.01, "cost_details": {"upstream_inference_cost": 0.09}},
            ("malformed", "valid", "valid"),
        ),
        (
            {"is_byok": True, "cost": 0.01},
            ("valid", "valid", "missing"),
        ),
        (
            {"is_byok": True, "cost": 0.01, "cost_details": {"upstream_inference_cost": "0.09"}},
            ("valid", "valid", "malformed"),
        ),
    ],
)
def test_openrouter_incomplete_byok_evidence_records_static_fallback_statuses(
    usage: dict[str, object],
    expected_statuses: tuple[str, str, str],
) -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": usage}}),
        provider="openrouter",
        model="deepseek/deepseek-v3.2",
    )

    assert cost is not None
    assert cost.evidence["settlement_source"] == "static_pricing"
    assert (
        cost.evidence["is_byok_status"],
        cost.evidence["cost_status"],
        cost.evidence["upstream_inference_cost_status"],
    ) == expected_statuses


def test_openrouter_missing_is_byok_assumes_non_byok_and_uses_returned_cost() -> None:
    cost = settled_response_cost(
        _response(
            metadata={
                "raw_response": {
                    "usage": {
                        "cost": 0.01,
                        "cost_details": {"upstream_inference_cost": 0.09},
                    }
                }
            }
        ),
        provider="openrouter",
        model="unknown/model",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.01)
    assert cost.evidence["pricing_origin"] == "openrouter_usage_cost"
    assert cost.evidence["is_byok_status"] == "missing"
    assert cost.evidence["usage"] == {
        "cost": 0.01,
        "cost_details": {"upstream_inference_cost": 0.09},
    }


def test_openrouter_malformed_is_byok_without_static_pricing_returns_none() -> None:
    cost = settled_response_cost(
        _response(metadata={"raw_response": {"usage": {"is_byok": "true", "cost": 0.0}}}),
        provider="openrouter",
        model="unknown/model",
    )

    assert cost is None


def test_custom_openai_compatible_alias_uses_static_pricing() -> None:
    cost = settled_response_cost(
        _response(),
        provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        model="nvidia/Gemma-4-31B-IT-NVFP4",
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.00089)
    assert cost.evidence["canonical_model"] == "google/gemma-4-31B-turbo-TEE"


def test_unknown_static_pricing_returns_none() -> None:
    cost = settled_response_cost(
        _response(),
        provider="vertex",
        model="unknown/model",
    )

    assert cost is None


def test_settled_cost_from_metadata_requires_complete_normalized_metadata() -> None:
    assert settled_cost_from_metadata({"actual_cost_usd": 0.01}) is None
    assert (
        settled_cost_from_metadata(
            {
                "actual_cost_usd": 0.01,
                "actual_cost_provider": "chutes",
                "actual_cost_evidence": {"settlement_source": "static_pricing"},
            }
        )
        is not None
    )


def test_settled_cost_from_metadata_prefers_retry_aggregate_cost() -> None:
    cost = settled_cost_from_metadata(
        {
            "actual_cost_usd": 0.02,
            "actual_cost_usd_total": 0.03,
            "actual_cost_provider": "chutes",
            "actual_cost_evidence": {"settlement_source": "cached_provider_pricing"},
        }
    )

    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.03)
    assert cost.provider == "chutes"
    assert cost.evidence == {
        "settlement_source": "retry_aggregate",
        "pricing_origin": "actual_cost_usd_total",
        "actual_cost_usd_total": 0.03,
        "final_response_actual_cost_usd": 0.02,
        "final_response_evidence": {"settlement_source": "cached_provider_pricing"},
    }


@pytest.mark.parametrize("bad_cost", [True, float("nan"), float("inf"), float("-inf"), -0.01, "0.01"])
def test_settled_cost_from_metadata_rejects_invalid_normalized_cost(bad_cost: object) -> None:
    with pytest.raises(ValueError, match="actual_cost_usd"):
        settled_cost_from_metadata(
            {
                "actual_cost_usd": bad_cost,
                "actual_cost_provider": "chutes",
                "actual_cost_evidence": {"settlement_source": "static_pricing"},
            }
        )


@pytest.mark.parametrize("bad_cost", [True, float("nan"), float("inf"), float("-inf"), -0.01, "0.01"])
def test_settled_cost_from_metadata_rejects_invalid_aggregate_cost(bad_cost: object) -> None:
    with pytest.raises(ValueError, match="actual_cost_usd_total"):
        settled_cost_from_metadata(
            {
                "actual_cost_usd": 0.01,
                "actual_cost_usd_total": bad_cost,
                "actual_cost_provider": "chutes",
                "actual_cost_evidence": {"settlement_source": "static_pricing"},
            }
        )
