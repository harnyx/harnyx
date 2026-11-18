"""Protect lossless evidence retention and the private/public boundary."""

import json

import pytest
from pydantic import TypeAdapter

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.miner_task import EvaluationDetails, EvaluationError, MinerTaskErrorCode
from harnyx_commons.llm.judge_usage import judge_usage_from_response, merge_judge_usage
from harnyx_commons.llm.schema import LlmResponse, LlmUsage
from harnyx_commons.provider_evidence import ProviderEvidence, judge_evidence_from_response


def test_original_excerpts_round_trip_without_repeating_stored_page_text():
    first, second = "first excerpt " * 1000, "second excerpt " * 1000
    base = {"response": {"content": first.strip() + "\n\n" + second.strip()}}
    original = {"results": [{"excerpts": ["\n " + first + "\t", second], "unknown": {"kept": True}}]}
    packed = ProviderEvidence.capture(original, base=base)
    serialized = packed.model_dump_json()
    assert len(serialized) < len(json.dumps(original)) / 10
    assert ProviderEvidence.model_validate_json(serialized).restore(base=base) == original
    assert original["results"][0]["excerpts"][0].startswith("\n ")


@pytest.mark.parametrize("unsafe", ["\x00", "\ud800", "\udfff"])
def test_postgres_unsafe_characters_are_retained_without_unsafe_jsonb_strings(unsafe):
    original = {"request": unsafe + "long text " * 100, "response": {unsafe: "exact"}}
    packed = ProviderEvidence.capture(original, base={"input": original["request"]})
    serialized = json.dumps(packed.model_dump(mode="json"), ensure_ascii=False)
    assert unsafe not in serialized
    assert ProviderEvidence.model_validate_json(serialized).restore() == original


def test_judge_evidence_survives_private_transport_but_is_absent_from_public_serialization():
    original = [{"request": {"prompt": "private prompt"}, "response": {"raw": "private output"}}]
    evidence = ProviderEvidence.capture(original)
    response = LlmResponse(
        id="receipt",
        choices=(),
        usage=LlmUsage(),
        metadata={"_provider_evidence": evidence.model_dump(mode="json")},
    )
    usage = judge_usage_from_response(response, default_provider="test", default_model="judge")
    adapter = TypeAdapter(EvaluationDetails)
    private = adapter.dump_json(
        EvaluationDetails(
            error=EvaluationError(code=MinerTaskErrorCode.NEVER_RAN, message="test"),
            scoring_judge_usage=usage,
            provider_evidence=judge_evidence_from_response(response),
        ),
        context={"private_provider_evidence": True},
    )
    restored = adapter.validate_json(private)
    assert restored.provider_evidence[0].restore() == original
    assert "provider_evidence" not in adapter.dump_python(restored, mode="json")
    assert "provider_evidence" not in adapter.json_schema(mode="serialization")["properties"]
    assert "_provider_evidence" not in response.payload["metadata"]
    assert not hasattr(usage, "provider_evidence")
    assert "provider_evidence" not in TypeAdapter(JudgeUsageSummary).dump_python(
        usage, context={"private_provider_evidence": True}
    )


def test_pairwise_judge_evidence_deduplicates_prompts_and_reuses_saved_reasoning():
    prompt, reasoning = "shared system prompt " * 1000, "saved reasoning " * 1000
    exchange = {"request": prompt, "response": {"reasoning": reasoning}}
    response = LlmResponse(
        id="receipt",
        choices=(),
        usage=LlmUsage(),
        metadata={"_provider_evidence": ProviderEvidence.capture([exchange]).model_dump(mode="json")},
    )
    usage = judge_usage_from_response(response, default_provider="test", default_model="judge")
    merged = merge_judge_usage((usage, usage))
    base = {"reasoning": reasoning}
    stored = TypeAdapter(EvaluationDetails).dump_python(
        EvaluationDetails(
            error=EvaluationError(code=MinerTaskErrorCode.NEVER_RAN, message="test"),
            scoring_judge_usage=merged,
            provider_evidence=judge_evidence_from_response(response) * 2,
        ),
        mode="json",
        context={"private_provider_evidence": True, "provider_evidence_base": base},
    )
    evidence = ProviderEvidence.model_validate(stored["scoring_judge_usage"]["provider_evidence"][0])
    assert evidence.restore(base=base) == [exchange, exchange]
    assert evidence.model_dump_json().count(prompt) == 1
    assert reasoning not in evidence.model_dump_json()
    reloaded = TypeAdapter(EvaluationDetails).validate_python(stored)
    moved = TypeAdapter(EvaluationDetails).dump_python(
        reloaded,
        mode="json",
        context={
            "private_provider_evidence": True,
            "provider_evidence_source_base": base,
        },
    )
    assert ProviderEvidence.model_validate(moved["scoring_judge_usage"]["provider_evidence"][0]).restore() == [
        exchange,
        exchange,
    ]
