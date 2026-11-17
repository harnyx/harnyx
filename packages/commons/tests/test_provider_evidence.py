"""Protect lossless evidence retention and the private/public boundary."""

import json

import pytest
from pydantic import TypeAdapter

from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.llm.judge_usage import judge_usage_from_response, merge_judge_usage
from harnyx_commons.llm.schema import LlmResponse, LlmUsage
from harnyx_commons.provider_evidence import ProviderEvidence


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
        id="receipt", choices=(), usage=LlmUsage(),
        metadata={"_provider_evidence": evidence.model_dump(mode="json")},
    )
    usage = judge_usage_from_response(response, default_provider="test", default_model="judge")
    adapter = TypeAdapter(JudgeUsageSummary)
    private = adapter.dump_json(usage, context={"private_provider_evidence": True})
    restored = adapter.validate_json(private)
    assert restored.provider_evidence[0].restore() == original
    assert "provider_evidence" not in adapter.dump_python(restored, mode="json")
    assert "provider_evidence" not in adapter.json_schema(mode="serialization")["properties"]
    assert "_provider_evidence" not in response.payload["metadata"]


def test_pairwise_judge_evidence_deduplicates_prompts_and_reuses_saved_reasoning():
    prompt, reasoning = "shared system prompt " * 1000, "saved reasoning " * 1000
    exchange = {"request": prompt, "response": {"reasoning": reasoning}}
    response = LlmResponse(
        id="receipt", choices=(), usage=LlmUsage(),
        metadata={"_provider_evidence": ProviderEvidence.capture([exchange]).model_dump(mode="json")},
    )
    usage = judge_usage_from_response(response, default_provider="test", default_model="judge")
    merged = merge_judge_usage((usage, usage))
    base = {"reasoning": reasoning}
    stored = TypeAdapter(JudgeUsageSummary).dump_python(
        merged, mode="json", context={"private_provider_evidence": True, "provider_evidence_base": base},
    )
    evidence = ProviderEvidence.model_validate(stored["provider_evidence"][0])
    assert evidence.restore(base=base) == [exchange, exchange]
    assert evidence.model_dump_json().count(prompt) == 1
    assert reasoning not in evidence.model_dump_json()
    reloaded = TypeAdapter(JudgeUsageSummary).validate_python(stored)
    moved = TypeAdapter(JudgeUsageSummary).dump_python(
        reloaded, mode="json", context={
            "private_provider_evidence": True, "provider_evidence_source_base": base,
        },
    )
    assert ProviderEvidence.model_validate(moved["provider_evidence"][0]).restore() == [exchange, exchange]
