from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from caster_commons.domain.claim import EvaluationClaim, ReferenceAnswer, Rubric
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_validator.application.providers.claims import FileClaimsProvider, StaticClaimsProvider

_EVALUATION_CLAIM_ADAPTER = TypeAdapter(EvaluationClaim)


def _sample_claim(text: str) -> EvaluationClaim:
    return EvaluationClaim(
        claim_id=uuid4(),
        text=text,
        rubric=Rubric(
            title="Accuracy",
            description="Check correctness.",
            verdict_options=BINARY_VERDICT_OPTIONS,
        ),
        reference_answer=ReferenceAnswer(verdict=1, justification="ok", citations=()),
    )


def test_static_provider_returns_supplied_claims() -> None:
    claims = (_sample_claim("A"), _sample_claim("B"))
    provider = StaticClaimsProvider(claims)

    result = provider.fetch()

    assert result == claims


def test_static_provider_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError):
        StaticClaimsProvider(())


def test_file_provider_loads_jsonl(tmp_path: Path) -> None:
    claim = _sample_claim("jsonl claim")
    path = tmp_path / "claims.jsonl"
    path.write_text(_EVALUATION_CLAIM_ADAPTER.dump_json(claim).decode() + "\n", encoding="utf-8")

    provider = FileClaimsProvider(path)
    result = provider.fetch()

    assert result[0].text == "jsonl claim"


def test_file_provider_requires_existing_path(tmp_path: Path) -> None:
    provider = FileClaimsProvider(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        provider.fetch()


def test_file_provider_rejects_non_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "claims.json"
    path.write_text("[]", encoding="utf-8")
    provider = FileClaimsProvider(path)

    with pytest.raises(ValueError):
        provider.fetch()
