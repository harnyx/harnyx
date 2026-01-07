"""Claim provider implementations."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from caster_commons.domain.claim import EvaluationClaim
from caster_validator.application.ports.claims import ClaimsProviderPort

_EVALUATION_CLAIM_ADAPTER = TypeAdapter(EvaluationClaim)


class StaticClaimsProvider(ClaimsProviderPort):
    """Returns a pre-supplied sequence of claims."""

    def __init__(self, claims: Sequence[EvaluationClaim]) -> None:
        if not claims:
            raise ValueError("claims sequence must not be empty")
        self._claims = tuple(claims)

    def fetch(self, *, run_id: UUID | None = None) -> tuple[EvaluationClaim, ...]:
        return self._claims


class FileClaimsProvider(ClaimsProviderPort):
    """Loads claims from a JSON Lines file on demand."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def fetch(self, *, run_id: UUID | None = None) -> tuple[EvaluationClaim, ...]:
        if not self._path.exists():
            raise FileNotFoundError(f"claims file {self._path} does not exist")
        if self._path.suffix.lower() not in (".jsonl", ""):
            raise ValueError("claims file must be a .jsonl document")

        claims: list[EvaluationClaim] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                data = line.strip()
                if not data:
                    continue
                payload: object = json.loads(data)
                try:
                    claims.append(_EVALUATION_CLAIM_ADAPTER.validate_python(payload))
                except ValidationError as exc:
                    raise ValueError(f"invalid claim at line {idx}") from exc

        if not claims:
            raise ValueError(f"claims file {self._path} did not contain any entries")
        return tuple(claims)


__all__ = ["FileClaimsProvider", "StaticClaimsProvider"]
