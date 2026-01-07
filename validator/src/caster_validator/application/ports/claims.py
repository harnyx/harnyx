"""Port describing access to reference claims batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from caster_commons.domain.claim import EvaluationClaim


class ClaimsProviderPort(Protocol):
    """Supplies the set of reference claims used for a validator run."""

    def fetch(self, *, run_id: UUID | None = None) -> Sequence[EvaluationClaim]:
        """Return the ordered collection of claims for the supplied run."""


__all__ = ["ClaimsProviderPort"]
