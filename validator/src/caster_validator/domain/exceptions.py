"""Domain-specific exception types."""

from __future__ import annotations

from caster_commons.errors import BudgetExceededError, ConcurrencyLimitError


class InvalidCitationError(ValueError):
    """Raised when a miner cites an unknown or mismatched receipt."""


__all__ = ["BudgetExceededError", "InvalidCitationError", "ConcurrencyLimitError"]
