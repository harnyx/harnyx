"""Domain-specific exception types."""

from __future__ import annotations

from harnyx_commons.errors import BudgetExceededError, ConcurrencyLimitError

__all__ = ["BudgetExceededError", "ConcurrencyLimitError"]
