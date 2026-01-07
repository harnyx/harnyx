"""Budget limit enforcement helpers."""

from __future__ import annotations

from caster_commons.errors import BudgetExceededError


class BudgetValidator:
    """Validates projected tool-call costs against a fixed USD limit."""

    def __init__(self, limit_usd: float) -> None:
        if limit_usd < 0:
            raise ValueError("cost limit must be non-negative")
        self._limit = float(limit_usd)

    @property
    def limit_usd(self) -> float:
        return self._limit

    def assert_within_limits(self, projected_total_usd: float) -> None:
        if projected_total_usd > self._limit:
            raise BudgetExceededError(
                "cost budget exhausted "
                f"(limit=${self._limit:.3f}, used=${projected_total_usd:.3f})"
            )

