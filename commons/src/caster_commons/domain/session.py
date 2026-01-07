"""Session lifecycle and budgeting primitives shared across services."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from uuid import UUID


class SessionStatus(str, Enum):
    """Lifecycle states for evaluation sessions."""

    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    ERROR = "error"
    TIMED_OUT = "timed_out"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class LlmUsageTotals:
    """Accumulated token usage for a single provider/model pair."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")

    def accumulate(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> LlmUsageTotals:
        """Return a new totals record with the supplied deltas applied."""
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        return LlmUsageTotals(
            prompt_tokens=self.prompt_tokens + prompt_tokens,
            completion_tokens=self.completion_tokens + completion_tokens,
            total_tokens=self.total_tokens + total_tokens,
            call_count=self.call_count + 1,
        )


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """Cost and LLM usage totals scoped to a session."""

    total_cost_usd: float = 0.0
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    llm_tokens_last_call: int = 0
    llm_usage_totals: dict[str, dict[str, LlmUsageTotals]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_cost_usd < 0.0:
            raise ValueError("total_cost_usd must be non-negative")
        if self.llm_tokens_last_call < 0:
            raise ValueError("llm_tokens_last_call must be non-negative")

    def update(
        self,
        *,
        llm_tokens_last_call: int | None = None,
        llm_usage_totals: dict[str, dict[str, LlmUsageTotals]] | None = None,
        total_cost_usd: float | None = None,
        cost_by_provider: dict[str, float] | None = None,
    ) -> SessionUsage:
        """Return a new usage record with updated counters."""
        return replace(
            self,
            llm_tokens_last_call=(
                self.llm_tokens_last_call if llm_tokens_last_call is None else llm_tokens_last_call
            ),
            llm_usage_totals=self.llm_usage_totals if llm_usage_totals is None else llm_usage_totals,
            total_cost_usd=self.total_cost_usd if total_cost_usd is None else total_cost_usd,
            cost_by_provider=self.cost_by_provider if cost_by_provider is None else cost_by_provider,
        )

    def require_usage_totals(self) -> dict[str, dict[str, LlmUsageTotals]]:
        """Return LLM usage totals, raising if none have been recorded."""
        if not self.llm_usage_totals:
            raise ValueError("llm usage totals missing for session usage")

        return {provider: dict(models) for provider, models in self.llm_usage_totals.items()}


@dataclass(frozen=True, slots=True)
class Session:
    """Session lifecycle record."""

    session_id: UUID
    uid: int
    claim_id: UUID
    issued_at: datetime
    expires_at: datetime
    usage: SessionUsage = field(default_factory=SessionUsage)
    status: SessionStatus = SessionStatus.ACTIVE

    def __post_init__(self) -> None:
        if self.uid <= 0:
            raise ValueError("uid must be positive")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")

    def mark_exhausted(self) -> Session:
        """Mark the session as exhausted."""
        return replace(self, status=SessionStatus.EXHAUSTED)

    def mark_timed_out(self) -> Session:
        """Mark the session as timed out."""
        return replace(self, status=SessionStatus.TIMED_OUT)

    def mark_error(self) -> Session:
        """Mark the session as failed."""
        return replace(self, status=SessionStatus.ERROR)

    def mark_completed(self) -> Session:
        """Mark the session as completed."""
        return replace(self, status=SessionStatus.COMPLETED)

    def with_usage(self, usage: SessionUsage) -> Session:
        """Return a session with updated usage counters."""
        return replace(self, usage=usage)


__all__ = [
    "LlmUsageTotals",
    "Session",
    "SessionUsage",
    "SessionStatus",
]
