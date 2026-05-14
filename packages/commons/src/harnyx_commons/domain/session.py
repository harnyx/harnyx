"""Session lifecycle and budgeting primitives shared across services."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Final, cast
from uuid import UUID

_UNSET: Final = object()


class SessionStatus(StrEnum):
    """Lifecycle states for evaluation sessions."""

    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    ERROR = "error"
    TIMED_OUT = "timed_out"
    COMPLETED = "completed"


class SessionFailureCode(StrEnum):
    """Transient execution markers attached to a live session."""

    TOOL_PROVIDER_FAILED = "tool_provider_failed"


@dataclass(frozen=True, slots=True)
class LlmUsageTotals:
    """Accumulated token usage for a single provider/model pair."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    call_count: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.reasoning_tokens < 0:
            raise ValueError("reasoning_tokens must be non-negative")
        if self.call_count < 0:
            raise ValueError("call_count must be non-negative")

    def accumulate(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> LlmUsageTotals:
        """Return a new totals record with the supplied deltas applied."""
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if reasoning_tokens < 0:
            raise ValueError("reasoning_tokens must be non-negative")
        return LlmUsageTotals(
            prompt_tokens=self.prompt_tokens + prompt_tokens,
            completion_tokens=self.completion_tokens + completion_tokens,
            total_tokens=self.total_tokens + total_tokens,
            reasoning_tokens=self.reasoning_tokens + reasoning_tokens,
            call_count=self.call_count + 1,
        )


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """Cost and LLM usage totals scoped to a session."""

    total_cost_usd: float = 0.0
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    reference_total_cost_usd: float = 0.0
    reference_cost_by_provider: dict[str, float] = field(default_factory=dict)
    actual_total_cost_usd: float | None = 0.0
    actual_cost_by_provider: dict[str, float] = field(default_factory=dict)
    llm_tokens_last_call: int = 0
    llm_usage_totals: dict[str, dict[str, LlmUsageTotals]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_total_cost_usd == 0.0 and self.total_cost_usd != 0.0:
            object.__setattr__(self, "reference_total_cost_usd", self.total_cost_usd)
        if self.total_cost_usd == 0.0 and self.reference_total_cost_usd != 0.0:
            object.__setattr__(self, "total_cost_usd", self.reference_total_cost_usd)
        if not self.reference_cost_by_provider and self.cost_by_provider:
            object.__setattr__(self, "reference_cost_by_provider", dict(self.cost_by_provider))
        if not self.cost_by_provider and self.reference_cost_by_provider:
            object.__setattr__(self, "cost_by_provider", dict(self.reference_cost_by_provider))
        if self.total_cost_usd < 0.0:
            raise ValueError("total_cost_usd must be non-negative")
        if self.reference_total_cost_usd < 0.0:
            raise ValueError("reference_total_cost_usd must be non-negative")
        if self.actual_total_cost_usd is not None and self.actual_total_cost_usd < 0.0:
            raise ValueError("actual_total_cost_usd must be non-negative when supplied")
        if self.llm_tokens_last_call < 0:
            raise ValueError("llm_tokens_last_call must be non-negative")

    def update(
        self,
        *,
        llm_tokens_last_call: int | None = None,
        llm_usage_totals: dict[str, dict[str, LlmUsageTotals]] | None = None,
        total_cost_usd: float | None = None,
        cost_by_provider: dict[str, float] | None = None,
        reference_total_cost_usd: float | None = None,
        reference_cost_by_provider: dict[str, float] | None = None,
        actual_total_cost_usd: float | None | object = _UNSET,
        actual_cost_by_provider: dict[str, float] | None = None,
    ) -> SessionUsage:
        """Return a new usage record with updated counters."""
        updated_reference_total = (
            self.reference_total_cost_usd
            if reference_total_cost_usd is None
            else reference_total_cost_usd
        )
        updated_reference_by_provider = (
            self.reference_cost_by_provider
            if reference_cost_by_provider is None
            else reference_cost_by_provider
        )
        return replace(
            self,
            llm_tokens_last_call=(
                self.llm_tokens_last_call if llm_tokens_last_call is None else llm_tokens_last_call
            ),
            llm_usage_totals=self.llm_usage_totals if llm_usage_totals is None else llm_usage_totals,
            total_cost_usd=(
                updated_reference_total if total_cost_usd is None else total_cost_usd
            ),
            cost_by_provider=(
                updated_reference_by_provider if cost_by_provider is None else cost_by_provider
            ),
            reference_total_cost_usd=updated_reference_total,
            reference_cost_by_provider=updated_reference_by_provider,
            actual_total_cost_usd=(
                self.actual_total_cost_usd
                if actual_total_cost_usd is _UNSET
                else cast(float | None, actual_total_cost_usd)
            ),
            actual_cost_by_provider=(
                self.actual_cost_by_provider
                if actual_cost_by_provider is None
                else actual_cost_by_provider
            ),
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
    task_id: UUID
    issued_at: datetime
    expires_at: datetime
    budget_usd: float
    hard_limit_usd: float | None = None
    usage: SessionUsage = field(default_factory=SessionUsage)
    status: SessionStatus = SessionStatus.ACTIVE
    active_attempt: int = 0
    failure_code: SessionFailureCode | None = None
    failure_attempt: int | None = None

    def __post_init__(self) -> None:
        if self.uid <= 0:
            raise ValueError("uid must be positive")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        if self.budget_usd < 0.0:
            raise ValueError("budget_usd must be non-negative")
        if self.hard_limit_usd is not None and self.hard_limit_usd < 0.0:
            raise ValueError("hard_limit_usd must be non-negative")
        if self.effective_hard_limit_usd < self.budget_usd:
            raise ValueError("hard_limit_usd must be greater than or equal to budget_usd")
        if self.active_attempt < 0:
            raise ValueError("active_attempt must be non-negative")
        if self.failure_code is None and self.failure_attempt is not None:
            raise ValueError("failure_attempt requires failure_code")
        if self.failure_code is not None and self.failure_attempt is None:
            raise ValueError("failure_code requires failure_attempt")
        if self.failure_attempt is not None and self.failure_attempt < 0:
            raise ValueError("failure_attempt must be non-negative")
        if self.failure_attempt is not None and self.failure_attempt > self.active_attempt:
            raise ValueError("failure_attempt must not exceed active_attempt")

    @property
    def effective_hard_limit_usd(self) -> float:
        """Return the enforced budget ceiling for this session."""
        return self.budget_usd if self.hard_limit_usd is None else self.hard_limit_usd

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

    def begin_attempt(self) -> Session:
        """Advance to the next retry attempt and clear any stale failure marker."""
        return replace(
            self,
            active_attempt=self.active_attempt + 1,
            failure_code=None,
            failure_attempt=None,
        )

    def mark_failure_code(self, failure_code: SessionFailureCode) -> Session:
        """Return a session annotated with a transient execution failure marker."""
        return replace(
            self,
            failure_code=failure_code,
            failure_attempt=self.active_attempt,
        )

    def clear_failure_code(self) -> Session:
        """Return a session with any transient execution failure marker removed."""
        return replace(self, failure_code=None, failure_attempt=None)

    def consume_failure_code(self) -> tuple[Session, SessionFailureCode | None]:
        """Return and clear the current-attempt failure marker, if present."""
        if self.failure_code is None:
            return self, None
        if self.failure_attempt != self.active_attempt:
            return self.clear_failure_code(), None
        code = self.failure_code
        return self.clear_failure_code(), code

    def with_usage(self, usage: SessionUsage) -> Session:
        """Return a session with updated usage counters."""
        return replace(self, usage=usage)


__all__ = [
    "LlmUsageTotals",
    "Session",
    "SessionFailureCode",
    "SessionUsage",
    "SessionStatus",
]
