"""Per-attempt LLM deadlines, independent of transport and observation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Literal

import httpx

from harnyx_miner_sdk.llm import Timeout

TimeoutPhase = Literal["prefill", "inactivity", "total"]


def streaming_transport_timeout(value: float | Timeout) -> httpx.Timeout:
    """Preserve request-derived setup limits; the attempt controller bounds reads."""
    total = value.total if isinstance(value, Timeout) else value
    return httpx.Timeout(total, read=None)


def resolve_timeout(value: float | Timeout | None, defaults: Timeout | None = None) -> Timeout | None:
    if value is None:
        return defaults
    policy = value if isinstance(value, Timeout) else Timeout(value)
    return policy.with_defaults(defaults) if defaults is not None else policy


class LlmAttemptTimeoutError(TimeoutError):
    """An owned deadline expired; contains timings but no provider output."""

    def __init__(self, attempt: _AttemptDeadline) -> None:
        self.phase = attempt.phase
        self.timeout = attempt.policy
        self.elapsed_seconds = attempt.now() - attempt.started_at
        self.first_output_seconds = (
            attempt.first_output_at - attempt.started_at if attempt.first_output_at is not None else None
        )
        self.last_output_seconds = (
            attempt.last_output_at - attempt.started_at if attempt.last_output_at is not None else None
        )
        super().__init__(f"LLM {self.phase} timeout after {self.elapsed_seconds:.3f}s")


class _AttemptDeadline:
    def __init__(self, policy: Timeout) -> None:
        self.policy = policy
        self.now = asyncio.get_running_loop().time
        self.started_at = self.now()
        self.total_at = self.started_at + policy.total
        self.first_output_at: float | None = None
        self.last_output_at: float | None = None
        self.phase: TimeoutPhase = "total"
        self.deadline_at = self.total_at
        self._select_deadline("prefill", policy.prefill, self.started_at)
        self.timer = asyncio.timeout_at(self.deadline_at)

    def _select_deadline(self, phase: TimeoutPhase, duration: float | None, start: float) -> None:
        self.phase, self.deadline_at = "total", self.total_at
        if duration is not None and start + duration < self.total_at:
            self.phase, self.deadline_at = phase, start + duration

    def progress(self) -> None:
        now = self.now()
        if now >= self.deadline_at:
            raise LlmAttemptTimeoutError(self)
        if self.first_output_at is None:
            self.first_output_at = now
        self.last_output_at = now
        self._select_deadline("inactivity", self.policy.inactivity, now)
        self.timer.reschedule(self.deadline_at)


_current_attempt: ContextVar[_AttemptDeadline | None] = ContextVar("llm_attempt_deadline", default=None)


def record_output_progress() -> None:
    attempt = _current_attempt.get()
    if attempt is not None:
        attempt.progress()


@asynccontextmanager
async def enforce_attempt_deadlines(policy: Timeout | None) -> AsyncIterator[None]:
    if policy is None:
        yield
        return
    attempt = _AttemptDeadline(policy)
    token = _current_attempt.set(attempt)
    try:
        try:
            async with attempt.timer:
                yield
                if attempt.now() >= attempt.deadline_at:
                    raise LlmAttemptTimeoutError(attempt)
        except TimeoutError as exc:
            if attempt.timer.expired():
                raise LlmAttemptTimeoutError(attempt) from exc
            raise
    finally:
        _current_attempt.reset(token)
