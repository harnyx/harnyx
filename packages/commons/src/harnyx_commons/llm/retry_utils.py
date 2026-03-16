"""Shared retry helpers for LLM providers and tool clients."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int
    initial_ms: int
    max_ms: int
    jitter: float  # fraction of backoff to add/subtract


def backoff_ms(attempt: int, policy: RetryPolicy) -> int:
    """Exponential backoff with jitter (attempt is zero-based)."""
    expo = policy.initial_ms * math.pow(2, attempt)
    capped = min(expo, policy.max_ms)
    jitter_span = capped * policy.jitter
    return int(max(0, capped + random.uniform(-jitter_span, jitter_span)))  # noqa: S311 - non-crypto backoff jitter


__all__ = ["RetryPolicy", "backoff_ms"]
