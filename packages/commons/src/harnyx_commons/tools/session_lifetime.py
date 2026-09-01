"""Tool-session authorization lifetime policy."""

from __future__ import annotations

import math
from datetime import timedelta

_TOOL_SESSION_EXPIRY_GRACE_SECONDS = 60.0


def tool_session_ttl_for_execution_limit(execution_time_limit_seconds: float) -> timedelta:
    """Return the authorization lifetime required for one invocation limit."""
    if isinstance(execution_time_limit_seconds, bool) or not isinstance(execution_time_limit_seconds, int | float):
        raise TypeError("execution_time_limit_seconds must be a finite number")
    if not math.isfinite(execution_time_limit_seconds) or execution_time_limit_seconds <= 0:
        raise ValueError("execution_time_limit_seconds must be finite and greater than zero")
    return timedelta(seconds=execution_time_limit_seconds + _TOOL_SESSION_EXPIRY_GRACE_SECONDS)


__all__ = ["tool_session_ttl_for_execution_limit"]
