"""Safe public summaries for LLM provider failures."""

from __future__ import annotations

import re

_SAFE_PROVIDER_FAILURE_REASONS = frozenset(
    {
        "empty_choices",
        "empty_output",
        "tool_call_args_invalid_json",
    }
)
_HTTP_REASON_RE = re.compile(r"^(http_(?P<status>[1-5][0-9][0-9]))(?::|$)")


def public_llm_provider_failure_summary(reason: str) -> str | None:
    """Return a public-safe high-level provider failure reason."""
    normalized = " ".join(reason.split())
    if normalized in _SAFE_PROVIDER_FAILURE_REASONS:
        return normalized
    match = _HTTP_REASON_RE.match(normalized)
    if match is not None:
        return f"http_{match.group('status')}"
    return None


__all__ = ["public_llm_provider_failure_summary"]
