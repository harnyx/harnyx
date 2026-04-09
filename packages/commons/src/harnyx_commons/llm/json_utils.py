"""Helpers for coercing model output into JSON and Pydantic objects."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from harnyx_commons.llm.schema import LlmResponse, PostprocessRecovery, PostprocessResult

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")


def _strip_wrappers(text: str) -> str:
    cleaned = _ZERO_WIDTH_RE.sub("", text.strip())
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _extract_balanced_braces(text: str) -> str:
    """Return first balanced {...} block or original text if none found."""
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1 if depth > 0 else 0
            if depth == 0 and start is not None:
                return text[start : idx + 1]
    return text


def coerce_json(text: str) -> tuple[bool, object | str]:
    _, candidate = _prepare_json_candidate(text)
    try:
        return True, json.loads(candidate)
    except json.JSONDecodeError as exc:
        return False, f"json decode error: {exc}"


def _prepare_json_candidate(text: str) -> tuple[str, str]:
    cleaned = _strip_wrappers(text)
    candidate = _extract_balanced_braces(cleaned)
    return cleaned, candidate


def _build_recovery(
    *,
    reason: str,
) -> PostprocessRecovery | None:
    from harnyx_commons.llm.schema import PostprocessRecovery

    return PostprocessRecovery(
        kind="retry_with_feedback",
        failure_reason=reason,
    )


def pydantic_postprocessor(model: type[BaseModel]) -> Callable[[LlmResponse], PostprocessResult]:
    """Build a postprocessor that parses first-choice text into a Pydantic model."""

    def _postprocess(response: LlmResponse) -> PostprocessResult:
        from harnyx_commons.llm.schema import PostprocessResult

        text = response.raw_text or ""
        ok, data = coerce_json(text)
        if not ok:
            reason = str(data)
            return PostprocessResult(
                ok=False,
                retryable=True,
                reason=reason,
                processed=None,
                recovery=_build_recovery(reason=reason),
            )
        if not isinstance(data, dict):
            reason = "json not an object"
            return PostprocessResult(
                ok=False,
                retryable=True,
                reason=reason,
                processed=None,
                recovery=_build_recovery(reason=reason),
            )
        try:
            parsed = model.model_validate(data)
        except ValidationError as exc:
            reason = str(exc)
            return PostprocessResult(
                ok=False,
                retryable=True,
                reason=reason,
                processed=None,
                recovery=_build_recovery(reason=reason),
            )
        return PostprocessResult(ok=True, retryable=False, reason=None, processed=parsed)

    return _postprocess


__all__ = ["coerce_json", "pydantic_postprocessor"]
