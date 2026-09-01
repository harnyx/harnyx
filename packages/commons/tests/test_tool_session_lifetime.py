from __future__ import annotations

from datetime import timedelta

import pytest

from harnyx_commons.tools.session_lifetime import tool_session_ttl_for_execution_limit


def test_tool_session_ttl_adds_expiry_grace_to_execution_limit() -> None:
    assert tool_session_ttl_for_execution_limit(3_600.0) == timedelta(seconds=3_660)


@pytest.mark.parametrize("value", [True, "300"])
def test_tool_session_ttl_rejects_non_numeric_execution_limit(value: object) -> None:
    with pytest.raises(TypeError, match="must be a finite number"):
        tool_session_ttl_for_execution_limit(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_tool_session_ttl_rejects_non_positive_or_non_finite_execution_limit(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and greater than zero"):
        tool_session_ttl_for_execution_limit(value)
