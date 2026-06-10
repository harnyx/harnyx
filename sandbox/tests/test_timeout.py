from __future__ import annotations

import harnyx_sandbox.sandbox.timeout as timeout_module
import pytest

from harnyx_miner_sdk.tools.proxy import PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS


def test_default_entrypoint_timeout_is_five_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(timeout_module.ENTRYPOINT_TIMEOUT_ENV_VAR, raising=False)

    assert timeout_module.load_entrypoint_timeout_seconds() == PLATFORM_TOOL_PROXY_SANDBOX_REQUEST_TIMEOUT_SECONDS


def test_entrypoint_timeout_override_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(timeout_module.ENTRYPOINT_TIMEOUT_ENV_VAR, "5")

    assert timeout_module.load_entrypoint_timeout_seconds() == 5.0


def test_entrypoint_timeout_invalid_values_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(timeout_module.ENTRYPOINT_TIMEOUT_ENV_VAR, "0")

    with pytest.raises(ValueError, match="must be > 0"):
        timeout_module.load_entrypoint_timeout_seconds()
