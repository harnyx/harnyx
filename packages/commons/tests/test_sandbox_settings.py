from __future__ import annotations

from harnyx_commons.config.sandbox import load_sandbox_settings


def test_load_sandbox_settings_uses_sandbox_env_names(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_IMAGE", "neutral-sandbox:latest")
    monkeypatch.setenv("SANDBOX_NETWORK", "neutral-sandbox-net")
    monkeypatch.setenv("SANDBOX_PULL_POLICY", "missing")

    settings = load_sandbox_settings()

    assert settings.sandbox_image == "neutral-sandbox:latest"
    assert settings.sandbox_network == "neutral-sandbox-net"
    assert settings.sandbox_pull_policy == "missing"
