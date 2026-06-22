from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import harnyx_miner.miner_config as config_module


class _FakeHotkey:
    ss58_address = "5FakeMinerHotkey"

    def __init__(self) -> None:
        self.signed: list[bytes] = []

    def sign(self, payload: bytes) -> bytes:
        self.signed.append(payload)
        return b"\xab" * 64


class _FakeClient:
    def __init__(self, *, response: httpx.Response, requests: list[dict[str, Any]], **_: object) -> None:
        self._response = response
        self._requests = requests

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def request(self, method: str, path: str, *, headers: dict[str, str], content: bytes = b"") -> httpx.Response:
        self._requests.append(
            {
                "method": method,
                "path": path,
                "headers": headers,
                "content": content,
            }
        )
        return self._response


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: httpx.Response | None = None,
) -> tuple[_FakeHotkey, list[dict[str, Any]]]:
    hotkey = _FakeHotkey()
    requests: list[dict[str, Any]] = []
    monkeypatch.setenv("PLATFORM_BASE_URL", "https://platform.example.com")
    monkeypatch.setattr(config_module.bt, "Wallet", lambda name, hotkey: SimpleNamespace(hotkey=_FakeHotkey()))

    def _wallet(name: str, hotkey: str) -> SimpleNamespace:
        _ = (name, hotkey)
        return SimpleNamespace(hotkey=hotkey_obj)

    hotkey_obj = hotkey
    monkeypatch.setattr(config_module.bt, "Wallet", _wallet)
    monkeypatch.setattr(
        config_module.httpx,
        "Client",
        lambda **kwargs: _FakeClient(
            response=response or httpx.Response(200, json={"ok": True}),
            requests=requests,
            **kwargs,
        ),
    )
    return hotkey, requests


def test_get_config_signs_empty_body_request(monkeypatch: pytest.MonkeyPatch) -> None:
    hotkey, requests = _install_fakes(monkeypatch, response=httpx.Response(200, json={"provider_credentials": {}}))

    assert config_module.get_config(wallet_name="wallet", hotkey_name="hotkey") == {"provider_credentials": {}}

    assert requests == [
        {
            "method": "GET",
            "path": "/v1/miner-config",
            "headers": {
                "Authorization": 'Bittensor ss58="5FakeMinerHotkey",sig="' + "ab" * 64 + '"',
            },
            "content": b"",
        }
    ]
    assert hotkey.signed[0].startswith(b"GET\n/v1/miner-config\n")


def test_put_provider_credential_sends_supported_provider_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    hotkey, requests = _install_fakes(monkeypatch)

    config_module.put_provider_credential(
        provider=" Chutes ",
        api_key="secret-provider-key",
        wallet_name="wallet",
        hotkey_name="hotkey",
    )

    assert requests[0]["method"] == "PUT"
    assert requests[0]["path"] == "/v1/miner-config"
    assert requests[0]["headers"]["Content-Type"] == "application/json"
    assert json.loads(requests[0]["content"]) == {
        "key": "provider_credentials.chutes",
        "value": "secret-provider-key",
    }
    assert hotkey.signed[0].startswith(b"PUT\n/v1/miner-config\n")


def test_delete_provider_credential_sends_key_only_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _hotkey, requests = _install_fakes(monkeypatch)

    config_module.delete_provider_credential(
        provider="parallel",
        wallet_name="wallet",
        hotkey_name="hotkey",
    )

    assert requests[0]["method"] == "DELETE"
    assert json.loads(requests[0]["content"]) == {"key": "provider_credentials.parallel"}


def test_put_task_retry_count_sends_supported_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _hotkey, requests = _install_fakes(monkeypatch)

    config_module.put_task_retry_count(
        task_retry_count=3,
        wallet_name="wallet",
        hotkey_name="hotkey",
    )

    assert requests[0]["method"] == "PUT"
    assert requests[0]["path"] == "/v1/miner-config"
    assert requests[0]["headers"]["Content-Type"] == "application/json"
    assert json.loads(requests[0]["content"]) == {
        "key": "task_retry_count",
        "value": "3",
    }


def test_provider_credential_cli_rejects_unknown_provider_and_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    with pytest.raises(ValueError, match="unsupported provider"):
        config_module.put_provider_credential(
            provider="unknown",
            api_key="secret",
            wallet_name="wallet",
            hotkey_name="hotkey",
        )

    with pytest.raises(ValueError, match="provider API key must be non-empty"):
        config_module.put_provider_credential(
            provider="chutes",
            api_key="   ",
            wallet_name="wallet",
            hotkey_name="hotkey",
        )


def test_task_retry_count_cli_rejects_out_of_range_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    with pytest.raises(ValueError, match="task retry count must be between 0 and 3"):
        config_module.put_task_retry_count(
            task_retry_count=4,
            wallet_name="wallet",
            hotkey_name="hotkey",
        )


def test_provider_credential_failures_do_not_echo_submitted_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(
        monkeypatch,
        response=httpx.Response(409, text="server echoed secret-provider-key"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        config_module.put_provider_credential(
            provider="chutes",
            api_key="secret-provider-key",
            wallet_name="wallet",
            hotkey_name="hotkey",
        )

    assert "secret-provider-key" not in str(exc_info.value)
