from __future__ import annotations

import bittensor as bt
import pytest

import caster_validator.infrastructure.auth.sr25519 as sr25519
from caster_commons.bittensor import VerificationError, build_canonical_request
from caster_validator.infrastructure.auth.sr25519 import BittensorSr25519InboundVerifier
from caster_validator.runtime import bootstrap
from caster_validator.runtime.settings import Settings


def test_build_inbound_auth_uses_subnet_owner_hotkey(monkeypatch) -> None:
    expected_owner_coldkey_ss58 = "5DdemoOwnerColdkey"
    captured: dict[str, object] = {}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            captured["network"] = network

        class _SubnetInfo:
            def __init__(self, owner_ss58: str) -> None:
                self.owner_ss58 = owner_ss58

        def get_subnet_info(self, netuid: int) -> _SubnetInfo:
            captured["netuid"] = netuid
            return self._SubnetInfo(expected_owner_coldkey_ss58)

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    verifier = bootstrap._build_inbound_auth(settings)

    assert verifier.owner_coldkey_ss58 == expected_owner_coldkey_ss58
    assert captured["network"] == settings.subtensor.endpoint
    assert captured["netuid"] == settings.subtensor.netuid
    assert captured["closed"] is True


def test_build_inbound_auth_raises_when_owner_hotkey_missing(monkeypatch) -> None:
    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_subnet_info(self, netuid: int):
            return None

        def close(self) -> None:
            return None

    monkeypatch.setenv("SUBTENSOR_ENDPOINT", "ws://127.0.0.1:9945")
    monkeypatch.setenv("SUBTENSOR_NETUID", "2")
    monkeypatch.setattr(bootstrap.bt, "Subtensor", FakeSubtensor)

    settings = Settings()
    with pytest.raises(RuntimeError, match="subnet info"):
        bootstrap._build_inbound_auth(settings)


def test_inbound_verifier_rejects_non_owner_hotkey(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    canonical = build_canonical_request("GET", "/v1/test", b"")
    signature = keypair.sign(canonical)
    header = f'Bittensor ss58="{keypair.ss58_address}",sig="{signature.hex()}"'

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_hotkey_owner(self, hotkey_ss58: str):
            return "5NotOwnerColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    with pytest.raises(VerificationError, match="subnet owner coldkey"):
        verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
