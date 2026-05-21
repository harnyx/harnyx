from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

import bittensor as bt
import pytest

import harnyx_validator.infrastructure.auth.sr25519 as sr25519
from harnyx_commons.bittensor import VerificationError, build_canonical_request
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.status import StatusProvider
from harnyx_validator.infrastructure.auth.sr25519 import BittensorSr25519InboundVerifier
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings


@dataclass(frozen=True)
class _StubHotkey:
    ss58_address: str = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return payload


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


def _build_signed_header(
    *,
    keypair: bt.Keypair,
    method: str = "GET",
    path_qs: str = "/v1/test",
    body: bytes = b"",
) -> str:
    canonical = build_canonical_request(method, path_qs, body)
    signature = keypair.sign(canonical)
    return f'Bittensor ss58="{keypair.ss58_address}",sig="{signature.hex()}"'


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
    header = _build_signed_header(keypair=keypair)
    warmup_done = Event()

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            warmup_done.set()
            return ("5different-hotkey",)

        def get_hotkey_owner(self, hotkey_ss58: str):
            assert hotkey_ss58 == keypair.ss58_address
            return "5OtherColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    verifier.start()
    try:
        assert warmup_done.wait(timeout=1.0) is True
        with pytest.raises(VerificationError, match="subnet owner coldkey"):
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
    finally:
        verifier.stop(timeout_seconds=1.0)


def test_inbound_verifier_rejects_requests_before_initial_warmup() -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )

    with pytest.raises(VerificationError, match="initial hotkey warmup"):
        verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)


def test_inbound_verifier_start_does_not_raise_when_initial_refresh_fails(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)
    refresh_failed = Event()

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            refresh_failed.set()
            raise RuntimeError("subtensor unavailable")

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=9999.0,
    )
    verifier.start()
    try:
        assert refresh_failed.wait(timeout=1.0) is True
        with pytest.raises(VerificationError, match="initial hotkey warmup"):
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


def test_inbound_verifier_start_warms_authorized_hotkeys_and_verify_uses_memory(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)
    calls: dict[str, int] = {"count": 0}
    warmup_done = Event()

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            calls["count"] += 1
            warmup_done.set()
            return (keypair.ss58_address,)

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=9999.0,
    )
    verifier.start()
    try:
        assert warmup_done.wait(timeout=1.0) is True
        assert (
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
            == keypair.ss58_address
        )
        assert (
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
            == keypair.ss58_address
        )
        assert calls["count"] == 1
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


def test_inbound_verifier_falls_back_to_hotkey_owner_lookup_on_cache_miss(monkeypatch) -> None:
    old_keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    rotated_keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    rotated_header = _build_signed_header(keypair=rotated_keypair)
    calls: dict[str, int] = {"owned_hotkeys": 0, "hotkey_owner": 0}
    warmup_done = Event()

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            calls["owned_hotkeys"] += 1
            warmup_done.set()
            return (old_keypair.ss58_address,)

        def get_hotkey_owner(self, hotkey_ss58: str):
            assert hotkey_ss58 == rotated_keypair.ss58_address
            calls["hotkey_owner"] += 1
            return "5OwnerColdkey"

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=9999.0,
    )
    verifier.start()
    try:
        assert warmup_done.wait(timeout=1.0) is True
        assert (
            verifier.verify(
                method="GET",
                path_qs="/v1/test",
                body=b"",
                authorization_header=rotated_header,
            )
            == rotated_keypair.ss58_address
        )
        assert (
            verifier.verify(
                method="GET",
                path_qs="/v1/test",
                body=b"",
                authorization_header=rotated_header,
            )
            == rotated_keypair.ss58_address
        )
        assert calls == {"owned_hotkeys": 1, "hotkey_owner": 1}
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


def test_inbound_verifier_cache_miss_rejects_unknown_hotkey(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)
    warmup_done = Event()

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            warmup_done.set()
            return ()

        def get_hotkey_owner(self, hotkey_ss58: str):
            assert hotkey_ss58 == keypair.ss58_address
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=9999.0,
    )
    verifier.start()
    try:
        assert warmup_done.wait(timeout=1.0) is True
        with pytest.raises(VerificationError, match="hotkey owner not found on chain") as exc_info:
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
        assert exc_info.value.code == "unknown_hotkey"
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


def test_inbound_verifier_retries_failed_initial_refresh_quickly(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)
    refresh_attempted = Event()
    refresh_recovered = Event()
    calls = {"count": 0}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            calls["count"] += 1
            if calls["count"] == 1:
                refresh_attempted.set()
                raise RuntimeError("subtensor unavailable")
            refresh_recovered.set()
            return (keypair.ss58_address,)

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519, "_FAILED_REFRESH_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=9999.0,
    )
    verifier.start()
    try:
        assert refresh_attempted.wait(timeout=1.0) is True
        assert refresh_recovered.wait(timeout=1.0) is True
        assert (
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
            == keypair.ss58_address
        )
        assert calls["count"] >= 2
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


def test_inbound_verifier_refresh_failure_keeps_last_known_hotkeys(monkeypatch) -> None:
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    header = _build_signed_header(keypair=keypair)
    refresh_failed = Event()
    calls: dict[str, int] = {"count": 0}

    class FakeSubtensor:
        def __init__(self, *, network: str) -> None:
            self.network = network

        def get_owned_hotkeys(self, coldkey_ss58: str):
            assert coldkey_ss58 == "5OwnerColdkey"
            calls["count"] += 1
            if calls["count"] == 1:
                return (keypair.ss58_address,)
            refresh_failed.set()
            raise RuntimeError("subtensor unavailable")

        def close(self) -> None:
            return None

    monkeypatch.setattr(sr25519.bt, "Subtensor", FakeSubtensor)

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
        refresh_interval_seconds=0.01,
    )
    verifier.start()
    try:
        assert refresh_failed.wait(timeout=1.0) is True
        assert (
            verifier.verify(method="GET", path_qs="/v1/test", body=b"", authorization_header=header)
            == keypair.ss58_address
        )
        assert calls["count"] >= 2
    finally:
        assert verifier.stop(timeout_seconds=1.0) is True


@pytest.mark.anyio
async def test_control_provider_offloads_auth_verification_to_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_verify_request(
        verifier: object,
        *,
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        observed["verifier"] = verifier
        observed["method"] = method
        observed["path_qs"] = path_qs
        observed["body"] = body
        observed["authorization_header"] = authorization_header
        return "5validator"

    async def fake_to_thread(func, /, *args, **kwargs):
        observed["to_thread_func"] = func
        observed["to_thread_args"] = args
        observed["to_thread_kwargs"] = kwargs
        return func(*args, **kwargs)

    inbound_auth = object()
    monkeypatch.setattr(bootstrap, "_verify_request", fake_verify_request)
    monkeypatch.setattr(bootstrap.asyncio, "to_thread", fake_to_thread)

    deps = bootstrap._make_control_provider(
        AcceptEvaluationBatch(InMemoryBatchInbox(), StatusProvider(), _progress(tmp_path)),
        StatusProvider(),
        inbound_auth,
        _progress(tmp_path),
        _StubHotkey(),
    )()

    result = await deps.auth("GET", "/validator/status", b"body", "Bittensor test")

    assert result == "5validator"
    assert observed["to_thread_func"] is fake_verify_request
    assert observed["to_thread_args"] == (inbound_auth,)
    assert observed["to_thread_kwargs"] == {
        "method": "GET",
        "path_qs": "/validator/status",
        "body": b"body",
        "authorization_header": "Bittensor test",
    }


def test_inbound_verifier_stop_timeout_returns_false() -> None:
    class StuckThread:
        def join(self, timeout: float) -> None:
            assert timeout == 1.0

        def is_alive(self) -> bool:
            return True

    verifier = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    verifier._refresh_thread = StuckThread()  # type: ignore[assignment]

    assert verifier.stop(timeout_seconds=1.0) is False


@pytest.mark.anyio
async def test_make_control_provider_verifies_request_inline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    verify_calls: list[tuple[object, dict[str, object]]] = []

    def _record_verify_request(verifier: object, **kwargs: object) -> str:
        verify_calls.append((verifier, kwargs))
        return "caller"

    monkeypatch.setattr(bootstrap, "_verify_request", _record_verify_request)

    progress_tracker = _progress(tmp_path)
    status_provider = StatusProvider()
    accept_batch = AcceptEvaluationBatch(
        inbox=InMemoryBatchInbox(),
        status=status_provider,
        progress=progress_tracker,
    )
    inbound_auth = BittensorSr25519InboundVerifier(
        netuid=2,
        network="ws://127.0.0.1:9945",
        owner_coldkey_ss58="5OwnerColdkey",
    )
    deps = bootstrap._make_control_provider(
        accept_batch=accept_batch,
        status_provider=status_provider,
        inbound_auth=inbound_auth,
        progress_tracker=progress_tracker,
        validator_hotkey=_StubHotkey(),
    )()

    caller = await deps.auth(
        "GET",
        "/validator/status?verbose=1",
        b"",
        'Bittensor ss58="5demo",sig="00"',
    )

    assert caller == "caller"
    assert verify_calls == [
        (
            inbound_auth,
            {
                "method": "GET",
                "path_qs": "/validator/status?verbose=1",
                "body": b"",
                "authorization_header": 'Bittensor ss58="5demo",sig="00"',
            },
        )
    ]


def test_make_control_provider_reuses_fallback_resource_usage_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[object] = []

    class _StubResourceUsageProvider:
        def __init__(self) -> None:
            created.append(self)

    monkeypatch.setattr(bootstrap, "ValidatorResourceUsageProvider", _StubResourceUsageProvider)

    provider = bootstrap._make_control_provider(
        AcceptEvaluationBatch(InMemoryBatchInbox(), StatusProvider(), _progress(tmp_path)),
        StatusProvider(),
        object(),
        _progress(tmp_path),
        _StubHotkey(),
    )

    first = provider()
    second = provider()

    assert len(created) == 1
    assert first.resource_usage_provider is second.resource_usage_provider
