from __future__ import annotations

import json
import re

import bittensor as bt
import httpx

import harnyx_validator.infrastructure.platform.registration_client as registration_module
from harnyx_commons.bittensor import build_canonical_request
from harnyx_validator.application.dto.registration import ValidatorRegistrationMetadata
from harnyx_validator.infrastructure.platform.registration_client import (
    PlatformRegistrationClient,
    register_with_retry,
)
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

_HEADER_PATTERN = re.compile(
    r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$'
)


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _assert_signed_body(request: httpx.Request, keypair: bt.Keypair) -> None:
    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address
    canonical = build_canonical_request(request.method, request.url.path, request.content)
    assert keypair.verify(canonical, bytes.fromhex(match.group("sig")))


def test_registration_client_posts_runtime_metadata_in_signed_body(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            self._base_url = base_url
            self._timeout = timeout

        def __enter__(self) -> _StubClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, path: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
            captured["request"] = httpx.Request(
                "POST",
                f"{self._base_url}{path}",
                headers=headers,
                content=content,
            )
            return httpx.Response(status_code=200, request=captured["request"])

    monkeypatch.setattr(registration_module.httpx, "Client", _StubClient)
    keypair = _keypair()
    metadata = ValidatorRegistrationMetadata(
        validator_version=VALIDATOR_RELEASE_VERSION,
        source_revision="abc123",
        registry_digest="sha256:registry",
        local_image_id="sha256:local",
    )
    client = PlatformRegistrationClient(
        platform_base_url="https://platform.invalid",
        hotkey=keypair,
    )

    client.register("https://validator.invalid", metadata)

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    _assert_signed_body(request, keypair)
    assert json.loads(request.content) == {
        "base_url": "https://validator.invalid",
        "validator_version": VALIDATOR_RELEASE_VERSION,
        "source_revision": "abc123",
        "registry_digest": "sha256:registry",
        "local_image_id": "sha256:local",
    }


def test_register_with_retry_forwards_metadata(monkeypatch) -> None:
    metadata = ValidatorRegistrationMetadata(
        validator_version=VALIDATOR_RELEASE_VERSION,
        source_revision=None,
        registry_digest=None,
        local_image_id=None,
    )
    calls: list[tuple[str, ValidatorRegistrationMetadata]] = []

    class _RecordingClient:
        platform_base_url = "https://platform.invalid"

        def register(self, public_url: str, metadata_arg: ValidatorRegistrationMetadata) -> None:
            calls.append((public_url, metadata_arg))

    monkeypatch.setattr(registration_module, "_log_platform_resolution", lambda _base_url: None)
    register_with_retry(
        _RecordingClient(),  # type: ignore[arg-type]
        "https://validator.invalid",
        metadata=metadata,
        attempts=1,
        delay_seconds=0.0,
    )

    assert calls == [("https://validator.invalid", metadata)]
