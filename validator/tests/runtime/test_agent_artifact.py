from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from harnyx_commons.sandbox.agent_staging import MAX_AGENT_BYTES
from harnyx_validator.application.dto.evaluation import ScriptArtifactSpec
from harnyx_validator.infrastructure.tools.platform_client import PlatformClientError
from harnyx_validator.runtime import agent_artifact as agent_artifact_mod
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError, resolve_platform_agent_spec


class _FlakyPlatformClient:
    def __init__(self, *, failures_before_success: int, data: bytes) -> None:
        self.failures_before_success = failures_before_success
        self.data = data
        self.calls = 0

    def fetch_artifact(self, _batch_id: UUID, _artifact_id: UUID) -> bytes:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise PlatformClientError(status_code=500, message="platform returned 500 for GET /artifact")
        return self.data


class _StaticPlatformClient:
    def __init__(self, *, data: bytes) -> None:
        self.data = data
        self.calls = 0

    def fetch_artifact(self, _batch_id: UUID, _artifact_id: UUID) -> bytes:
        self.calls += 1
        return self.data


def _artifact(*, content_hash: str, size_bytes: int) -> ScriptArtifactSpec:
    return ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash=content_hash,
        size_bytes=size_bytes,
    )


def test_resolve_platform_agent_spec_retries_transient_fetch_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"print('ok')\n"
    content_hash = hashlib.sha256(data).hexdigest()
    artifact = _artifact(content_hash=content_hash, size_bytes=len(data))
    client = _FlakyPlatformClient(failures_before_success=2, data=data)
    batch_id = uuid4()
    sleeps: list[float] = []
    monkeypatch.setattr(agent_artifact_mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    resolved = resolve_platform_agent_spec(
        batch_id=batch_id,
        artifact=artifact,
        platform_client=client,
        state_dir=tmp_path,
        container_root="/sandbox/state",
    )

    assert client.calls == 3
    assert sleeps == [0.25, 0.5]
    assert resolved.content_hash == content_hash
    assert resolved.host_path.exists()


def test_resolve_platform_agent_spec_does_not_retry_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"print('bad-hash')\n"
    artifact = _artifact(content_hash="expected-hash", size_bytes=len(data))
    client = _StaticPlatformClient(data=data)
    batch_id = uuid4()

    def _unexpected_sleep(_seconds: float) -> None:
        raise AssertionError("deterministic artifact failures must not sleep/retry")

    monkeypatch.setattr(agent_artifact_mod.time, "sleep", _unexpected_sleep)

    with pytest.raises(ArtifactPreparationError, match="sha256 mismatch") as exc_info:
        resolve_platform_agent_spec(
            batch_id=batch_id,
            artifact=artifact,
            platform_client=client,
            state_dir=tmp_path,
            container_root="/sandbox/state",
        )

    assert exc_info.value.error_code == "artifact_hash_mismatch"
    assert client.calls == 1


def test_resolve_platform_agent_spec_maps_oversized_source_to_script_validation_failed(
    tmp_path: Path,
) -> None:
    data = b"x" * (MAX_AGENT_BYTES + 1)
    artifact = _artifact(content_hash=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    client = _StaticPlatformClient(data=data)

    with pytest.raises(ArtifactPreparationError, match="exceeds size limit") as exc_info:
        resolve_platform_agent_spec(
            batch_id=uuid4(),
            artifact=artifact,
            platform_client=client,
            state_dir=tmp_path,
            container_root="/sandbox/state",
        )

    assert exc_info.value.error_code == "script_validation_failed"


def test_resolve_platform_agent_spec_maps_syntax_error_to_script_validation_failed(
    tmp_path: Path,
) -> None:
    data = b"def broken(:\n"
    artifact = _artifact(content_hash=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    client = _StaticPlatformClient(data=data)

    with pytest.raises(ArtifactPreparationError, match="failed bytecode compilation") as exc_info:
        resolve_platform_agent_spec(
            batch_id=uuid4(),
            artifact=artifact,
            platform_client=client,
            state_dir=tmp_path,
            container_root="/sandbox/state",
        )

    assert exc_info.value.error_code == "script_validation_failed"
