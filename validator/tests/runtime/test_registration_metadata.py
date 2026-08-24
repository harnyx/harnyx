from __future__ import annotations

import logging

import pytest

import harnyx_validator.runtime.registration_metadata as metadata_mod
from harnyx_validator.runtime.registration_metadata import resolve_validator_registration_metadata
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION


def test_resolve_registration_metadata_reads_version_revision_and_image_identity(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_REVISION", "abc123")
    monkeypatch.setattr(metadata_mod, "_inspect_current_image_id", lambda: "sha256:local")
    monkeypatch.setattr(metadata_mod, "_inspect_registry_digest", lambda _: "sha256:registry")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.source_revision == "abc123"
    assert metadata.registry_digest == "sha256:registry"
    assert metadata.local_image_id == "sha256:local"


def test_resolve_registration_metadata_degrades_when_image_inspection_fails(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("SOURCE_REVISION", "abc123")
    monkeypatch.setattr(
        metadata_mod,
        "_inspect_current_image_id",
        lambda: (_ for _ in ()).throw(RuntimeError("docker unavailable")),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_validator.runtime.registration")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.source_revision == "abc123"
    assert metadata.local_image_id is None
    assert metadata.registry_digest is None
    assert "validator registration image inspection unavailable" in caplog.text


@pytest.mark.parametrize("registry_outcome", ["missing", "error"])
def test_resolve_registration_metadata_keeps_local_identity_when_registry_digest_is_unavailable(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
    registry_outcome: str,
) -> None:
    monkeypatch.setattr(metadata_mod, "_inspect_current_image_id", lambda: "sha256:local")

    def inspect_registry_digest(_image_id: str) -> str | None:
        if registry_outcome == "error":
            raise RuntimeError("registry inspection failed")
        return None

    monkeypatch.setattr(metadata_mod, "_inspect_registry_digest", inspect_registry_digest)
    caplog.set_level(logging.WARNING, logger="harnyx_validator.runtime.registration")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.local_image_id == "sha256:local"
    assert metadata.registry_digest is None
    if registry_outcome == "error":
        assert "validator registration registry digest inspection unavailable" in caplog.text


def test_inspect_current_image_id_uses_mountinfo_container_when_hostname_is_stale(monkeypatch) -> None:
    monkeypatch.setenv("HOSTNAME", "stale-hostname")
    monkeypatch.setattr(metadata_mod, "_resolve_current_container_id_from_mountinfo", lambda: "abc123def456")
    seen: list[str] = []

    def _record_container(container: str) -> str:
        seen.append(container)
        return "sha256:local"

    monkeypatch.setattr(metadata_mod, "_inspect_container_image_id", _record_container)

    image_id = metadata_mod._inspect_current_image_id()

    assert image_id == "sha256:local"
    assert seen == ["abc123def456"]


def test_inspect_current_image_id_requires_container_identity(monkeypatch) -> None:
    monkeypatch.delenv("HOSTNAME", raising=False)
    monkeypatch.setattr(metadata_mod, "_resolve_current_container_id_from_mountinfo", lambda: None)

    with pytest.raises(RuntimeError, match="failed to resolve current validator container id"):
        metadata_mod._inspect_current_image_id()
