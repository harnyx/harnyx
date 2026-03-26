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


def test_resolve_registration_metadata_allows_missing_registry_digest(monkeypatch) -> None:
    monkeypatch.setattr(metadata_mod, "_inspect_current_image_id", lambda: "sha256:local")
    monkeypatch.setattr(metadata_mod, "_inspect_registry_digest", lambda _: None)

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.local_image_id == "sha256:local"
    assert metadata.registry_digest is None


def test_resolve_registration_metadata_allows_missing_image_identity(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("SOURCE_REVISION", "abc123")
    monkeypatch.setattr(
        metadata_mod,
        "_inspect_current_image_id",
        lambda: (_ for _ in ()).throw(RuntimeError("failed to resolve current validator container id")),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_validator.runtime.registration")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.source_revision == "abc123"
    assert metadata.local_image_id is None
    assert metadata.registry_digest is None
    assert "validator registration image inspection unavailable" in caplog.text


def test_resolve_registration_metadata_allows_missing_docker_cli(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOSTNAME", "validator-host")
    monkeypatch.setenv("SOURCE_REVISION", "abc123")
    monkeypatch.setattr(metadata_mod, "_resolve_current_container_id_from_mountinfo", lambda: None)
    monkeypatch.setattr(
        metadata_mod.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("docker")),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_validator.runtime.registration")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.source_revision == "abc123"
    assert metadata.local_image_id is None
    assert metadata.registry_digest is None
    assert "validator registration image inspection unavailable" in caplog.text


def test_resolve_registration_metadata_allows_registry_digest_inspection_failure(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(metadata_mod, "_inspect_current_image_id", lambda: "sha256:local")
    monkeypatch.setattr(
        metadata_mod,
        "_inspect_registry_digest",
        lambda _image_id: (_ for _ in ()).throw(RuntimeError("docker image inspect failed")),
    )
    caplog.set_level(logging.WARNING, logger="harnyx_validator.runtime.registration")

    metadata = resolve_validator_registration_metadata()

    assert metadata.validator_version == VALIDATOR_RELEASE_VERSION
    assert metadata.local_image_id == "sha256:local"
    assert metadata.registry_digest is None
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
