"""Shared state directory conventions for sandbox execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

from harnyx_commons.sandbox.options import ReadonlyVolumeSubpathMount

DEFAULT_STATE_DIR: Final[str] = "/workspace/.harnyx_state"
_STATE_MOUNT_SOURCE_ENV: Final[str] = "STATE_VOLUME_NAME"


@dataclass(frozen=True, slots=True)
class ArtifactStateMount:
    """Docker mounts that expose only one staged artifact directory."""

    bind_volumes: tuple[tuple[str, str, str | None], ...] = field(default_factory=tuple)
    volume_subpaths: tuple[ReadonlyVolumeSubpathMount, ...] = field(default_factory=tuple)


def default_state_dir_path() -> Path:
    return Path(DEFAULT_STATE_DIR)


def resolve_state_mount_source() -> str:
    """Return the docker mount source for the shared state directory.

    In DinD/Kubernetes this should be a path (default: `/workspace/.harnyx_state`) mounted into the
    dockerd container. In Docker Compose (docker socket) this can be a named volume.
    """

    return (
        os.getenv(_STATE_MOUNT_SOURCE_ENV)
        or DEFAULT_STATE_DIR
    )


def default_state_volumes(*, mode: str | None = "ro") -> tuple[tuple[str, str, str | None], ...]:
    """Return a default sandbox volume mount for the shared state directory."""

    return ((resolve_state_mount_source(), DEFAULT_STATE_DIR, mode),)


def resolve_artifact_state_mount(
    *,
    artifact_host_path: Path,
    state_dir: Path,
    artifact_container_path: str,
    mount_source: str | None = None,
) -> ArtifactStateMount:
    """Resolve a read-only mount for the staged artifact's directory only."""

    relative_directory = artifact_host_path.parent.relative_to(state_dir)
    container_directory = PurePosixPath(artifact_container_path).parent
    if not container_directory.is_absolute():
        raise ValueError("artifact container path must be absolute")
    source = mount_source or resolve_state_mount_source()
    if Path(source).is_absolute():
        source_directory = Path(source).joinpath(*relative_directory.parts)
        return ArtifactStateMount(
            bind_volumes=((str(source_directory), container_directory.as_posix(), "ro"),),
        )
    return ArtifactStateMount(
        volume_subpaths=(
            ReadonlyVolumeSubpathMount(
                source=source,
                subpath=PurePosixPath(*relative_directory.parts).as_posix(),
                target=container_directory.as_posix(),
            ),
        ),
    )


__all__ = [
    "ArtifactStateMount",
    "DEFAULT_STATE_DIR",
    "default_state_dir_path",
    "default_state_volumes",
    "resolve_artifact_state_mount",
    "resolve_state_mount_source",
]
