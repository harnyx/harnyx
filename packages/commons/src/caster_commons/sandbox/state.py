"""Shared state directory conventions for sandbox execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

DEFAULT_STATE_DIR: Final[str] = "/workspace/.caster_state"
_STATE_MOUNT_SOURCE_ENV: Final[str] = "CASTER_STATE_VOLUME_NAME"


def default_state_dir_path() -> Path:
    return Path(DEFAULT_STATE_DIR)


def resolve_state_mount_source() -> str:
    """Return the docker mount source for the shared state directory.

    In DinD/Kubernetes this should be a path (default: `/workspace/.caster_state`) mounted into the
    dockerd container. In Docker Compose (docker socket) this can be a named volume.
    """

    return os.getenv(_STATE_MOUNT_SOURCE_ENV, DEFAULT_STATE_DIR)


def default_state_volumes(*, mode: str | None = "ro") -> tuple[tuple[str, str, str | None], ...]:
    """Return a default sandbox volume mount for the shared state directory."""

    return ((resolve_state_mount_source(), DEFAULT_STATE_DIR, mode),)


__all__ = [
    "DEFAULT_STATE_DIR",
    "default_state_dir_path",
    "default_state_volumes",
    "resolve_state_mount_source",
]
