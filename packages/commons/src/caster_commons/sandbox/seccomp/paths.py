from __future__ import annotations

from pathlib import Path


def default_profile_path() -> str:
    """Return the absolute path to the bundled sandbox seccomp profile."""

    return str(Path(__file__).with_name("sandbox-default.json"))


__all__ = ["default_profile_path"]
