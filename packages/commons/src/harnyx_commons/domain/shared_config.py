"""Shared strict Pydantic config for commons-owned models."""

from __future__ import annotations

from pydantic import ConfigDict

COMMONS_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)

COMMONS_STRICT_DATACLASS_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    str_strip_whitespace=True,
)

__all__ = ["COMMONS_STRICT_CONFIG", "COMMONS_STRICT_DATACLASS_CONFIG"]
