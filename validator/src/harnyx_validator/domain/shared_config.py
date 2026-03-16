"""Shared strict Pydantic config for validator-owned models."""

from __future__ import annotations

from pydantic import ConfigDict

VALIDATOR_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    str_strip_whitespace=True,
)

__all__ = ["VALIDATOR_STRICT_CONFIG"]
