"""Shared client defaults (base URLs, timeouts) for external services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeSearchDefaults:
    base_url: str = "https://api.desearch.ai"
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class ChutesDefaults:
    base_url: str = "https://llm.chutes.ai"
    timeout_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class OpenAIDefaults:
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class PlatformDefaults:
    timeout_seconds: float = 10.0


# Instances
DESEARCH = DeSearchDefaults()
CHUTES = ChutesDefaults()
OPENAI = OpenAIDefaults()
PLATFORM = PlatformDefaults()

__all__ = [
    "CHUTES",
    "DESEARCH",
    "OPENAI",
    "PLATFORM",
    "ChutesDefaults",
    "DeSearchDefaults",
    "OpenAIDefaults",
    "PlatformDefaults",
]
