"""Shared client defaults (base URLs, timeouts) for external services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeSearchDefaults:
    base_url: str = "https://api.desearch.ai"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class ParallelDefaults:
    base_url: str = "https://api.parallel.ai"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class FirecrawlDefaults:
    base_url: str = "https://api.firecrawl.dev"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class ExaDefaults:
    base_url: str = "https://api.exa.ai"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class TavilyDefaults:
    base_url: str = "https://api.tavily.com"
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class ChutesDefaults:
    base_url: str = "https://llm.chutes.ai"
    timeout_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class PlatformDefaults:
    timeout_seconds: float = 10.0


# Instances
DESEARCH = DeSearchDefaults()
PARALLEL = ParallelDefaults()
FIRECRAWL = FirecrawlDefaults()
EXA = ExaDefaults()
TAVILY = TavilyDefaults()
CHUTES = ChutesDefaults()
PLATFORM = PlatformDefaults()

__all__ = [
    "CHUTES",
    "DESEARCH",
    "FIRECRAWL",
    "EXA",
    "PARALLEL",
    "TAVILY",
    "PLATFORM",
    "ChutesDefaults",
    "DeSearchDefaults",
    "FirecrawlDefaults",
    "ExaDefaults",
    "ParallelDefaults",
    "TavilyDefaults",
    "PlatformDefaults",
]
