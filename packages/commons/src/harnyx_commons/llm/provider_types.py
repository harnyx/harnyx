"""Shared type aliases for LLM providers.

Kept separate from provider implementations to avoid import cycles with settings modules.
"""

from __future__ import annotations

from typing import Literal

CHUTES_PROVIDER = "chutes"
VERTEX_PROVIDER = "vertex"
VERTEX_MAAS_PROVIDER = "vertex-maas"

LlmProviderName = Literal["chutes", "vertex", "vertex-maas"]


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    normalized = reasoning_effort.strip()
    if not normalized:
        return None
    try:
        if int(normalized) <= 0:
            return None
    except ValueError:
        return normalized
    return normalized


__all__ = [
    "CHUTES_PROVIDER",
    "LlmProviderName",
    "VERTEX_MAAS_PROVIDER",
    "VERTEX_PROVIDER",
    "normalize_reasoning_effort",
]
