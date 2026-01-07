"""Shared type aliases for LLM providers.

Kept separate from provider implementations to avoid import cycles with settings modules.
"""

from __future__ import annotations

from typing import Literal

LlmProviderName = Literal["openai", "chutes", "vertex", "vertex-maas"]

__all__ = ["LlmProviderName"]
