"""Shared sandbox client protocol used by managers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from caster_commons.json_types import JsonValue


class SandboxClient(Protocol):
    """Adapter responsible for calling miner entrypoints."""

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, JsonValue]:
        """Invoke the sandbox entrypoint and return its response payload."""

    def close(self) -> None:
        """Release any client-side resources."""


__all__ = ["SandboxClient"]
