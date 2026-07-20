"""Shared sandbox client protocol used by managers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

from harnyx_commons.json_types import JsonValue


class SandboxInvokeError(RuntimeError):
    """Structured sandbox invocation failure surfaced by shared clients."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        detail_code: str | None,
        detail_exception: str | None,
        detail_error: str | None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail_code = detail_code
        self.detail_exception = detail_exception
        self.detail_error = detail_error


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
        include_failure_details: bool = True,
    ) -> Mapping[str, JsonValue]:
        """Invoke the sandbox entrypoint and return its response payload."""

    def close(self) -> None:
        """Release any client-side resources."""


__all__ = ["SandboxClient", "SandboxInvokeError"]
