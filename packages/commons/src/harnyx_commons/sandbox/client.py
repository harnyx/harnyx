"""Shared sandbox client protocol used by managers."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from uuid import UUID

from harnyx_commons.json_types import JsonValue
from harnyx_miner_sdk.sandbox_protocol import SandboxAdmission


class InvalidSandboxResponseError(ValueError):
    """The sandbox returned an HTTP response that violates the response contract."""


class SandboxResponseProcessingError(RuntimeError):
    """A received sandbox response failed during unexpected local processing."""


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
        remote_state_uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail_code = detail_code
        self.detail_exception = detail_exception
        self.detail_error = detail_error
        self.remote_state_uncertain = remote_state_uncertain


class SandboxClient(Protocol):
    """Adapter responsible for calling miner entrypoints."""

    def admission(self, limit_seconds: float, token: str) -> AbstractAsyncContextManager[SandboxAdmission]:
        """Reserve execution before creating a host tool session."""
        ...

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
        include_failure_details: bool = True,
        admission: SandboxAdmission | None = None,
    ) -> Mapping[str, JsonValue]:
        """Invoke the sandbox entrypoint and return its response payload."""

    def close(self) -> None:
        """Release any client-side resources."""


__all__ = ["InvalidSandboxResponseError", "SandboxClient", "SandboxInvokeError"]
