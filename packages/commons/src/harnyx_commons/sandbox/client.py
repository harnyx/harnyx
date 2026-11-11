"""Shared sandbox client protocol used by managers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from harnyx_commons.json_types import JsonValue
from harnyx_miner_sdk.sandbox_headers import EXECUTION_COMPLETED_HEADER
from harnyx_miner_sdk.sandbox_protocol import SandboxAdmission, SandboxResourceStatus


@dataclass(frozen=True, slots=True)
class SandboxInvocationResult:
    payload: JsonValue
    completed_at: datetime


def parse_execution_completion(headers: Mapping[str, str]) -> datetime | None:
    raw = headers.get(EXECUTION_COMPLETED_HEADER)
    if raw is None:
        return None
    try:
        completed_at = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise InvalidSandboxResponseError("invalid execution completion timestamp") from exc
    if completed_at.tzinfo is None or completed_at > datetime.now(UTC):
        raise InvalidSandboxResponseError("untrustworthy execution completion timestamp")
    return completed_at


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
        completed_at: datetime | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail_code = detail_code
        self.detail_exception = detail_exception
        self.detail_error = detail_error
        self.remote_state_uncertain = remote_state_uncertain
        self.completed_at = completed_at


class SandboxClient(Protocol):
    """Adapter responsible for calling miner entrypoints."""

    @property
    def draining(self) -> bool:
        """Whether positive OOM evidence has retired this deployment."""
        return False

    @property
    def cleanup_retirement_required(self) -> bool:
        """Whether cleanup uncertainty requires physical container retirement."""
        return False

    async def resource_status(self) -> SandboxResourceStatus:
        return SandboxResourceStatus(counter_status="disabled", oom_kill_count=0, draining=False)

    def configure_resources(
        self,
        observe: Callable[[], bool | None],
        close_observer: Callable[[], None],
        reconcile: Callable[[], bool | None] | None = None,
    ) -> None:
        raise RuntimeError("sandbox client does not support evaluation resource control")

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
    ) -> SandboxInvocationResult | Mapping[str, JsonValue]:
        """Return query payload with trusted completion, or a non-query payload."""

    def close(self) -> None:
        """Release any client-side resources."""


__all__ = ["InvalidSandboxResponseError", "SandboxClient", "SandboxInvokeError"]
