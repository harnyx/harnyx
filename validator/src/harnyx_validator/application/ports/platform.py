"""Port definitions for interacting with the platform."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.tools.types import ToolName
from harnyx_validator.application.dto.evaluation import (
    MinerTaskWorkAssignment,
    PlatformOwnedTaskResult,
)


class PlatformPort(Protocol):
    """Abstract platform client capable of champion lookup and logging."""

    def fetch_artifact(self, batch_id: UUID, artifact_id: UUID) -> bytes:
        """Download the python agent artifact for a given candidate in the batch."""
        ...

    def get_champion_weights(self) -> ChampionWeights:
        """Return platform-computed champion weights."""
        ...

    async def request_miner_task_work(
        self,
        *,
        target_concurrency: int,
        max_active_artifacts: int,
        active_attempts: Sequence[PlatformTaskAttemptIdentity],
    ) -> tuple[MinerTaskWorkAssignment, ...]:
        """Return platform-assigned attempts for current task and artifact capacity."""
        ...

    def submit_miner_task_work_results(
        self,
        results: Sequence[PlatformOwnedTaskResult],
    ) -> tuple[PlatformTaskResultAcknowledgement, ...]:
        """Submit completed platform-assigned attempts."""
        ...


class PlatformToolProxyPlatformPort(Protocol):
    """Async platform client for platform-tool-proxy token and execution calls."""

    async def create_platform_tool_proxy_grant(
        self,
        *,
        batch_id: UUID,
        artifact_id: UUID,
        task_id: UUID,
        validator_session_id: UUID,
        attempt_number: int,
        assignment_token: str,
    ) -> PlatformToolProxyGrant:
        """Create a platform-tool-proxy grant for a validator-owned batch delivery."""
        ...

    async def execute_platform_tool_proxy_tool(
        self,
        *,
        token: str,
        uid: int,
        artifact_id: UUID,
        task_id: UUID,
        validator_session_id: UUID,
        attempt_number: int,
        receipt_id: str,
        tool: ToolName,
        args: tuple[JsonValue, ...],
        kwargs: dict[str, JsonValue],
        transport_timeout_seconds: float,
    ) -> PlatformToolProxyToolResult:
        """Execute a provider-backed tool through the platform tool proxy."""
        ...


@dataclass(frozen=True)
class ChampionWeights:
    champion_uid: int | None
    weights: dict[int, float]


class PlatformWeightsUnavailableError(RuntimeError):
    """Raised when platform has no safe weights for this submission tick."""


@dataclass(frozen=True)
class PlatformToolProxyGrant:
    token: str
    expires_at: datetime


class PlatformToolProxyControlError(PermissionError):
    """Raised when validator-local proxy control state denies invocation."""

    error_code = "platform_tool_proxy_denied"
    status_code = 403


class PlatformToolProxyTokenExpiredError(PlatformToolProxyControlError):
    """Raised when a non-renewable platform-tool-proxy token expires in-session."""


@dataclass(frozen=True)
class PlatformToolProxyToolResult:
    response: JsonObject
    execution: ToolExecutionFacts | None = None
    actual_cost_usd: float | None = None
    actual_cost_provider: str | None = None


@dataclass(frozen=True)
class PlatformTaskAttemptIdentity:
    batch_id: UUID
    artifact_id: UUID
    task_id: UUID
    attempt_number: int
    validator_session_id: UUID | None = None


@dataclass(frozen=True)
class PlatformTaskResultAcknowledgement:
    batch_id: UUID
    artifact_id: UUID
    task_id: UUID
    attempt_number: int
    outcome: Literal["accepted", "rejected"]
    canonical: bool
    reason_code: str | None = None
    reason: str | None = None


__all__ = [
    "PlatformPort",
    "PlatformWeightsUnavailableError",
    "PlatformToolProxyPlatformPort",
    "ChampionWeights",
    "PlatformToolProxyControlError",
    "PlatformToolProxyGrant",
    "PlatformToolProxyTokenExpiredError",
    "PlatformToolProxyToolResult",
    "PlatformTaskAttemptIdentity",
    "PlatformTaskResultAcknowledgement",
]
