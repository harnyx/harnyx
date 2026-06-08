"""Port definitions for interacting with the platform."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.json_types import JsonObject, JsonValue
from harnyx_commons.tools.types import ToolName
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission
from harnyx_validator.application.ports.progress import ProviderFailureEvidence


class PlatformPort(Protocol):
    """Abstract platform client capable of champion lookup and logging."""

    def get_miner_task_batch(self, batch_id: UUID) -> MinerTaskBatchSpec:
        """Retrieve a platform-composed miner-task batch."""
        ...

    def fetch_artifact(self, batch_id: UUID, artifact_id: UUID) -> bytes:
        """Download the python agent artifact for a given candidate in the batch."""
        ...

    def get_champion_weights(self) -> ChampionWeights:
        """Return platform-computed champion weights."""
        ...

    def get_restore_metadata(self, batch_id: UUID) -> RestoreMetadata:
        """Return restore metadata for the validator-owned batch delivery."""
        ...

    def get_restore_runs_page(
        self,
        *,
        batch: MinerTaskBatchSpec,
        snapshot_received_at: datetime,
        cursor: int,
        limit: int,
    ) -> RestoreRunsPage:
        """Return one restore page converted to validator-domain submissions."""
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
class RestoreAttemptNumberHighWater:
    artifact_id: UUID
    task_id: UUID
    max_attempt_number: int


@dataclass(frozen=True)
class RestoreMetadata:
    batch_id: UUID
    snapshot_received_at: datetime
    total_restore_runs: int
    page_limit: int
    last_progress_detail_sequence: int = 0
    provider_model_evidence: tuple[ProviderFailureEvidence, ...] = ()
    terminated_miner_task_attempts: tuple[RestoreAttemptNumberHighWater, ...] = ()
    consumed_platform_tool_proxy_attempts: tuple[RestoreAttemptNumberHighWater, ...] = ()


@dataclass(frozen=True)
class RestoreRunsPage:
    batch_id: UUID
    snapshot_received_at: datetime
    cursor: int
    limit: int
    next_cursor: int | None
    items: tuple[MinerTaskRunSubmission, ...] = ()


__all__ = [
    "PlatformPort",
    "PlatformToolProxyPlatformPort",
    "ChampionWeights",
    "PlatformToolProxyControlError",
    "PlatformToolProxyGrant",
    "PlatformToolProxyTokenExpiredError",
    "PlatformToolProxyToolResult",
    "RestoreMetadata",
    "RestoreAttemptNumberHighWater",
    "RestoreRunsPage",
]
