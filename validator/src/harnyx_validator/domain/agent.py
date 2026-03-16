"""Agent registry models and transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum


class AgentStatus(StrEnum):
    """Lifecycle states for a registered agent."""

    ACTIVE = "active"
    ERRORED = "errored"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Description of a tool exposed by a miner."""

    name: str
    description: str
    schema_url: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if self.schema_url is not None and not self.schema_url.strip():
            raise ValueError("schema_url must not be empty when supplied")


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """Sandbox configuration supplied by the host runtime."""

    runtime_image: str
    sdk_version: str
    tools: tuple[ToolDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if not self.runtime_image.strip():
            raise ValueError("runtime_image must not be empty")
        if not self.sdk_version.strip():
            raise ValueError("sdk_version must not be empty")
        names = {tool.name for tool in self.tools}
        if len(names) != len(self.tools):
            raise ValueError("tool names must be unique within a sandbox specification")


@dataclass(frozen=True, slots=True)
class AgentRegistry:
    """Registry record describing the latest synced miner manifest."""

    uid: int
    gist_id: str
    gist_file: str
    gist_commit_sha: str
    runtime_image: str
    last_synced_block: int
    last_synced_at: datetime
    status: AgentStatus = AgentStatus.ACTIVE
    sync_error: str | None = None
    sandbox: SandboxSpec | None = None

    def __post_init__(self) -> None:
        if self.uid <= 0:
            raise ValueError("uid must be positive")
        if not self.gist_id.strip():
            raise ValueError("gist_id must not be empty")
        if not self.gist_file.strip():
            raise ValueError("gist_file must not be empty")
        if not self.gist_commit_sha.strip():
            raise ValueError("gist_commit_sha must not be empty")
        if not self.runtime_image.strip():
            raise ValueError("runtime_image must not be empty")
        if self.last_synced_block < 0:
            raise ValueError("last_synced_block must be non-negative")
        if self.sync_error is not None and not self.sync_error.strip():
            raise ValueError("sync_error must not be empty when supplied")

    def mark_synced(
        self,
        *,
        gist_commit_sha: str,
        last_synced_block: int,
        last_synced_at: datetime,
        sandbox: SandboxSpec | None = None,
    ) -> AgentRegistry:
        """Return an updated record after a successful manifest sync."""
        if not gist_commit_sha.strip():
            raise ValueError("gist_commit_sha must not be empty")
        if last_synced_block < 0:
            raise ValueError("last_synced_block must be non-negative")
        return replace(
            self,
            gist_commit_sha=gist_commit_sha,
            last_synced_block=last_synced_block,
            last_synced_at=last_synced_at,
            status=AgentStatus.ACTIVE,
            sync_error=None,
            sandbox=self.sandbox if sandbox is None else sandbox,
        )

    def mark_error(self, message: str, *, at: datetime) -> AgentRegistry:
        """Flag the record as errored with the latest failure reason."""
        if not message.strip():
            raise ValueError("sync error message must not be empty")
        return replace(
            self,
            status=AgentStatus.ERRORED,
            sync_error=message,
            last_synced_at=at,
        )

    def disable(self, *, reason: str | None = None, at: datetime | None = None) -> AgentRegistry:
        """Disable the agent while preserving the last sync metadata."""
        if reason is not None and not reason.strip():
            raise ValueError("reason must not be empty when supplied")
        return replace(
            self,
            status=AgentStatus.DISABLED,
            sync_error=self.sync_error if reason is None else reason,
            last_synced_at=self.last_synced_at if at is None else at,
        )


__all__ = ["AgentRegistry", "AgentStatus", "SandboxSpec", "ToolDescriptor"]
