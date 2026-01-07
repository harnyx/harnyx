"""Port describing access to the miner agent registry."""

from __future__ import annotations

from typing import Protocol

from caster_validator.domain.agent import AgentRegistry, AgentStatus


class AgentRegistryPort(Protocol):
    """Abstract store for miner agent registry entries."""

    def upsert(self, agent: AgentRegistry) -> None:
        """Insert or update the supplied agent record."""

    def get(self, uid: int) -> AgentRegistry | None:
        """Return the agent identified by UID."""

    def list_by_status(self, status: AgentStatus) -> tuple[AgentRegistry, ...]:
        """Return all agents with the specified status."""


__all__ = ["AgentRegistryPort"]
