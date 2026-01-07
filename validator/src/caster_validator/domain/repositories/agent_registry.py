"""Persistence contract for agent registry records."""

from __future__ import annotations

from typing import Protocol

from caster_validator.domain.agent import AgentRegistry, AgentStatus


class AgentRegistryRepository(Protocol):
    """Abstract storage for agent registry entries."""

    def upsert(self, agent: AgentRegistry) -> None:
        """Insert or update the supplied agent record."""

    def by_uid(self, uid: int) -> AgentRegistry | None:
        """Return the agent identified by UID."""

    def list_by_status(self, status: AgentStatus) -> tuple[AgentRegistry, ...]:
        """Return all agents with the specified status."""
