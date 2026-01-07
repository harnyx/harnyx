"""Port describing runtime receipt bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol
from uuid import UUID

from caster_commons.domain.tool_call import ToolCall


class ReceiptLogPort(Protocol):
    """Stores tool call receipts for the lifetime of a session."""

    def record(self, receipt: ToolCall) -> None:
        """Insert or update a receipt entry."""

    def lookup(self, receipt_id: str) -> ToolCall | None:
        """Return the receipt identified by ``receipt_id``."""

    def values(self) -> Iterable[ToolCall]:
        """Return an iterable of all stored receipts."""

    def for_session(self, session_id: UUID) -> Iterable[ToolCall]:
        """Return all receipts recorded for the supplied session."""

    def clear_session(self, session_id: UUID) -> None:
        """Remove receipts associated with the supplied session."""


__all__ = ["ReceiptLogPort"]
