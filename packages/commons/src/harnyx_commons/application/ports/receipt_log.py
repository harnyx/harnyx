"""Port describing runtime receipt bookkeeping."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol
from uuid import UUID

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import StartedToolCall, ToolCall


class ReceiptLogPort(Protocol):
    """Stores tool call receipts for the lifetime of a session."""

    def record(self, receipt: ToolCall) -> None:
        """Insert or update a receipt entry."""

    def start_pending_receipt(
        self,
        *,
        started_call: StartedToolCall,
    ) -> None:
        """Reserve a receipt id for a started tool call."""

    def complete_pending_receipt(
        self,
        receipt: ToolCall,
        settle_usage: Callable[[], tuple[Session, bool]],
    ) -> tuple[Session, bool] | None:
        """Record a final receipt and settle usage when the pending receipt still exists."""

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        """Remove a pending receipt that failed before final materialization."""

    def lookup(self, receipt_id: str) -> ToolCall | None:
        """Return the receipt identified by ``receipt_id``."""

    def values(self) -> Iterable[ToolCall]:
        """Return an iterable of all stored receipts."""

    def for_session(self, session_id: UUID) -> Iterable[ToolCall]:
        """Return all receipts recorded for the supplied session."""

    def clear_session(self, session_id: UUID) -> None:
        """Remove receipts associated with the supplied session."""


__all__ = ["ReceiptLogPort"]
