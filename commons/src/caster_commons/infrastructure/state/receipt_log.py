"""In-memory receipt log implementation."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from uuid import UUID

from caster_commons.application.ports.receipt_log import ReceiptLogPort
from caster_commons.domain.tool_call import ToolCall


class InMemoryReceiptLog(ReceiptLogPort):
    """Stores tool call receipts in-memory for the lifetime of a session."""

    def __init__(self) -> None:
        self._receipts: dict[str, ToolCall] = {}
        self._session_index: defaultdict[UUID, set[str]] = defaultdict(set)
        self._lock = Lock()

    def record(self, receipt: ToolCall) -> None:
        with self._lock:
            self._receipts[receipt.receipt_id] = receipt
            self._session_index[receipt.session_id].add(receipt.receipt_id)

    def lookup(self, receipt_id: str) -> ToolCall | None:
        with self._lock:
            return self._receipts.get(receipt_id)

    def values(self) -> tuple[ToolCall, ...]:
        with self._lock:
            receipts = tuple(self._receipts.values())
        return receipts

    def for_session(self, session_id: UUID) -> tuple[ToolCall, ...]:
        with self._lock:
            receipt_ids = tuple(self._session_index.get(session_id, ()))
            receipts = tuple(self._receipts[receipt_id] for receipt_id in receipt_ids)
        return receipts

    def clear_session(self, session_id: UUID) -> None:
        with self._lock:
            receipt_ids = self._session_index.pop(session_id, set())
            for receipt_id in receipt_ids:
                self._receipts.pop(receipt_id, None)


__all__ = ["InMemoryReceiptLog"]
