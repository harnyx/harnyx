"""In-memory receipt log implementation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from threading import Condition, Lock
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import (
    IN_FLIGHT_LLM_UNKNOWN_EVIDENCE,
    StartedToolCall,
    ToolCall,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.tools.types import ToolName


class InMemoryReceiptLog(ReceiptLogPort):
    """Stores tool call receipts in-memory for the lifetime of a session."""

    def __init__(self) -> None:
        self._receipts: dict[str, ToolCall] = {}
        self._session_index: defaultdict[UUID, set[str]] = defaultdict(set)
        self._pending: dict[str, StartedToolCall] = {}
        self._pending_by_session: defaultdict[UUID, set[str]] = defaultdict(set)
        self._closed_windows: set[tuple[UUID, int, ToolName]] = set()
        self._lock = Lock()
        self._condition = Condition(self._lock)

    def record(self, receipt: ToolCall) -> None:
        with self._condition:
            self._record_locked(receipt)
            self._condition.notify_all()

    def start_pending_receipt(
        self,
        *,
        started_call: StartedToolCall,
    ) -> None:
        with self._condition:
            window = (
                started_call.session_id,
                started_call.session_active_attempt,
                started_call.tool,
            )
            if window in self._closed_windows:
                raise RuntimeError(
                    "cannot start pending receipt after timeout review window closed"
                )
            receipt_id = started_call.receipt_id
            if receipt_id in self._receipts or receipt_id in self._pending:
                raise RuntimeError(f"receipt {receipt_id} already exists")
            self._pending[receipt_id] = started_call
            self._pending_by_session[started_call.session_id].add(receipt_id)
            self._condition.notify_all()

    def complete_pending_receipt(
        self,
        receipt: ToolCall,
        settle_usage: Callable[[], tuple[Session, bool]],
    ) -> tuple[Session, bool] | None:
        with self._condition:
            pending = self._pending.get(receipt.receipt_id)
            if pending is None:
                return None
            if receipt.session_id != pending.session_id:
                raise RuntimeError("completed receipt session does not match pending receipt")
            if receipt.tool != pending.tool:
                raise RuntimeError("completed receipt tool does not match pending receipt")
            try:
                settlement = settle_usage()
            except BaseException:
                self._record_locked(receipt)
                self._remove_pending_locked(receipt.receipt_id)
                self._condition.notify_all()
                raise
            self._record_locked(receipt)
            self._remove_pending_locked(receipt.receipt_id)
            self._condition.notify_all()
            return settlement

    def abandon_pending_receipt(self, receipt_id: str) -> None:
        with self._condition:
            self._remove_pending_locked(receipt_id)
            self._condition.notify_all()

    def wait_and_materialize_unknown_receipts(
        self,
        session_id: UUID,
        *,
        session_active_attempt: int,
        tool: ToolName,
        timeout_seconds: float,
        clock: Callable[[], datetime],
    ) -> tuple[ToolCall, ...]:
        with self._condition:
            self._condition.wait_for(
                lambda: not self._matching_pending_ids_locked(
                    session_id,
                    session_active_attempt=session_active_attempt,
                    tool=tool,
                ),
                timeout=timeout_seconds,
            )
            window = (session_id, session_active_attempt, tool)
            self._closed_windows.add(window)
            pending_ids = self._matching_pending_ids_locked(
                session_id,
                session_active_attempt=session_active_attempt,
                tool=tool,
            )
            materialized_at = clock()
            receipts: list[ToolCall] = []
            for receipt_id in pending_ids:
                started_call = self._pending[receipt_id]
                started_at = started_call.execution.started_at
                elapsed_ms = None
                if started_at is not None:
                    elapsed_ms = (materialized_at - started_at).total_seconds() * 1000.0
                receipts.append(
                    started_call.materialize(
                        outcome=ToolCallOutcome.TIMEOUT,
                        response_payload=None,
                        results=(),
                        cost_usd=None,
                        extra={
                            "timeout_attribution_evidence": IN_FLIGHT_LLM_UNKNOWN_EVIDENCE
                        },
                        execution=ToolExecutionFacts(
                            elapsed_ms=elapsed_ms,
                            started_at=started_at,
                            finished_at=materialized_at,
                        ),
                    ),
                )
            for receipt in receipts:
                self._record_locked(receipt)
                self._remove_pending_locked(receipt.receipt_id)
            self._condition.notify_all()
            return tuple(receipts)

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
        with self._condition:
            receipt_ids = self._session_index.pop(session_id, set())
            for receipt_id in receipt_ids:
                self._receipts.pop(receipt_id, None)
            pending_ids = self._pending_by_session.pop(session_id, set())
            for receipt_id in pending_ids:
                self._pending.pop(receipt_id, None)
            self._closed_windows = {
                window for window in self._closed_windows if window[0] != session_id
            }
            self._condition.notify_all()

    def _record_locked(self, receipt: ToolCall) -> None:
        self._receipts[receipt.receipt_id] = receipt
        self._session_index[receipt.session_id].add(receipt.receipt_id)

    def _remove_pending_locked(self, receipt_id: str) -> None:
        pending = self._pending.pop(receipt_id, None)
        if pending is None:
            return
        receipt_ids = self._pending_by_session.get(pending.session_id)
        if receipt_ids is None:
            return
        receipt_ids.discard(receipt_id)
        if not receipt_ids:
            self._pending_by_session.pop(pending.session_id, None)

    def _matching_pending_ids_locked(
        self,
        session_id: UUID,
        *,
        session_active_attempt: int,
        tool: ToolName,
    ) -> tuple[str, ...]:
        receipt_ids = self._pending_by_session.get(session_id, set())
        return tuple(
            receipt_id
            for receipt_id in receipt_ids
            if self._pending[receipt_id].session_active_attempt == session_active_attempt
            and self._pending[receipt_id].tool == tool
        )


__all__ = ["InMemoryReceiptLog"]
