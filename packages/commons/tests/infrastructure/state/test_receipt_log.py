from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.domain.session import Session
from harnyx_commons.domain.tool_call import (
    StartedToolCall,
    ToolCall,
    ToolCallOutcome,
    ToolExecutionFacts,
    ToolResultPolicy,
)
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog


def test_receipt_log_records_and_completes_pending_receipt() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    receipt_id = "receipt-1"
    started_call = _started_call(session, receipt_id=receipt_id, active_attempt=1)
    receipt_log.start_pending_receipt(started_call=started_call)

    completion = receipt_log.complete_pending_receipt(
        _ok_receipt(started_call),
        lambda: (session, False),
    )

    assert completion == (session, False)
    assert receipt_log.lookup(receipt_id) == _ok_receipt(
        started_call,
    )


def test_receipt_log_keeps_pending_receipt_until_usage_settlement_finishes() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    started_call = _started_call(session, receipt_id="receipt-1", active_attempt=1)
    receipt = _ok_receipt(started_call)
    receipt_log.start_pending_receipt(started_call=started_call)
    settlement_started = threading.Event()
    settlement_release = threading.Event()
    completion_result: list[tuple[Session, bool] | None] = []

    def settle_usage() -> tuple[Session, bool]:
        settlement_started.set()
        assert settlement_release.wait(timeout=1.0)
        return session, False

    def complete() -> None:
        completion_result.append(receipt_log.complete_pending_receipt(receipt, settle_usage))

    completion_thread = threading.Thread(target=complete)
    completion_thread.start()
    assert settlement_started.wait(timeout=1.0)

    assert completion_thread.is_alive()
    assert completion_result == []

    settlement_release.set()
    completion_thread.join(timeout=1.0)

    assert completion_result == [(session, False)]
    assert receipt_log.lookup(receipt.receipt_id) == receipt


def test_receipt_log_records_successful_receipt_when_usage_settlement_fails() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    started_call = _started_call(session, receipt_id="receipt-1", active_attempt=1)
    receipt = _ok_receipt(started_call)
    receipt_log.start_pending_receipt(started_call=started_call)

    def settle_usage() -> tuple[Session, bool]:
        raise RuntimeError("settlement failed")

    with pytest.raises(RuntimeError, match="settlement failed"):
        receipt_log.complete_pending_receipt(receipt, settle_usage)

    assert receipt_log.lookup(receipt.receipt_id) == receipt


def test_receipt_log_completion_returns_none_after_pending_receipt_is_abandoned() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    started_call = _started_call(session, receipt_id="receipt-1", active_attempt=2)
    receipt_log.start_pending_receipt(
        started_call=started_call,
    )
    receipt_log.abandon_pending_receipt(started_call.receipt_id)

    completion = receipt_log.complete_pending_receipt(
        _ok_receipt(started_call),
        lambda: (session, False),
    )

    assert completion is None
    assert receipt_log.for_session(session.session_id) == ()


def test_receipt_log_allows_new_pending_receipt_after_abandonment() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    abandoned_call = _started_call(session, receipt_id="receipt-1", active_attempt=1)
    receipt_log.start_pending_receipt(
        started_call=abandoned_call,
    )
    receipt_log.abandon_pending_receipt(abandoned_call.receipt_id)

    replacement_call = _started_call(session, receipt_id="receipt-2", active_attempt=1)
    receipt_log.start_pending_receipt(
        started_call=replacement_call,
    )

    assert (
        receipt_log.complete_pending_receipt(
            _ok_receipt(replacement_call),
            lambda: (session, False),
        )
        == (session, False)
    )


def test_receipt_log_clear_session_removes_pending_receipts() -> None:
    receipt_log = InMemoryReceiptLog()
    session = _session()
    started_call = _started_call(session, receipt_id="receipt-1", active_attempt=1)
    receipt_log.start_pending_receipt(
        started_call=started_call,
    )

    receipt_log.clear_session(session.session_id)

    assert (
        receipt_log.complete_pending_receipt(
            _ok_receipt(started_call),
            lambda: (session, False),
        )
        is None
    )


def _session() -> Session:
    issued_at = _issued_at()
    return Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
        budget_usd=1.0,
    )


def _issued_at() -> datetime:
    return datetime(2026, 5, 14, 12, tzinfo=UTC)


def _started_call(
    session: Session,
    *,
    receipt_id: str,
    active_attempt: int,
) -> StartedToolCall:
    return StartedToolCall(
        receipt_id=receipt_id,
        session_id=session.session_id,
        session_active_attempt=active_attempt,
        uid=session.uid,
        tool="llm_chat",
        issued_at=_issued_at(),
        request_payload={"args": [], "kwargs": {"model": "zai-org/GLM-5-TEE"}},
        result_policy=ToolResultPolicy.LOG_ONLY,
        execution=ToolExecutionFacts(started_at=_issued_at()),
    )


def _ok_receipt(started_call: StartedToolCall) -> ToolCall:
    return started_call.materialize(
        outcome=ToolCallOutcome.OK,
        response_payload={"usage": {"total_tokens": 10}},
        execution=ToolExecutionFacts(elapsed_ms=1000.0, started_at=_issued_at()),
    )
