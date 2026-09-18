"""Validator-owned miner transport and forwarding of original signed answers."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.parse import urlsplit
from uuid import UUID

from harnyx_commons.bittensor import verify_signed_request
from harnyx_commons.endpoint_execution import (
    ENDPOINT_START_ALLOWANCE,
    EXECUTION_PATH,
    EndpointAuthority,
    EndpointExecutionWork,
    EndpointFailureReport,
    EndpointProgress,
    EndpointResponseReport,
    EndpointStartReport,
    verify_delegation,
)
from harnyx_commons.task_ownership import wait_for_owned_task
from harnyx_miner_sdk.endpoint_protocol import (
    EndpointAssignment,
    EndpointCallback,
    EndpointCallbackAcknowledgement,
    EndpointDelegation,
    EndpointDurableTerminalResult,
    EndpointMinerStatus,
    EndpointStatusResponse,
    query_digest,
)
from harnyx_validator.application.ports.platform import EndpointPlatformPort

logger = logging.getLogger(__name__)


class EndpointStatusUnavailableError(RuntimeError):
    """No trusted miner status was available."""


class EndpointReportRejectedError(ValueError):
    """Platform definitively rejected a candidate's schema or receipts."""


@dataclass(slots=True)
class MinerRequestAttempt:
    attempted_at: datetime | None = None
    admission_confirmed: bool = False


class MinerEndpointPort(Protocol):
    async def send_assignment(
        self, *, endpoint_url: str, expected_hotkey: str, assignment: EndpointAssignment, stop_at: float
    ) -> MinerRequestAttempt: ...
    async def get_status(
        self,
        *,
        endpoint_url: str,
        expected_hotkey: str,
        assignment_id: UUID,
        deadline_at: datetime,
        stop_at: float,
    ) -> EndpointStatusResponse: ...
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class _Execution:
    work: EndpointExecutionWork | None
    authority: EndpointAuthority
    platform_hotkey: str
    state: Literal["queued", "active", "reporting", "saved"] = "queued"
    attempted_at: datetime | None = None
    pending: list[EndpointResponseReport] = field(default_factory=list)
    terminal: EndpointCallbackAcknowledgement | None = None
    task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class ValidatorEndpointExecution:
    platform: EndpointPlatformPort
    miner: MinerEndpointPort
    authorize_delegation: Callable[[EndpointDelegation], Awaitable[EndpointAuthority]]
    validator_hotkey: str
    callback_base_url: str
    capacity: int = 100
    _running: bool = field(default=False, init=False)
    _executions: dict[UUID, _Execution] = field(default_factory=dict, init=False)
    _callback_authorizations: set[UUID] = field(default_factory=set, init=False)

    def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        tasks = [item.task for item in self._executions.values() if item.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            await wait_for_owned_task(task)
        self._executions.clear()
        await self.miner.aclose()

    async def _authority(self, work: EndpointExecutionWork) -> EndpointAuthority:
        authority = await self.authorize_delegation(work.delegation)
        expected = f"{self.callback_base_url.rstrip('/')}{EXECUTION_PATH}/{authority.assignment_id}/callback"
        if authority.validator_hotkey != self.validator_hotkey or authority.callback_url != expected:
            raise PermissionError("delegation targets another validator or callback URL")
        if query_digest(work.query) != authority.query_digest:
            raise ValueError("query does not match delegation")
        return authority

    async def execute(self, work: EndpointExecutionWork) -> EndpointProgress:
        authority = await self._authority(work)
        existing = self._executions.get(authority.assignment_id)
        if existing is not None and (
            existing.task is None or not existing.task.done() or existing.terminal is not None
        ):
            return EndpointProgress(assignment_id=authority.assignment_id, state=existing.state)
        if not self._running:
            raise RuntimeError("endpoint execution worker is stopped")
        self._prune_completed()
        existing = self._executions.get(authority.assignment_id)
        if existing is not None:
            return EndpointProgress(assignment_id=authority.assignment_id, state=existing.state)
        if len(self._executions) >= self.capacity:
            raise RuntimeError("endpoint execution capacity is exhausted")
        if authority.assignment_id in self._callback_authorizations:
            raise RuntimeError("callback recovery is authenticating this assignment")
        item = _Execution(work=work, authority=authority, platform_hotkey=work.delegation.platform_hotkey)
        self._executions[authority.assignment_id] = item
        item.task = asyncio.create_task(self._run(item))
        return EndpointProgress(assignment_id=authority.assignment_id, state="queued")

    def _prune_completed(self) -> None:
        # Saved results are recoverable from Platform. Keep live work and any
        # original receive events that can still be forwarded after a task exits.
        now = datetime.now(UTC)
        self._executions = {
            key: item
            for key, item in self._executions.items()
            if (item.task is not None and not item.task.done())
            or (item.pending and now <= item.authority.report_cutoff)
        }

    async def _run(self, item: _Execution) -> None:
        try:
            await self._continue(item)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "endpoint execution interrupted", extra={"assignment_id": str(item.authority.assignment_id)}
            )

    async def _continue(self, item: _Execution) -> None:
        assignment_id = item.authority.assignment_id
        assert item.work is not None
        proposed_start = datetime.now(UTC)
        while self._running:
            try:
                snapshot = await self.platform.saved_endpoint(assignment_id, item.work.delegation)
                item.work = snapshot.work
                item.authority = await self._authority(item.work)
                if snapshot.status in {"succeeded", "endpoint_failed", "platform_void"}:
                    item.terminal = _terminal_ack(snapshot.status)
                    item.state = "saved"
                    return
                if item.authority.started_at is None:
                    item.work = await self.platform.start_endpoint(
                        assignment_id,
                        EndpointStartReport(delegation=item.work.delegation, proposed_start=proposed_start),
                    )
                    item.authority = await self._authority(item.work)
                break
            except Exception:
                cutoff = (
                    item.authority.report_cutoff
                    if item.authority.started_at
                    else item.authority.first_scheduled_at + ENDPOINT_START_ALLOWANCE
                )
                if datetime.now(UTC) > cutoff:
                    return
                await asyncio.sleep(1)
        authority = item.authority
        if authority.deadline_at is None:
            return
        item.state = "active"
        # Re-presenting the same assignment registers this validator's callback destination.
        # An idempotent progress request above never reaches this send again.
        admission_confirmed = False
        if datetime.now(UTC) < authority.deadline_at:
            remaining = (authority.deadline_at - datetime.now(UTC)).total_seconds()
            attempt = await self.miner.send_assignment(
                endpoint_url=authority.endpoint_url,
                expected_hotkey=authority.miner_hotkey,
                assignment=authority.assignment(item.work.query, item.work.delegation),
                stop_at=asyncio.get_running_loop().time() + min(10.0, remaining),
            )
            item.attempted_at = attempt.attempted_at
            admission_confirmed = attempt.admission_confirmed
        failure_saved = False
        while self._running and datetime.now(UTC) <= authority.report_cutoff:
            for report in tuple(item.pending):
                try:
                    await self._forward(item, report)
                except Exception:
                    logger.debug("endpoint operation will retry")
            if item.terminal is not None:
                return
            now = datetime.now(UTC)
            if now >= authority.deadline_at:
                item.state = "reporting"
                if item.attempted_at is not None and not item.pending and not failure_saved:
                    try:
                        failure_saved = await self.platform.fail_endpoint(
                            assignment_id,
                            EndpointFailureReport(
                                delegation=item.work.delegation,
                                attempted_at=item.attempted_at,
                                observed_through=now,
                            ),
                        )
                    except Exception:
                        logger.debug("endpoint failure report will retry")
            else:
                status = None
                try:
                    status = await self.miner.get_status(
                        endpoint_url=authority.endpoint_url,
                        expected_hotkey=authority.miner_hotkey,
                        assignment_id=assignment_id,
                        deadline_at=authority.deadline_at,
                        stop_at=asyncio.get_running_loop().time()
                        + min(10.0, (authority.deadline_at - now).total_seconds()),
                    )
                except Exception:
                    logger.debug("endpoint status unavailable")
                if status is not None:
                    admission_confirmed = status.state is not EndpointMinerStatus.UNKNOWN
                try:
                    remaining = (authority.deadline_at - datetime.now(UTC)).total_seconds()
                    # A callback may have arrived while the status request was pending.
                    if not admission_confirmed and not item.pending and item.terminal is None and remaining > 0:
                        attempt = await self.miner.send_assignment(
                            endpoint_url=authority.endpoint_url,
                            expected_hotkey=authority.miner_hotkey,
                            assignment=authority.assignment(item.work.query, item.work.delegation),
                            stop_at=asyncio.get_running_loop().time() + min(10.0, remaining),
                        )
                        admission_confirmed = attempt.admission_confirmed
                        if item.attempted_at is None:
                            item.attempted_at = attempt.attempted_at
                except Exception:
                    logger.debug("endpoint operation will retry")
            await asyncio.sleep(1)

    async def accept_callback(
        self,
        *,
        assignment_id: UUID,
        delegation: EndpointDelegation,
        raw_body: bytes,
        authorization_header: str | None,
        received_at: datetime,
        signed_path: str,
    ) -> EndpointCallbackAcknowledgement:
        existing = self._executions.get(assignment_id)
        if existing is not None:
            # The assignment already authenticated this signer. Verify new delegation
            # bytes synchronously so deadline observation cannot miss this callback.
            authority = verify_delegation(delegation, existing.platform_hotkey)
        else:
            if assignment_id in self._callback_authorizations or len(self._callback_authorizations) >= self.capacity:
                raise RuntimeError("callback authorization capacity exhausted")
            self._callback_authorizations.add(assignment_id)
            try:
                authority = await self.authorize_delegation(delegation)
            finally:
                self._callback_authorizations.remove(assignment_id)
        if authority.assignment_id != assignment_id or authority.validator_hotkey != self.validator_hotkey:
            raise PermissionError("callback delegation does not authorize this validator")
        expected = f"{self.callback_base_url.rstrip('/')}{EXECUTION_PATH}/{assignment_id}/callback"
        if authority.callback_url != expected or signed_path != urlsplit(expected).path:
            raise PermissionError("callback destination does not match delegation")
        signature = verify_signed_request(
            method="POST",
            path_qs=signed_path,
            body=raw_body,
            authorization_header=authorization_header,
            allowed_ss58=(authority.miner_hotkey,),
        )
        callback = EndpointCallback.model_validate_json(raw_body, strict=True)
        if (
            callback.assignment_id != assignment_id
            or callback.query_digest != authority.query_digest
            or callback.nonce != authority.nonce
            or callback.expires_at != authority.deadline_at
        ):
            raise ValueError("callback does not match assignment")
        item = self._executions.get(assignment_id)
        if item is not None and item.terminal is not None:
            return item.terminal
        report = EndpointResponseReport(
            delegation=delegation,
            received_at=received_at,
            callback_base64=base64.b64encode(raw_body).decode("ascii"),
            signature_hex=signature.signature_hex,
            signed_callback_path=signed_path,
        )
        if item is None:
            # A restarted validator can authenticate the callback without its lost registry.
            # Keep the complete receive event while recovering the canonical assignment.
            self._prune_completed()
            if len(self._executions) >= self.capacity:
                raise RuntimeError("endpoint callback capacity exhausted")
            item = _Execution(
                work=None, authority=authority, platform_hotkey=delegation.platform_hotkey, state="reporting"
            )
            self._executions[assignment_id] = item
        # A duplicate delivery is the original receive event, even after the deadline.
        for pending in item.pending:
            if (
                pending.callback_base64 == report.callback_base64
                and pending.signed_callback_path == report.signed_callback_path
            ):
                return await self._forward(item, pending)
        if authority.deadline_at is None or received_at >= authority.deadline_at:
            snapshot = await self.platform.saved_endpoint(assignment_id, delegation)
            if snapshot.status in {"succeeded", "endpoint_failed", "platform_void"}:
                return _terminal_ack(snapshot.status)
            raise ValueError("answer arrived after the miner deadline")
        if len(item.pending) >= 4:
            raise RuntimeError("pending answer capacity exhausted")
        item.pending.append(report)
        if item.task is None or item.task.done():
            item.task = asyncio.create_task(self._retry_reports(item))
        return await self._forward(item, report)

    async def _retry_reports(self, item: _Execution) -> None:
        while self._running and item.pending and datetime.now(UTC) <= item.authority.report_cutoff:
            await asyncio.sleep(1)
            for report in tuple(item.pending):
                try:
                    await self._forward(item, report)
                except Exception:
                    logger.debug("endpoint operation will retry")

    async def _forward(self, item: _Execution, report: EndpointResponseReport) -> EndpointCallbackAcknowledgement:
        try:
            ack = await self.platform.report_endpoint(item.authority.assignment_id, report)
        except EndpointReportRejectedError:
            if report in item.pending:
                item.pending.remove(report)
            raise
        item.terminal = ack
        item.state = "saved"
        item.pending.clear()
        return ack


def _terminal_ack(status: str) -> EndpointCallbackAcknowledgement:
    return EndpointCallbackAcknowledgement(
        durable_terminal_result=(
            EndpointDurableTerminalResult.PERSISTED if status == "succeeded" else EndpointDurableTerminalResult.CLOSED
        )
    )
