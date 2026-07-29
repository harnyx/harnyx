"""Validator-side platform-tool-proxy token scoping and proxy invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from uuid import UUID

from harnyx_commons.json_types import JsonValue
from harnyx_commons.platform_tool_proxy import PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS
from harnyx_commons.tools.executor import ToolInvocationContext, ToolInvocationOutput, ToolInvoker
from harnyx_commons.tools.types import ToolName, is_embedding_tool, is_search_tool
from harnyx_validator.application.assigned_work import PhaseRecorder
from harnyx_validator.application.ports.platform import (
    PlatformToolProxyControlError,
    PlatformToolProxyPlatformPort,
    PlatformToolProxyTokenExpiredError,
)


@dataclass(frozen=True, slots=True)
class PlatformToolProxyAttemptGrant:
    token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class PlatformToolProxySessionScope:
    batch_id: UUID
    artifact_id: UUID
    task_id: UUID
    attempt_number: int
    assignment_token: str
    phase_recorder: PhaseRecorder | None
    grants_by_attempt: Mapping[int, PlatformToolProxyAttemptGrant]


@dataclass(slots=True)
class PlatformToolProxyScopeRegistry:
    _session_scopes: dict[UUID, PlatformToolProxySessionScope]
    _grant_locks: dict[tuple[UUID, int], asyncio.Lock]
    _lock: Lock

    def __init__(self) -> None:
        self._session_scopes = {}
        self._grant_locks = {}
        self._lock = Lock()

    def clear_batch(self, batch_id: UUID) -> None:
        with self._lock:
            stale_sessions = [
                session_id for session_id, scope in self._session_scopes.items() if scope.batch_id == batch_id
            ]
            for session_id in stale_sessions:
                self._session_scopes.pop(session_id, None)
            for key in [key for key in self._grant_locks if key[0] in stale_sessions]:
                self._grant_locks.pop(key, None)

    def register_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        artifact_id: UUID,
        task_id: UUID,
        assignment_token: str,
        attempt_number: int = 1,
        phase_recorder: PhaseRecorder | None = None,
    ) -> None:
        with self._lock:
            self._session_scopes[session_id] = PlatformToolProxySessionScope(
                batch_id=batch_id,
                artifact_id=artifact_id,
                task_id=task_id,
                attempt_number=attempt_number,
                assignment_token=assignment_token,
                phase_recorder=phase_recorder,
                grants_by_attempt={},
            )

    def clear_session(self, session_id: UUID) -> None:
        with self._lock:
            self._session_scopes.pop(session_id, None)
            for key in [key for key in self._grant_locks if key[0] == session_id]:
                self._grant_locks.pop(key, None)

    def require_session(self, session_id: UUID) -> PlatformToolProxySessionScope:
        with self._lock:
            scope = self._session_scopes.get(session_id)
        if scope is None:
            raise PlatformToolProxyControlError("platform tool proxy scope is not registered for session")
        return scope

    def store_session_grant(
        self,
        *,
        session_id: UUID,
        attempt_number: int,
        token: str,
        expires_at: datetime,
    ) -> PlatformToolProxyAttemptGrant:
        with self._lock:
            scope = self._session_scopes.get(session_id)
            if scope is None:
                raise PlatformToolProxyControlError("platform tool proxy scope is not registered for session")
            attempt_grant = PlatformToolProxyAttemptGrant(token=token, expires_at=expires_at)
            updated = PlatformToolProxySessionScope(
                batch_id=scope.batch_id,
                artifact_id=scope.artifact_id,
                task_id=scope.task_id,
                attempt_number=scope.attempt_number,
                assignment_token=scope.assignment_token,
                phase_recorder=scope.phase_recorder,
                grants_by_attempt={
                    **scope.grants_by_attempt,
                    attempt_number: attempt_grant,
                },
            )
            self._session_scopes[session_id] = updated
            return attempt_grant

    def grant_lock(self, session_id: UUID, attempt_number: int) -> asyncio.Lock:
        with self._lock:
            key = (session_id, attempt_number)
            lock = self._grant_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._grant_locks[key] = lock
            return lock


class PlatformToolProxyProxyToolInvoker(ToolInvoker):
    def __init__(
        self,
        *,
        local_invoker: ToolInvoker,
        platform_tool_proxy_platform: PlatformToolProxyPlatformPort,
        scopes: PlatformToolProxyScopeRegistry,
    ) -> None:
        self._local = local_invoker
        self._platform = platform_tool_proxy_platform
        self._scopes = scopes

    async def invoke(
        self,
        tool_name: ToolName,
        *,
        args: Sequence[JsonValue],
        kwargs: Mapping[str, JsonValue],
        context: ToolInvocationContext | None = None,
    ) -> object:
        if not _is_platform_tool_proxy_tool(tool_name):
            return await self._local.invoke(tool_name, args=args, kwargs=kwargs, context=context)
        if context is None:
            raise PlatformToolProxyControlError(
                "platform tool proxy execution requires tool invocation context"
            )
        scope = self._scopes.require_session(context.session_id)
        attempt_number = scope.attempt_number
        attempt_grant = scope.grants_by_attempt.get(attempt_number)
        if attempt_grant is None:
            async with self._scopes.grant_lock(context.session_id, attempt_number):
                scope = self._scopes.require_session(context.session_id)
                attempt_grant = scope.grants_by_attempt.get(attempt_number)
                if attempt_grant is None:
                    previous_phase: str | None = None
                    if scope.phase_recorder is not None:
                        previous_phase = scope.phase_recorder.mark("platform_tool_proxy_grant_create")
                    grant = await self._platform.create_platform_tool_proxy_grant(
                        batch_id=scope.batch_id,
                        artifact_id=scope.artifact_id,
                        task_id=scope.task_id,
                        validator_session_id=context.session_id,
                        attempt_number=attempt_number,
                        assignment_token=scope.assignment_token,
                    )
                    attempt_grant = self._scopes.store_session_grant(
                        session_id=context.session_id,
                        attempt_number=attempt_number,
                        token=grant.token,
                        expires_at=grant.expires_at,
                    )
                    if scope.phase_recorder is not None and previous_phase is not None:
                        scope.phase_recorder.mark(previous_phase)
        if attempt_grant is None:
            raise PlatformToolProxyControlError("platform tool proxy token is not registered for session")
        if _grant_expired(attempt_grant.expires_at):
            raise PlatformToolProxyTokenExpiredError("platform tool proxy token expired for session")
        if context.receipt_started_at is None or context.receipt_issued_at is None:
            raise PlatformToolProxyControlError("platform tool proxy receipt chronology is missing")
        previous_phase = None
        if scope.phase_recorder is not None:
            previous_phase = scope.phase_recorder.mark("platform_tool_proxy_execute")
        result = await self._platform.execute_platform_tool_proxy_tool(
            token=attempt_grant.token,
            uid=context.uid,
            artifact_id=scope.artifact_id,
            task_id=scope.task_id,
            validator_session_id=context.session_id,
            attempt_number=attempt_number,
            receipt_id=context.receipt_id,
            receipt_started_at=context.receipt_started_at,
            receipt_issued_at=context.receipt_issued_at,
            tool=tool_name,
            args=tuple(args),
            kwargs=dict(kwargs),
            transport_timeout_seconds=PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS,
        )
        if scope.phase_recorder is not None and previous_phase is not None:
            scope.phase_recorder.mark(previous_phase)
        return ToolInvocationOutput(
            public_payload=result.response,
            execution=result.execution,
            actual_cost_usd=result.actual_cost_usd,
            actual_cost_provider=result.actual_cost_provider,
            actual_cost_evidence=result.actual_cost_evidence,
        )


def _is_platform_tool_proxy_tool(tool_name: ToolName) -> bool:
    return is_search_tool(tool_name) or is_embedding_tool(tool_name) or tool_name == "llm_chat"


def _grant_expired(expires_at: datetime) -> bool:
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


__all__ = [
    "PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS",
    "PlatformToolProxyAttemptGrant",
    "PlatformToolProxyProxyToolInvoker",
    "PlatformToolProxyScopeRegistry",
    "PlatformToolProxySessionScope",
]
