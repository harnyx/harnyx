from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.tools.executor import ToolInvocationContext
from harnyx_validator.application.platform_tool_proxy import (
    PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS,
    PlatformToolProxyProxyToolInvoker,
    PlatformToolProxyScopeRegistry,
)
from harnyx_validator.application.ports.platform import (
    PlatformToolProxyControlError,
    PlatformToolProxyGrant,
    PlatformToolProxyTokenExpiredError,
    PlatformToolProxyToolResult,
)

pytestmark = pytest.mark.anyio("asyncio")

_GRANT_VALUE = "platform-tool-proxy-grant"
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token


class _RecordingLocalInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, tool_name, *, args, kwargs, context=None):  # type: ignore[no-untyped-def]
        self.calls.append(tool_name)
        return {"local": True}


@dataclass(slots=True)
class _RecordingPlatformToolProxyPlatform:
    calls: list[dict[str, object]]
    grants: list[dict[str, object]]
    grant_delay_seconds: float = 0.0

    async def create_platform_tool_proxy_grant(
        self,
        *,
        batch_id,
        artifact_id,
        task_id,
        validator_session_id,
        attempt_number,
        assignment_token,
    ):  # type: ignore[no-untyped-def]
        if self.grant_delay_seconds:
            await asyncio.sleep(self.grant_delay_seconds)
        token = f"{_GRANT_VALUE}-{attempt_number}"
        self.grants.append(
            {
                "batch_id": batch_id,
                "artifact_id": artifact_id,
                "task_id": task_id,
                "validator_session_id": validator_session_id,
                "attempt_number": attempt_number,
                "assignment_token": assignment_token,
            }
        )
        return PlatformToolProxyGrant(token=token, expires_at=datetime.now(UTC) + timedelta(minutes=5))

    async def execute_platform_tool_proxy_tool(
        self,
        *,
        token: str,
        uid: int,
        artifact_id,
        task_id,
        validator_session_id,
        attempt_number: int,
        receipt_id: str,
        tool: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        transport_timeout_seconds: float,
    ) -> PlatformToolProxyToolResult:
        self.calls.append(
            {
                "token": token,
                "uid": uid,
                "artifact_id": artifact_id,
                "task_id": task_id,
                "validator_session_id": validator_session_id,
                "attempt_number": attempt_number,
                "receipt_id": receipt_id,
                "tool": tool,
                "args": args,
                "kwargs": kwargs,
                "transport_timeout_seconds": transport_timeout_seconds,
            }
        )
        return PlatformToolProxyToolResult(
            response={"data": [{"url": "https://example.com"}]},
            execution=ToolExecutionFacts(),
            actual_cost_usd=0.25,
            actual_cost_provider="parallel",
        )


async def test_platform_tool_proxy_proxy_forwards_provider_tool_with_session_scope() -> None:
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
        attempt_number=2,
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    local = _RecordingLocalInvoker()
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=local,
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    result = await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
        context=ToolInvocationContext(
            receipt_id=str(uuid4()),
            session_id=session_id,
            active_attempt=2,
            uid=7,
        ),
    )

    assert local.calls == []
    assert platform.grants == [
        {
            "batch_id": batch_id,
            "artifact_id": artifact_id,
            "task_id": task_id,
                "validator_session_id": session_id,
                "attempt_number": 2,
                "assignment_token": _ASSIGNMENT_TOKEN,
            }
        ]
    call = platform.calls[0]
    receipt_id = call["receipt_id"]
    assert platform.calls == [
        {
            "token": f"{_GRANT_VALUE}-2",
            "uid": 7,
            "artifact_id": artifact_id,
            "task_id": task_id,
            "validator_session_id": session_id,
            "attempt_number": 2,
            "receipt_id": receipt_id,
            "tool": "search_web",
            "args": (),
            "kwargs": {"provider": "parallel", "search_queries": ["harnyx"]},
            "transport_timeout_seconds": PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS,
        }
    ]
    assert result.public_payload == {"data": [{"url": "https://example.com"}]}
    assert result.actual_cost_usd == 0.25
    assert result.actual_cost_provider == "parallel"
    scope = scopes.require_session(session_id)
    assert scope.grants_by_attempt[2].token == f"{_GRANT_VALUE}-2"


async def test_platform_tool_proxy_proxy_uses_fixed_execute_transport_timeout() -> None:
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "desearch", "search_queries": ["harnyx"], "timeout": 1.0},
        context=ToolInvocationContext(
            receipt_id=str(uuid4()),
            session_id=session_id,
            active_attempt=1,
            uid=7,
        ),
    )

    assert platform.calls[0]["transport_timeout_seconds"] == (
        PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS
    )
    assert platform.calls[0]["transport_timeout_seconds"] > 1.0


async def test_platform_tool_proxy_proxy_serializes_concurrent_first_grant_creation() -> None:
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[], grant_delay_seconds=0.01)
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    async def invoke_once() -> object:
        return await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=ToolInvocationContext(
                receipt_id=str(uuid4()),
                session_id=session_id,
                active_attempt=1,
                uid=7,
            ),
        )

    await asyncio.gather(invoke_once(), invoke_once())

    assert len(platform.grants) == 1
    assert len(platform.calls) == 2
    assert platform.grants[0]["attempt_number"] == 1


async def test_platform_tool_proxy_proxy_rejects_expired_cached_token_without_reissue() -> None:
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    scopes.store_session_grant(
        session_id=session_id,
        attempt_number=1,
        token=_GRANT_VALUE,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    with pytest.raises(PlatformToolProxyTokenExpiredError) as exc_info:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=ToolInvocationContext(
                receipt_id=str(uuid4()),
                session_id=session_id,
                active_attempt=1,
                uid=7,
            ),
        )

    assert platform.grants == []
    assert platform.calls == []
    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert exc_info.value.status_code == 403


async def test_platform_tool_proxy_proxy_mints_new_grant_for_later_attempt() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    session_ids = (uuid4(), uuid4())
    for attempt_number, session_id in enumerate(session_ids, start=1):
        scopes.register_session(
            batch_id=batch_id,
            session_id=session_id,
            artifact_id=artifact_id,
            task_id=task_id,
            assignment_token=_ASSIGNMENT_TOKEN,
            attempt_number=attempt_number,
        )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    for session_id in session_ids:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=ToolInvocationContext(
                receipt_id=str(uuid4()),
                session_id=session_id,
                active_attempt=1,
                uid=7,
            ),
        )

    assert [grant["attempt_number"] for grant in platform.grants] == [1, 2]
    assert [call["token"] for call in platform.calls] == [f"{_GRANT_VALUE}-1", f"{_GRANT_VALUE}-2"]


async def test_platform_tool_proxy_proxy_allows_later_attempt_after_earlier_attempt_token_expires() -> None:
    batch_id = uuid4()
    expired_session_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=expired_session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
        attempt_number=1,
    )
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
        attempt_number=2,
    )
    scopes.store_session_grant(
        session_id=expired_session_id,
        attempt_number=1,
        token="expired-attempt-1",  # noqa: S106 - fixed test-only proxy token
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
        context=ToolInvocationContext(
            receipt_id=str(uuid4()),
            session_id=session_id,
            active_attempt=2,
            uid=7,
        ),
    )

    assert [grant["attempt_number"] for grant in platform.grants] == [2]
    assert platform.calls[0]["token"] == f"{_GRANT_VALUE}-2"


async def test_platform_tool_proxy_missing_scope_preserves_denied_metadata() -> None:
    session_id = uuid4()
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=PlatformToolProxyScopeRegistry(),
    )

    with pytest.raises(PlatformToolProxyControlError) as exc_info:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=ToolInvocationContext(
                receipt_id=str(uuid4()),
                session_id=session_id,
                active_attempt=1,
                uid=7,
            ),
        )

    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert exc_info.value.status_code == 403
    assert platform.grants == []
    assert platform.calls == []


async def test_platform_tool_proxy_missing_context_preserves_denied_metadata() -> None:
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=PlatformToolProxyScopeRegistry(),
    )

    with pytest.raises(PlatformToolProxyControlError) as exc_info:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            context=None,
        )

    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert exc_info.value.status_code == 403
    assert platform.grants == []
    assert platform.calls == []


async def test_platform_tool_proxy_proxy_forwards_invalid_provider_selection_to_platform() -> None:
    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    await invoker.invoke(
        "search_web",
        args=(),
        kwargs={"provider": "chutes", "search_queries": ["harnyx"], "timeout": 3.5},
        context=ToolInvocationContext(
            receipt_id=str(uuid4()),
            session_id=session_id,
            active_attempt=1,
            uid=7,
        ),
    )

    assert len(platform.calls) == 1
    call = platform.calls[0]
    assert call["tool"] == "search_web"
    assert call["kwargs"] == {"provider": "chutes", "search_queries": ["harnyx"], "timeout": 3.5}
    assert call["transport_timeout_seconds"] == PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS


async def test_platform_tool_proxy_proxy_keeps_local_tools_local() -> None:
    scopes = PlatformToolProxyScopeRegistry()
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    local = _RecordingLocalInvoker()
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=local,
        platform_tool_proxy_platform=platform,
        scopes=scopes,
    )

    result = await invoker.invoke("test_tool", args=(), kwargs={}, context=None)

    assert result == {"local": True}
    assert local.calls == ["test_tool"]
    assert platform.calls == []
