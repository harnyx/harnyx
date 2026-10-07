from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolExecutionFacts
from harnyx_commons.tools.executor import ToolInvocationContext as _ToolInvocationContext
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


def ToolInvocationContext(**kwargs: object) -> _ToolInvocationContext:  # noqa: N802
    started_at = datetime.now(UTC)
    return _ToolInvocationContext(
        **kwargs,  # type: ignore[arg-type]
        receipt_started_at=started_at,
        receipt_issued_at=started_at,
    )


class _RecordingLocalInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(self, tool_name, *, args, kwargs, context=None):  # type: ignore[no-untyped-def]
        self.calls.append(tool_name)
        return {"local": True}


class _RecordingPhaseRecorder:
    def __init__(self) -> None:
        self.phase = "entrypoint_invocation"
        self.marks: list[str] = []

    def mark(self, phase: str) -> str:
        previous = self.phase
        self.phase = phase
        self.marks.append(phase)
        return previous


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
        receipt_started_at: datetime,
        receipt_issued_at: datetime,
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
                "receipt_started_at": receipt_started_at,
                "receipt_issued_at": receipt_issued_at,
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
            actual_cost_evidence={"settlement_source": "provider_returned"},
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
    receipt_started_at = call["receipt_started_at"]
    receipt_issued_at = call["receipt_issued_at"]
    assert platform.calls == [
        {
            "token": f"{_GRANT_VALUE}-2",
            "uid": 7,
            "artifact_id": artifact_id,
            "task_id": task_id,
            "validator_session_id": session_id,
            "attempt_number": 2,
            "receipt_id": receipt_id,
            "receipt_started_at": receipt_started_at,
            "receipt_issued_at": receipt_issued_at,
            "tool": "search_web",
            "args": (),
            "kwargs": {"provider": "parallel", "search_queries": ["harnyx"]},
            "transport_timeout_seconds": PLATFORM_TOOL_PROXY_EXECUTE_TRANSPORT_TIMEOUT_SECONDS,
        }
    ]
    assert result.public_payload == {"data": [{"url": "https://example.com"}]}
    assert result.actual_cost_usd == 0.25
    assert result.actual_cost_provider == "parallel"
    assert result.actual_cost_evidence == {"settlement_source": "provider_returned"}
    scope = scopes.require_session(session_id)
    assert scope.grants_by_attempt[2].token == f"{_GRANT_VALUE}-2"


async def test_platform_tool_proxy_successful_proxy_call_restores_attempt_phase() -> None:
    """Keep later sandbox timeouts attributed to miner invocation after a successful tool call."""

    batch_id = uuid4()
    session_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    phase_recorder = _RecordingPhaseRecorder()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
        phase_recorder=phase_recorder,
    )
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=_RecordingLocalInvoker(),
        platform_tool_proxy_platform=_RecordingPlatformToolProxyPlatform(calls=[], grants=[]),
        scopes=scopes,
    )

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

    assert phase_recorder.marks == [
        "platform_tool_proxy_grant_create",
        "entrypoint_invocation",
        "platform_tool_proxy_execute",
        "entrypoint_invocation",
    ]
    assert phase_recorder.phase == "entrypoint_invocation"


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


@pytest.mark.parametrize("tool", ["search_web", "decision_query"])
async def test_platform_tool_proxy_proxy_rejects_expired_cached_token_without_reissue(tool) -> None:
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

    dispatched = []
    with pytest.raises(PlatformToolProxyTokenExpiredError) as exc_info:
        await invoker.invoke(
            tool,
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]}
            if tool == "search_web"
            else {
                "provider": "openrouter",
                "model": "cloudflare/clef-flash",
                "state": "s",
                "questions": {"b": {"type": "boolean", "instructions": "?"}},
            },
            context=ToolInvocationContext(
                receipt_id=str(uuid4()),
                session_id=session_id,
                active_attempt=1,
                uid=7,
                on_provider_dispatch=lambda: dispatched.append(True),
            ),
        )

    assert platform.grants == []
    assert platform.calls == []
    assert dispatched == []
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


async def test_decision_query_uses_hosted_grant_instead_of_local_provider() -> None:
    session_id, batch_id, artifact_id, task_id = (uuid4() for _ in range(4))
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=batch_id,
        session_id=session_id,
        artifact_id=artifact_id,
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
        attempt_number=1,
    )
    platform = _RecordingPlatformToolProxyPlatform(calls=[], grants=[])
    local = _RecordingLocalInvoker()
    invoker = PlatformToolProxyProxyToolInvoker(
        local_invoker=local, platform_tool_proxy_platform=platform, scopes=scopes
    )
    kwargs = {
        "provider": "openrouter",
        "model": "cloudflare/clef-flash",
        "state": "s",
        "questions": {"b": {"type": "boolean", "instructions": "?"}},
    }
    dispatched = []
    await invoker.invoke(
        "decision_query",
        args=(),
        kwargs=kwargs,
        context=ToolInvocationContext(
            receipt_id=str(uuid4()),
            session_id=session_id,
            active_attempt=1,
            uid=7,
            on_provider_dispatch=lambda: dispatched.append(True),
        ),
    )
    assert dispatched == [True]
    assert local.calls == []
    assert len(platform.calls) == 1
    assert platform.calls[0]["tool"] == "decision_query"
    assert platform.calls[0]["kwargs"] == kwargs


@pytest.mark.parametrize(
    "failure",
    [
        "invalid_request",
        "miner_credential_missing",
        "platform_error",
        "platform_error_unpriced",
        "budget_exhausted",
        "concurrency_exhausted",
        "duplicate_call",
        "provider_failed",
        "billed_timeout",
        "billed_interruption",
        "tool_timeout",
        "transport",
        "missing_provider",
        "unknown_provider",
        "unsupported_provider",
    ],
)
async def test_hosted_decision_failure_preserves_prior_actual_total(failure: str) -> None:
    """HTTP rejection evidence must survive the hosted invoker and executor without hiding unknown I/O costs."""
    import bittensor as bt
    import httpx

    from harnyx_commons.domain.session import Session
    from harnyx_commons.domain.tool_call import ToolCallOutcome
    from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
    from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
    from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
    from harnyx_commons.tools.dto import ToolInvocationRequest
    from harnyx_commons.tools.executor import ToolExecutor
    from harnyx_commons.tools.usage_tracker import UsageTracker
    from harnyx_validator.application.services.evaluation_runner import _usage_from_receipts
    from harnyx_validator.infrastructure.tools.platform_client import AsyncPlatformToolProxyPlatformClient

    now = datetime.now(UTC)
    session_id, task_id = uuid4(), uuid4()
    sessions = InMemorySessionRegistry()
    sessions.create(
        Session(
            session_id=session_id,
            uid=7,
            task_id=task_id,
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
            budget_usd=1.0,
        )
    )
    tokens = InMemoryTokenRegistry()
    tokens.register(session_id, "token")
    receipts = InMemoryReceiptLog()
    scopes = PlatformToolProxyScopeRegistry()
    scopes.register_session(
        batch_id=uuid4(),
        session_id=session_id,
        artifact_id=uuid4(),
        task_id=task_id,
        assignment_token=_ASSIGNMENT_TOKEN,
    )
    scopes.store_session_grant(
        session_id=session_id, attempt_number=1, token=_GRANT_VALUE, expires_at=now + timedelta(minutes=5)
    )
    requests = 0
    local_rejection = failure in {"missing_provider", "unknown_provider", "unsupported_provider"}
    billed_failure = failure in {"provider_failed", "billed_timeout", "billed_interruption"}
    known_zero = local_rejection or failure in {
        "invalid_request",
        "miner_credential_missing",
        "platform_error",
        "budget_exhausted",
        "concurrency_exhausted",
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                200,
                json={
                    "response": {
                        "model": "cloudflare/clef-flash",
                        "answers": {"b": {"type": "boolean", "probability": 0.9}},
                    },
                    "execution": None,
                    "actual_cost_usd": 0.02,
                    "actual_cost_provider": "openrouter",
                    "actual_cost_evidence": {"source": "provider"},
                },
            )
        if failure == "transport":
            raise httpx.ReadError("response lost")
        error = {
            "error_code": {
                "platform_error_unpriced": "platform_error",
                "billed_timeout": "tool_timeout",
                "billed_interruption": "platform_interrupted",
            }.get(failure, failure),
            "message": "decision rejected",
        }
        if known_zero:
            error["billing"] = {"actual_cost_usd": 0.0, "actual_cost_provider": "openrouter"}
        elif billed_failure:
            error["billing"] = {
                "actual_cost_usd": 0.01,
                "actual_cost_provider": "openrouter",
                "usage": {"input_tokens": 100, "output_tokens": 0},
            }
        return httpx.Response(400, json=error)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic()),
        transport=httpx.MockTransport(handler),
    )
    executor = ToolExecutor(
        session_registry=sessions,
        receipt_log=receipts,
        usage_tracker=UsageTracker(),
        token_registry=tokens,
        clock=lambda: now,
        tool_invoker=PlatformToolProxyProxyToolInvoker(
            local_invoker=_RecordingLocalInvoker(), platform_tool_proxy_platform=client, scopes=scopes
        ),
    )
    request = ToolInvocationRequest(
        session_id,
        "token",
        "decision_query",
        (),
        {
            "provider": "openrouter",
            "model": "cloudflare/clef-flash",
            "state": "s",
            "questions": {"b": {"type": "boolean", "instructions": "?"}},
        },
    )
    try:
        await executor.execute(request)
        if local_rejection:
            invalid_kwargs = dict(request.kwargs)
            if failure == "missing_provider":
                invalid_kwargs.pop("provider")
            else:
                invalid_kwargs["provider"] = "unknown" if failure == "unknown_provider" else "chutes"
            request = replace(request, kwargs=invalid_kwargs)
        with pytest.raises(ValueError if local_rejection else RuntimeError):
            await executor.execute(request)
    finally:
        await client.aclose()
    updated = sessions.get(session_id)
    assert updated is not None
    assert updated.usage.total_cost_usd == (0.03 if billed_failure else 0.02)
    assert updated.usage.actual_total_cost_usd == (0.03 if billed_failure else 0.02 if known_zero else None)
    failed = next(call for call in receipts.for_session(session_id) if call.outcome is not ToolCallOutcome.OK)
    assert failed.details.actual_cost_usd == (0.01 if billed_failure else 0.0 if known_zero else None)
    assert (failed.details.extra.get("actual_cost_settlement_source") == "unavailable") is (
        not known_zero and not billed_failure
    )
    if billed_failure:
        assert failed.details.response_payload == {"usage": {"input_tokens": 100, "output_tokens": 0}}
    assert updated.usage.llm_usage_totals == {}
    reconstructed = _usage_from_receipts(tuple(receipts.for_session(session_id)))
    assert reconstructed.total_cost_usd == updated.usage.total_cost_usd
    assert reconstructed.actual_total_cost_usd == updated.usage.actual_total_cost_usd
    assert failed.outcome is (
        ToolCallOutcome.PROVIDER_ERROR
        if failure == "provider_failed"
        else ToolCallOutcome.TIMEOUT
        if failure in {"tool_timeout", "billed_timeout"}
        else ToolCallOutcome.BUDGET_EXCEEDED
        if failure == "budget_exhausted"
        else ToolCallOutcome.INTERNAL_ERROR
    )
    assert requests == (1 if local_rejection else 2)
