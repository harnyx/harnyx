from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import bittensor as bt
import httpx
import pytest

from harnyx_commons.bittensor import build_canonical_request
from harnyx_commons.domain.miner_task import EvaluationDetails, Response, ScoreBreakdown
from harnyx_commons.domain.session import Session, SessionStatus
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome
from harnyx_commons.errors import BudgetExceededError, ToolInvocationTimeoutError, ToolProviderError
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskRunSubmission,
    PlatformOwnedTaskResult,
    SandboxFailureDiagnostics,
    TokenUsageSummary,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.application.ports.platform import (
    PlatformTaskAttemptIdentity,
    PlatformWeightsUnavailableError,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.tools.platform_client import (
    AsyncPlatformToolProxyPlatformClient,
    HttpPlatformClient,
    PlatformClientError,
    PlatformToolProxyBudgetExceededError,
    PlatformToolProxyInterruptedError,
    PlatformToolProxyInvocationError,
    PlatformToolProxyProviderError,
    PlatformToolProxyToolTimeoutError,
)

_HEADER_PATTERN = re.compile(r'^Bittensor\s+ss58="(?P<ss58>[^"]+)",\s*sig="(?P<sig>[0-9a-f]+)"$')
_ASSIGNMENT_TOKEN = "assignment-token"  # noqa: S105 - fixed test-only assignment token


def _keypair() -> bt.Keypair:
    return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())


def _weights_response() -> httpx.Response:
    payload = {
        "weights": {"42": 0.7, "7": 0.3},
        "champion_uid": 42,
    }
    return httpx.Response(status_code=200, json=payload)


def _artifact_response(content: bytes) -> httpx.Response:
    return httpx.Response(status_code=200, content=content)


class _FlakyTransport:
    def __init__(
        self,
        *,
        first_exception: type[httpx.TransportError],
        success_response: httpx.Response,
    ) -> None:
        self._first_exception = first_exception
        self._success_response = success_response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise self._first_exception("timed out", request=request)
        return self._success_response


class _AlwaysFailTransport:
    def __init__(self, *, exceptions: list[httpx.TransportError]) -> None:
        self._exceptions = exceptions
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise self._exceptions[len(self.requests) - 1]


def _assert_signed(request: httpx.Request, keypair: bt.Keypair) -> None:
    header = request.headers.get("Authorization")
    assert header is not None
    match = _HEADER_PATTERN.match(header)
    assert match is not None
    assert match.group("ss58") == keypair.ss58_address
    path = request.url.raw_path.decode()
    query = request.url.query
    if query and "?" not in path:
        path = f"{path}?{query}"
    body = request.content or b""
    canonical = build_canonical_request(request.method, path, body)
    signature = bytes.fromhex(match.group("sig"))
    assert keypair.verify(canonical, signature)


def _tool_call(*, session_id, issued_at: datetime) -> ToolCall:
    return ToolCall(
        receipt_id="receipt-1",
        session_id=session_id,
        uid=7,
        tool="search_web",
        issued_at=issued_at,
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="request-hash",
            request_payload={"query": "smoke"},
            response_hash="response-hash",
            response_payload={"ok": True},
        ),
    )


def test_get_champion_weights_returns_weights() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == "/v1/weights":
            return _weights_response()
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.champion_uid == 42


def test_get_champion_weights_maps_weights_unavailable_response() -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        _assert_signed(request, keypair)
        if request.method == "GET" and request.url.path == "/v1/weights":
            return httpx.Response(
                status_code=503,
                json={
                    "error_code": "weights_unavailable",
                    "message": "participant emission unavailable",
                },
            )
        return httpx.Response(status_code=404)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformWeightsUnavailableError, match="participant emission unavailable"):
        client.get_champion_weights()

    assert [request.url.path for request in seen_requests] == ["/v1/weights"]


@pytest.mark.anyio("asyncio")
async def test_request_miner_task_work_posts_active_attempts_and_parses_assignments() -> None:
    keypair = _keypair()
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    session_id = uuid4()
    seen_body: dict[str, object] | None = None
    seen_timeout: dict[str, float] | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body, seen_timeout
        _assert_signed(request, keypair)
        if request.method == "POST" and request.url.path == "/v2/miner-task-work/tasks":
            seen_body = json.loads(request.content)
            seen_timeout = request.extensions.get("timeout")
            return httpx.Response(
                status_code=200,
                json={
                    "server_time": "2026-06-22T00:00:00+00:00",
                    "tasks": [
                        {
                            "batch_id": str(batch_id),
                            "artifact": {
                                "uid": 7,
                                "artifact_id": str(artifact_id),
                                "content_hash": "abc",
                                "size_bytes": 1,
                                "miner_hotkey_ss58": "miner-hotkey",
                            },
                            "task": {
                                "task_id": str(task_id),
                                "query": {"text": "smoke"},
                                "reference_answer": {"text": "ok"},
                                "budget_usd": 0.05,
                            },
                            "attempt_number": 2,
                            "max_attempts": 3,
                            "assignment_token": _ASSIGNMENT_TOKEN,
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    assignments = await client.request_miner_task_work(
        target_concurrency=4,
        max_active_artifacts=2,
        active_attempts=(
            PlatformTaskAttemptIdentity(
                batch_id=batch_id,
                artifact_id=artifact_id,
                task_id=task_id,
                attempt_number=1,
                validator_session_id=session_id,
            ),
        ),
    )

    assert seen_body == {
        "target_concurrency": 4,
        "max_active_artifacts": 2,
        "active_attempts": [
            {
                "batch_id": str(batch_id),
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "attempt_number": 1,
                "validator_session_id": str(session_id),
            }
        ],
    }
    assert seen_timeout == {
        "connect": 10.0,
        "read": 300.0,
        "write": 10.0,
        "pool": 10.0,
    }
    assert len(assignments) == 1
    assert assignments[0].artifact.artifact_id == artifact_id
    assert assignments[0].task.task_id == task_id
    assert assignments[0].attempt_number == 2
    assert assignments[0].max_attempts == 3
    assert assignments[0].assignment_token == _ASSIGNMENT_TOKEN


@pytest.mark.anyio("asyncio")
async def test_request_miner_task_work_cancels_http_request() -> None:
    keypair = _keypair()
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        assert request.method == "POST"
        assert request.url.path == "/v2/miner-task-work/tasks"
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    task = asyncio.create_task(
        client.request_miner_task_work(
            target_concurrency=1,
            max_active_artifacts=1,
            active_attempts=(),
        )
    )
    await asyncio.wait_for(request_started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(request_cancelled.wait(), timeout=1.0)


def test_submit_miner_task_work_results_posts_audit_only_retry_attempt() -> None:
    keypair = _keypair()
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    session_id = uuid4()
    started_at = datetime(2026, 6, 22, 0, 0, tzinfo=UTC)
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        _assert_signed(request, keypair)
        if request.method == "POST" and request.url.path == "/v2/miner-task-work/results":
            seen_body = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json={
                    "server_time": "2026-06-22T00:00:02+00:00",
                    "results": [
                        {
                            "batch_id": str(batch_id),
                            "artifact_id": str(artifact_id),
                            "task_id": str(task_id),
                            "attempt_number": 1,
                            "outcome": "accepted",
                            "canonical": False,
                            "reason_code": None,
                            "reason": None,
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    acknowledgements = client.submit_miner_task_work_results(
        (
            PlatformOwnedTaskResult(
                batch_id=batch_id,
                artifact_id=artifact_id,
                task_id=task_id,
                attempt_number=1,
                result=None,
                terminal_attempt=MinerTaskAttemptAuditRecord(
                    validator_session_id=session_id,
                    batch_id=batch_id,
                    artifact_id=artifact_id,
                    task_id=task_id,
                    attempt_number=1,
                    uid=7,
                    miner_hotkey_ss58="miner-hotkey",
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=1),
                    status=MinerTaskAttemptStatus.FAILED,
                    error_code="timeout_miner_owned",
                    error_summary_code="timeout_miner_owned",
                    retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
                    terminal_effect=None,
                    max_attempts=2,
                    execution_log=(_tool_call(session_id=session_id, issued_at=started_at),),
                ),
            ),
        )
    )

    assert seen_body is not None
    item = seen_body["results"][0]  # type: ignore[index]
    assert item["result"] is None
    assert item["terminal_attempt"]["validator_session_id"] == str(session_id)
    assert item["terminal_attempt"]["retry_decision"] == "will_retry"
    assert item["terminal_attempt"]["terminal_effect"] is None
    execution_log = item["terminal_attempt"]["execution_log"]
    assert len(execution_log) == 1
    assert execution_log[0]["receipt_id"] == "receipt-1"
    assert execution_log[0]["session_id"] == str(session_id)
    assert execution_log[0]["tool"] == "search_web"
    assert execution_log[0]["outcome"] == "ok"
    assert execution_log[0]["details"]["response_payload"] == {"ok": True}
    assert acknowledgements[0].outcome == "accepted"
    assert acknowledgements[0].canonical is False
    assert acknowledgements[0].reason_code is None


def test_submit_miner_task_work_results_posts_delivery_failure_detail() -> None:
    keypair = _keypair()
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    session_id = uuid4()
    started_at = datetime(2026, 6, 22, 0, 0, tzinfo=UTC)
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        _assert_signed(request, keypair)
        if request.method == "POST" and request.url.path == "/v2/miner-task-work/results":
            seen_body = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json={
                    "server_time": "2026-06-22T00:00:02+00:00",
                    "results": [
                        {
                            "batch_id": str(batch_id),
                            "artifact_id": str(artifact_id),
                            "task_id": str(task_id),
                            "attempt_number": 1,
                            "outcome": "accepted",
                            "canonical": True,
                            "reason_code": None,
                            "reason": None,
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )
    client.submit_miner_task_work_results(
        (
            PlatformOwnedTaskResult(
                batch_id=batch_id,
                artifact_id=artifact_id,
                task_id=task_id,
                attempt_number=1,
                result=None,
                terminal_attempt=MinerTaskAttemptAuditRecord(
                    validator_session_id=session_id,
                    batch_id=batch_id,
                    artifact_id=artifact_id,
                    task_id=task_id,
                    attempt_number=1,
                    uid=7,
                    miner_hotkey_ss58="miner-hotkey",
                    started_at=started_at,
                    finished_at=started_at + timedelta(seconds=1),
                    status=MinerTaskAttemptStatus.FAILED,
                    error_code="sandbox_start_failed",
                    error_summary_code="sandbox_start_failed",
                    retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
                    terminal_effect=MinerTaskAttemptTerminalEffect.DELIVERY_FAILURE,
                    max_attempts=1,
                    delivery_failure_detail=ValidatorBatchFailureDetail(
                        error_code="sandbox_start_failed",
                        error_message="sandbox start failed",
                        occurred_at=started_at,
                        artifact_id=artifact_id,
                        task_id=task_id,
                        uid=7,
                        sandbox_diagnostics=SandboxFailureDiagnostics(
                            exit_code=255,
                            docker_inspect_error_tail="command=docker inspect stderr=No such container",
                            docker_logs_error_tail="command=docker logs stderr=daemon unavailable",
                        ),
                    ),
                ),
            ),
        )
    )

    assert seen_body is not None
    attempt = seen_body["results"][0]["terminal_attempt"]  # type: ignore[index]
    assert attempt["delivery_failure_detail"]["error_message"] == "sandbox start failed"
    assert attempt["delivery_failure_detail"]["sandbox_diagnostics"]["exit_code"] == 255
    assert attempt["delivery_failure_detail"]["sandbox_diagnostics"]["docker_inspect_error_tail"] == (
        "command=docker inspect stderr=No such container"
    )
    assert attempt["delivery_failure_detail"]["sandbox_diagnostics"]["docker_logs_error_tail"] == (
        "command=docker logs stderr=daemon unavailable"
    )


def test_submit_miner_task_work_results_serializes_run_execution_log() -> None:
    keypair = _keypair()
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    session_id = uuid4()
    started_at = datetime(2026, 6, 22, 0, 0, tzinfo=UTC)
    seen_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_body
        _assert_signed(request, keypair)
        if request.method == "POST" and request.url.path == "/v2/miner-task-work/results":
            seen_body = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json={
                    "server_time": "2026-06-22T00:00:02+00:00",
                    "results": [
                        {
                            "batch_id": str(batch_id),
                            "artifact_id": str(artifact_id),
                            "task_id": str(task_id),
                            "attempt_number": 1,
                            "outcome": "accepted",
                            "canonical": True,
                            "reason_code": None,
                            "reason": None,
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404)

    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )
    session = Session(
        session_id=session_id,
        uid=7,
        task_id=task_id,
        issued_at=started_at,
        expires_at=started_at + timedelta(minutes=5),
        budget_usd=0.05,
        status=SessionStatus.COMPLETED,
    )
    result = PlatformOwnedTaskResult(
        batch_id=batch_id,
        artifact_id=artifact_id,
        task_id=task_id,
        attempt_number=1,
        result=MinerTaskRunSubmission(
            batch_id=batch_id,
            run=MinerTaskRun(
                session_id=session_id,
                uid=7,
                artifact_id=artifact_id,
                task_id=task_id,
                response=Response(text="answer"),
                details=EvaluationDetails(
                    score_breakdown=ScoreBreakdown(
                        comparison_score=1.0,
                        total_score=1.0,
                        scoring_version="v1",
                    ),
                ),
                completed_at=started_at + timedelta(seconds=1),
            ),
            score=1.0,
            execution_log=(_tool_call(session_id=session_id, issued_at=started_at),),
            usage=TokenUsageSummary.empty(),
            session=session,
        ),
        terminal_attempt=MinerTaskAttemptAuditRecord(
            validator_session_id=session_id,
            batch_id=batch_id,
            artifact_id=artifact_id,
            task_id=task_id,
            attempt_number=1,
            uid=7,
            miner_hotkey_ss58="miner-hotkey",
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=1),
            status=MinerTaskAttemptStatus.SUCCEEDED,
            error_code=None,
            error_summary_code=None,
            retry_decision=MinerTaskAttemptRetryDecision.WILL_NOT_RETRY,
            terminal_effect=MinerTaskAttemptTerminalEffect.TASK_RESULT,
            max_attempts=1,
            execution_log=(),
        ),
    )

    acknowledgements = client.submit_miner_task_work_results((result,))

    assert seen_body is not None
    item = seen_body["results"][0]  # type: ignore[index]
    assert "validator" not in item["result"]
    execution_log = item["result"]["execution_log"]
    assert len(execution_log) == 1
    assert execution_log[0]["receipt_id"] == "receipt-1"
    assert execution_log[0]["session_id"] == str(session_id)
    assert execution_log[0]["tool"] == "search_web"
    assert execution_log[0]["outcome"] == "ok"
    assert execution_log[0]["details"]["request_payload"] == {"query": "smoke"}
    assert acknowledgements[0].outcome == "accepted"
    assert acknowledgements[0].canonical is True



async def test_platform_tool_proxy_grant_posts_attempt_number() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    task_id = uuid4()
    validator_session_id = uuid4()

    async def handler(request: httpx.Request) -> httpx.Response:
        _assert_signed(request, keypair)
        assert request.method == "POST"
        assert request.url.path == "/v1/platform-tool-proxy/grants"
        assert json.loads(request.content) == {
            "batch_id": str(batch_id),
            "artifact_id": str(artifact_id),
            "task_id": str(task_id),
            "validator_session_id": str(validator_session_id),
            "attempt_number": 2,
            "assignment_token": _ASSIGNMENT_TOKEN,
        }
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    keypair = _keypair()
    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=batch_id,
        artifact_id=artifact_id,
        task_id=task_id,
        validator_session_id=validator_session_id,
        attempt_number=2,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_retries_transient_connect_timeout_then_succeeds() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectTimeout("grant connect timed out", request=request)
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0,),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=uuid4(),
        artifact_id=uuid4(),
        task_id=uuid4(),
        validator_session_id=uuid4(),
        attempt_number=1,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
async def test_platform_tool_proxy_grant_retries_transient_status_then_succeeds(status_code: int) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status_code=status_code, json={"error_code": "temporary_platform_error"})
        return httpx.Response(
            status_code=200,
            json={"token": "platform-tool-proxy-token", "expires_at": "2026-05-30T12:15:00+00:00"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0,),
    )

    grant = await client.create_platform_tool_proxy_grant(
        batch_id=uuid4(),
        artifact_id=uuid4(),
        task_id=uuid4(),
        validator_session_id=uuid4(),
        attempt_number=1,
        assignment_token=_ASSIGNMENT_TOKEN,
    )

    assert grant.token == "platform-tool-proxy-token"  # noqa: S105 - fixed test-only proxy token
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_retry_exhaustion_maps_to_grant_failed() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectTimeout("grant connect timed out", request=request)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_transient_status_retry_exhaustion_maps_to_grant_failed() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=503, json={"error_code": "temporary_platform_error"})

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"
    assert [request.url.path for request in requests] == [
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
        "/v1/platform-tool-proxy/grants",
    ]


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_grant_does_not_retry_deterministic_response() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=403,
            json={"error_code": "platform_tool_proxy_denied", "message": "grant denied"},
        )

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.error_code == "platform_tool_proxy_denied"
    assert len(requests) == 1


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("error_code", "status_code"),
    [
        ("platform_tool_proxy_denied", 403),
        ("platform_tool_proxy_grant_failed", 400),
    ],
)
async def test_platform_tool_proxy_grant_preserves_proxy_error_code(
    error_code: str,
    status_code: int,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return httpx.Response(
                status_code=status_code,
                json={"error_code": error_code, "message": f"{error_code} message"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.error_code == error_code
    assert str(exc_info.value) == f"{error_code} message"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=500, text="not json"),
        httpx.Response(status_code=500, json={"error_code": "unknown_platform_error"}),
    ],
)
async def test_platform_tool_proxy_grant_unknown_error_response_preserves_grant_failed_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
        grant_retry_delays_seconds=(),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=200, text="not json"),
        httpx.Response(status_code=200, json={"token": "grant"}),
    ],
)
async def test_platform_tool_proxy_grant_invalid_success_response_preserves_grant_failed_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/grants":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.create_platform_tool_proxy_grant(
            batch_id=uuid4(),
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            assignment_token=_ASSIGNMENT_TOKEN,
        )

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_code == "platform_tool_proxy_grant_failed"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_tool_timeout_error_code_to_timeout_exception() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=408,
                json={"error_code": "tool_timeout", "message": "search_web timed out after 1 second"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyToolTimeoutError, match="search_web timed out") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolInvocationTimeoutError)
    assert exc_info.value.status_code == 408
    assert exc_info.value.error_code == "tool_timeout"
    assert requests[0].headers["x-platform-tool-proxy-token"] == "proxy-token"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_read_timeout_to_tool_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise httpx.ReadTimeout("platform execution timed out", request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyToolTimeoutError, match="awaiting tool result") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolInvocationTimeoutError)
    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "tool_timeout"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("exception_type", "message"),
    [
        (httpx.RemoteProtocolError, "server disconnected without response"),
        (httpx.ReadError, "response stream closed"),
    ],
)
async def test_platform_tool_proxy_execute_maps_response_side_interruption_to_platform_interrupted(
    exception_type: type[httpx.HTTPError],
    message: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise exception_type(message, request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInterruptedError, match="interrupted before a response") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_interrupted"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_platform_interrupted_error_code_to_interrupted_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={
                    "error_code": "platform_interrupted",
                    "message": "platform tool proxy execution interrupted before completion",
                },
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInterruptedError, match="interrupted before completion") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_interrupted"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    ("exception_type", "message"),
    [
        (httpx.ConnectTimeout, "platform endpoint unavailable"),
        (httpx.WriteError, "request body write failed"),
    ],
)
async def test_platform_tool_proxy_execute_keeps_pre_response_start_failures_validator_owned(
    exception_type: type[httpx.HTTPError],
    message: str,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            raise exception_type(message, request=request)
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 0
    assert exc_info.value.error_code == "platform_tool_proxy_execution_failed"
    assert len(requests) == 1


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_provider_failed_to_tool_provider_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=502,
                json={"error_code": "provider_failed", "message": "provider rejected"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyProviderError, match="provider rejected") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, ToolProviderError)
    assert exc_info.value.status_code == 502
    assert exc_info.value.error_code == "provider_failed"


@pytest.mark.anyio("asyncio")
async def test_platform_tool_proxy_execute_maps_budget_exhausted_to_budget_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={"error_code": "budget_exhausted", "message": "grant budget exhausted"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyBudgetExceededError, match="grant budget exhausted") as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert isinstance(exc_info.value, BudgetExceededError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "budget_exhausted"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_denied",
        "miner_credential_missing",
        "concurrency_exhausted",
        "unsupported_provider",
        "unsupported_model",
        "invalid_request",
        "platform_error",
        "platform_tool_proxy_execution_failed",
        "duplicate_call",
    ],
)
async def test_platform_tool_proxy_execute_maps_proxy_policy_errors_to_non_provider_error(
    error_code: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return httpx.Response(
                status_code=400,
                json={"error_code": error_code, "message": f"{error_code} message"},
            )
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == error_code
    assert str(exc_info.value) == f"{error_code} message"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=500, text="not json"),
        httpx.Response(status_code=500, json={"error_code": "unknown_platform_error"}),
    ],
)
async def test_platform_tool_proxy_execute_unknown_error_response_preserves_platform_error_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_code == "platform_error"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(status_code=200, text="not json"),
        httpx.Response(status_code=200, json={"execution": {"elapsed_ms": 1.0}}),
    ],
)
async def test_platform_tool_proxy_execute_invalid_success_response_preserves_platform_error_metadata(
    response: httpx.Response,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/platform-tool-proxy/tools/execute":
            return response
        return httpx.Response(status_code=404)

    client = AsyncPlatformToolProxyPlatformClient(
        base_url="https://mock.local",
        hotkey=_keypair(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformToolProxyInvocationError) as exc_info:
        await client.execute_platform_tool_proxy_tool(
            token="proxy-token",  # noqa: S106 - fixed test-only proxy token
            uid=7,
            artifact_id=uuid4(),
            task_id=uuid4(),
            validator_session_id=uuid4(),
            attempt_number=1,
            receipt_id=str(uuid4()),
            tool="search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["harnyx"]},
            transport_timeout_seconds=11.0,
        )

    assert exc_info.value.status_code == 200
    assert exc_info.value.error_code == "platform_error"


def test_get_champion_weights_retries_transient_connect_timeout() -> None:
    keypair = _keypair()
    transport = _FlakyTransport(
        first_exception=httpx.ConnectTimeout,
        success_response=_weights_response(),
    )
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    weights = client.get_champion_weights()

    assert weights.weights == {42: 0.7, 7: 0.3}
    assert weights.champion_uid == 42
    assert [request.url.path for request in transport.requests] == [
        "/v1/weights",
        "/v1/weights",
    ]
    for request in transport.requests:
        _assert_signed(request, keypair)



def test_fetch_artifact_retries_transient_connect_timeout() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    content = b"artifact"
    keypair = _keypair()
    transport = _FlakyTransport(
        first_exception=httpx.ConnectTimeout,
        success_response=_artifact_response(content),
    )
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(transport),
    )

    fetched = client.fetch_artifact(batch_id, artifact_id)

    expected_path = f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}"
    assert fetched == content
    assert [request.url.path for request in transport.requests] == [
        expected_path,
        expected_path,
    ]
    for request in transport.requests:
        _assert_signed(request, keypair)



def test_get_champion_weights_does_not_retry_non_connect_transport_failure() -> None:
    requests: list[httpx.Request] = []
    read_timeout = httpx.ReadTimeout("read timed out")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_signed(request, keypair)
        raise read_timeout

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.ReadTimeout) as exc_info:
        client.get_champion_weights()

    assert exc_info.value is read_timeout
    assert [request.url.path for request in requests] == ["/v1/weights"]


def test_fetch_artifact_does_not_retry_http_status_failure() -> None:
    batch_id = uuid4()
    artifact_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        _assert_signed(request, keypair)
        return httpx.Response(status_code=500)

    keypair = _keypair()
    client = HttpPlatformClient(
        base_url="https://mock.local",
        hotkey=keypair,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PlatformClientError):
        client.fetch_artifact(batch_id, artifact_id)

    assert [request.url.path for request in requests] == [
        f"/v1/miner-task-batches/{batch_id}/artifacts/{artifact_id}",
    ]
