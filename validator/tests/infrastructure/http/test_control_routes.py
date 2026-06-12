from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

import harnyx_validator.infrastructure.http.routes as routes_mod
from harnyx_commons.bittensor import VerificationError
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
    ScorerReasoning,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionUsage
from harnyx_commons.domain.tool_call import (
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolExecutionFacts,
)
from harnyx_commons.domain.tool_usage import (
    LlmModelUsageCost,
    LlmUsageSummary,
    SearchToolUsageSummary,
    ToolUsageSummary,
)
from harnyx_validator.application.accept_batch import AcceptEvaluationBatch
from harnyx_validator.application.dto.evaluation import (
    MinerTaskAttemptAuditRecord,
    MinerTaskAttemptRetryDecision,
    MinerTaskAttemptStatus,
    MinerTaskAttemptTerminalEffect,
    MinerTaskBatchSpec,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TokenUsageSummary,
)
from harnyx_validator.application.restore_batch import RestoreEvaluationBatch
from harnyx_validator.application.services.evaluation_runner import ValidatorBatchFailureDetail
from harnyx_validator.application.status import BatchActivityTracker, StatusProvider
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.http.middleware import request_logging_middleware
from harnyx_validator.infrastructure.http.routes import (
    ControlRouteAuth,
    ValidatorControlDeps,
    add_control_routes,
)
from harnyx_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from harnyx_validator.infrastructure.state.run_progress import (
    FileBackedRunProgress,
    ProgressCursorBeforeRestoreFloorError,
)
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageSnapshot


def _create_test_app(
    provider: DemoControlDependencyProvider,
) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)
    add_control_routes(app, provider)
    return app


@dataclass(frozen=True)
class _StubHotkey:
    ss58_address: str = "5validator"

    def sign(self, payload: bytes) -> bytes:
        return b"sig:" + payload


class StubAcceptBatch:
    def __init__(
        self,
        *,
        lifecycle: str | None = "processing",
        error_code: str | None = None,
        failure_detail: ValidatorBatchFailureDetail | None = None,
    ) -> None:
        self.received_batch: MinerTaskBatchSpec | None = None
        self.received_restore_runs: tuple[MinerTaskRunSubmission, ...] = ()
        self.received_restore_provider_evidence: tuple[dict[str, object], ...] = ()
        self._lifecycle = lifecycle
        self._error_code = error_code
        self._failure_detail = failure_detail

    def execute(
        self,
        batch: object,
        *,
        restore_runs: tuple[MinerTaskRunSubmission, ...] = (),
        restore_provider_evidence: tuple[dict[str, object], ...] = (),
    ) -> None:
        if not isinstance(batch, MinerTaskBatchSpec):
            raise AssertionError(f"expected MinerTaskBatchSpec, got {type(batch)!r}")
        self.received_batch = batch
        self.received_restore_runs = restore_runs
        self.received_restore_provider_evidence = restore_provider_evidence
        return None

    def lifecycle_for(self, batch_id: UUID) -> str | None:
        _ = batch_id
        return self._lifecycle

    def public_lifecycle_for(self, batch_id: UUID) -> str | None:
        lifecycle = self.lifecycle_for(batch_id)
        if lifecycle == "restoring":
            return "queued"
        return lifecycle

    def error_code_for(self, batch_id: UUID) -> str | None:
        _ = batch_id
        return self._error_code

    def failure_detail_for(self, batch_id: UUID) -> ValidatorBatchFailureDetail | None:
        _ = batch_id
        return self._failure_detail


class StubStatusProvider:
    def snapshot(self) -> dict[str, object]:
        return {"status": "ok", "running": False}


class StubResourceUsageProvider:
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        return ValidatorResourceUsageSnapshot(
            captured_at=datetime(2026, 3, 31, 6, 42, tzinfo=UTC),
            cpu_percent=12.5,
            cpu_capacity_cores=4.0,
            memory_used_bytes=512,
            memory_total_bytes=2048,
            memory_percent=25.0,
            disk_used_bytes=4096,
            disk_total_bytes=8192,
            disk_percent=50.0,
        )


class StubRestoreBatch:
    def __init__(self, accept_batch: StubAcceptBatch) -> None:
        self._accept_batch = accept_batch
        self.received_batch: MinerTaskBatchSpec | None = None

    def accept(self, batch: MinerTaskBatchSpec) -> object:
        self.received_batch = batch
        self._accept_batch.execute(batch)
        return type("Decision", (), {"batch_id": batch.batch_id, "should_start_restore": True})()


class StubRestoreWorker:
    def __init__(self) -> None:
        self.received_decisions: list[object] = []

    def request_restore(self, decision: object) -> bool:
        if not bool(getattr(decision, "should_start_restore", False)):
            return False
        self.received_decisions.append(decision)
        return True


class _ExplodingStatusProvider:
    def snapshot(self) -> dict[str, object]:
        raise RuntimeError("status exploded")


class _ExplodingResourceUsageProvider:
    def snapshot(self) -> ValidatorResourceUsageSnapshot:
        raise RuntimeError("resource usage exploded")


RunProgressFixture = dict[str, object]


class FakeProgressTracker:
    def __init__(self, *, snapshot: RunProgressFixture) -> None:
        self._snapshot = snapshot

    def summary(self, _: UUID) -> dict[str, object]:
        return {
            "batch_id": self._snapshot["batch_id"],
            "total": self._snapshot["total"],
            "completed": self._snapshot["completed"],
            "remaining": self._snapshot["remaining"],
            "latest_sequence": len(tuple(self._snapshot["run_page_submissions"])),
            "provider_evidence": self._snapshot["provider_evidence"],
        }

    def completed_run_page(
        self,
        _: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> dict[str, object]:
        page_error = self._snapshot.get("page_error")
        if isinstance(page_error, Exception):
            raise page_error
        runs = tuple(self._snapshot["run_page_submissions"])
        selected = tuple(
            {
                "sequence": sequence,
                "kind": "completed_run",
                "submission": submission,
                "attempt": None,
            }
            for sequence, submission in enumerate(runs, start=1)
            if sequence > after_sequence
        )[:limit]
        next_after_sequence = selected[-1]["sequence"] if selected else after_sequence
        return {
            "batch_id": self._snapshot["batch_id"],
            "after_sequence": after_sequence,
            "limit": limit,
            "latest_sequence": len(runs),
            "next_after_sequence": next_after_sequence,
            "has_more": next_after_sequence < len(runs),
            "items": selected,
        }


class DemoControlDependencyProvider:
    def __init__(
        self,
        *,
        snapshot: RunProgressFixture,
        auth: ControlRouteAuth | None = None,
        lifecycle: str | None = "processing",
        error_code: str | None = None,
        failure_detail: ValidatorBatchFailureDetail | None = None,
        validator_hotkey: _StubHotkey | None = None,
        status_provider: StatusProvider | None = None,
        resource_usage_provider: object | None = None,
    ) -> None:
        self.accept_batch = StubAcceptBatch(
            lifecycle=lifecycle,
            error_code=error_code,
            failure_detail=failure_detail,
        )
        self.restore_batch = StubRestoreBatch(self.accept_batch)
        self.restore_worker = StubRestoreWorker()
        self._deps = ValidatorControlDeps(
            accept_batch=self.accept_batch,
            restore_batch=self.restore_batch,
            restore_worker=self.restore_worker,
            status_provider=StubStatusProvider() if status_provider is None else status_provider,
            auth=_allow_all_auth if auth is None else auth,
            progress_tracker=FakeProgressTracker(snapshot=snapshot),
            validator_hotkey=validator_hotkey or _StubHotkey(),
            resource_usage_provider=(
                StubResourceUsageProvider() if resource_usage_provider is None else resource_usage_provider
            ),
            batch_activity=BatchActivityTracker(),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


class RealAcceptBatchDependencyProvider:
    def __init__(self, *, storage_root: Path) -> None:
        self.validator_hotkey = _StubHotkey()
        self.inbox = InMemoryBatchInbox()
        self.status_provider = StatusProvider()
        self.progress_tracker = FileBackedRunProgress(storage_root=storage_root)
        self.accept_batch = AcceptEvaluationBatch(
            inbox=self.inbox,
            status=self.status_provider,
            progress=self.progress_tracker,
        )
        self.restore_batch = RestoreEvaluationBatch(
            accepted_batches=self.accept_batch,
            platform=object(),
            batch_activity=BatchActivityTracker(),
        )
        self.restore_worker = StubRestoreWorker()
        self._deps = ValidatorControlDeps(
            accept_batch=self.accept_batch,
            restore_batch=self.restore_batch,
            restore_worker=self.restore_worker,
            status_provider=self.status_provider,
            auth=_allow_all_auth,
            progress_tracker=self.progress_tracker,
            validator_hotkey=self.validator_hotkey,
            resource_usage_provider=StubResourceUsageProvider(),
            batch_activity=BatchActivityTracker(),
        )

    def __call__(self) -> ValidatorControlDeps:
        return self._deps


async def _allow_all_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    return "caller"


async def _explode_auth(_: str, __: str, ___: bytes, ____: str | None) -> str:
    raise RuntimeError("auth exploded")


async def _auth_unavailable(_: str, __: str, ___: bytes, ____: str | None) -> str:
    raise VerificationError(
        "auth_unavailable",
        "inbound auth verifier has not completed initial hotkey warmup",
    )


def test_status_endpoint_awaits_auth_with_request_primitives() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    auth_calls: list[tuple[str, str, bytes, str | None]] = []

    async def _record_auth(
        method: str,
        path_qs: str,
        body: bytes,
        authorization_header: str | None,
    ) -> str:
        auth_calls.append((method, path_qs, body, authorization_header))
        return "caller"

    provider = DemoControlDependencyProvider(snapshot=snapshot, auth=_record_auth)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status?verbose=1",
        headers={"Authorization": 'Bittensor ss58="5demo",sig="00"'},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["hotkey"] == "5validator"
    assert response.json()["resource_usage"] == {
        "captured_at": "2026-03-31T06:42:00+00:00",
        "cpu_percent": 12.5,
        "cpu_capacity_cores": 4.0,
        "memory_used_bytes": 512,
        "memory_total_bytes": 2048,
        "memory_percent": 25.0,
        "disk_used_bytes": 4096,
        "disk_total_bytes": 8192,
        "disk_percent": 50.0,
    }
    assert auth_calls == [
        (
            "GET",
            "/validator/status?verbose=1",
            b"",
            'Bittensor ss58="5demo",sig="00"',
        )
    ]


def test_status_endpoint_returns_signed_ownership_proof_when_timestamp_header_is_present() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, validator_hotkey=_StubHotkey("5proof"))
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(
        "/validator/status",
        headers={
            "Authorization": 'Bittensor ss58="5demo",sig="00"',
            "X-Harnyx-Status-Ts": "2026-03-26T04:00:00+00:00",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["hotkey"] == "5proof"
    expected = "\n".join(
        (
            "validator-status-v1",
            "path=/validator/status",
            "request_ts=2026-03-26T04:00:00+00:00",
            "hotkey=5proof",
            "status=ok",
            "running=False",
        )
    ).encode("utf-8")
    assert payload["signature_hex"] == (b"sig:" + expected).hex()


def test_control_routes_return_503_when_auth_warmup_is_unavailable() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, auth=_auth_unavailable)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get("/validator/status")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "inbound auth verifier has not completed initial hotkey warmup",
    }


def _make_task_submission(*, batch_id: UUID) -> tuple[MinerTask, MinerTaskRunSubmission]:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    total_tool_usage = ToolUsageSummary(
        search_tool=SearchToolUsageSummary(call_count=2, cost=0.005),
        search_tool_cost=0.005,
        llm=LlmUsageSummary(
            call_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=0.02,
            providers={
                "chutes": {
                    "openai/gpt-oss-20b-TEE": LlmModelUsageCost(
                        usage=LlmUsageTotals(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                            call_count=1,
                        ),
                        cost=0.02,
                    )
                }
            },
        ),
        llm_cost=0.02,
    )
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        task_id=task.task_id,
        response=Response(text="The miner answer."),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=0.9,
                total_score=0.9,
                scoring_version="v1",
                reasoning=ScorerReasoning(
                    text="miner-first trace\n\n---\n\nreference-first trace",
                    reasoning_tokens=18,
                ),
            ),
            total_tool_usage=total_tool_usage,
            elapsed_ms=2500.0,
        ),
        completed_at=datetime.now(UTC),
    )

    issued_at = datetime.now(UTC)
    session = Session(
        session_id=run.session_id,
        uid=run.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(total_cost_usd=0.025),
    )

    submission = MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=4,
        run=run,
        score=0.9,
        execution_log=_sample_execution_log(run.session_id, run.uid),
        usage=TokenUsageSummary.empty(),
        session=session,
    )
    return task, submission


def _make_submission_for_batch_pair(
    *,
    batch_id: UUID,
    task: MinerTask,
    artifact: ScriptArtifactSpec,
) -> MinerTaskRunSubmission:
    issued_at = datetime(2026, 3, 8, tzinfo=UTC)
    completed_at = issued_at + timedelta(seconds=5)
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        response=Response(text=f"answer for {task.query.text}"),
        details=EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                similarity_score=0.8,
                total_score=0.9,
                scoring_version="v1",
            ),
            total_tool_usage=ToolUsageSummary.zero(),
            elapsed_ms=2500.0,
        ),
        completed_at=completed_at,
    )
    session = Session(
        session_id=run.session_id,
        uid=run.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(total_cost_usd=0.0),
    )
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=4,
        run=run,
        score=0.9,
        execution_log=(),
        usage=TokenUsageSummary.empty(),
        session=session,
    )


def _make_attempt_for_batch_pair(
    *,
    batch_id: UUID,
    task: MinerTask,
    artifact: ScriptArtifactSpec,
) -> MinerTaskAttemptAuditRecord:
    session_id = uuid4()
    started_at = datetime(2026, 3, 8, 0, 0, 1, tzinfo=UTC)
    return MinerTaskAttemptAuditRecord(
        validator_session_id=session_id,
        batch_id=batch_id,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        attempt_number=1,
        uid=artifact.uid,
        miner_hotkey_ss58="miner-hotkey",
        started_at=started_at,
        finished_at=started_at + timedelta(seconds=3),
        status=MinerTaskAttemptStatus.FAILED,
        error_code="tool_execution_timeout",
        error_summary_code="timeout_miner_owned",
        retry_decision=MinerTaskAttemptRetryDecision.WILL_RETRY,
        terminal_effect=MinerTaskAttemptTerminalEffect.NONE,
        max_attempts=2,
        execution_log=_sample_execution_log(session_id, artifact.uid),
    )


def _make_failed_task_submission(*, batch_id: UUID, error_code: str) -> tuple[MinerTask, MinerTaskRunSubmission]:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    run = MinerTaskRun(
        session_id=uuid4(),
        uid=7,
        artifact_id=uuid4(),
        task_id=task.task_id,
        response=None,
        details=EvaluationDetails(
            error=EvaluationError(code=error_code, message="terminal timeout"),
            total_tool_usage=ToolUsageSummary.zero(),
            elapsed_ms=2500.0,
        ),
        completed_at=datetime.now(UTC),
    )
    issued_at = datetime.now(UTC)
    session = Session(
        session_id=run.session_id,
        uid=run.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=0.1,
        usage=SessionUsage(total_cost_usd=0.0),
    )
    submission = MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=4,
        run=run,
        score=0.0,
        execution_log=_sample_execution_log(run.session_id, run.uid),
        usage=TokenUsageSummary.empty(),
        session=session,
    )
    return task, submission


def _sample_execution_log(session_id: UUID, uid: int) -> tuple[ToolCall, ...]:
    started_at = datetime(2026, 3, 8, 0, 0, 1, tzinfo=UTC)
    finished_at = datetime(2026, 3, 8, 0, 0, 3, tzinfo=UTC)
    return (
        ToolCall(
            receipt_id="receipt-1",
            session_id=session_id,
            uid=uid,
            tool="search_web",
            issued_at=started_at,
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash="request-hash",
                request_payload={"args": [], "kwargs": {"query": "what happened?"}},
                response_hash="response-hash",
                response_payload={"results": [{"title": "Example"}]},
                cost_usd=0.001,
                execution=ToolExecutionFacts(
                    elapsed_ms=2000.0,
                    started_at=started_at,
                    finished_at=finished_at,
                ),
            ),
        ),
    )


def _make_batch_payload(
    *,
    batch_id: UUID,
    task_id: UUID | None = None,
    artifact_id: UUID | None = None,
    query_text: str = "What happened?",
) -> dict[str, object]:
    return {
        "batch_id": str(batch_id),
        "cutoff_at": "2026-03-08T00:00:00+00:00",
        "created_at": "2026-03-08T00:00:00+00:00",
        "tasks": [
            {
                "task_id": str(task_id or uuid4()),
                "query": {"text": query_text},
                "reference_answer": {"text": "The reference answer."},
                "budget_usd": 0.05,
            }
        ],
        "artifacts": [
            {
                "uid": 7,
                "artifact_id": str(artifact_id or uuid4()),
                "miner_hotkey_ss58": "miner-hotkey",
                "task_retry_count": 0,
                "content_hash": "hash-123",
                "size_bytes": 42,
            }
        ],
    }


def _make_restore_run_payload(submission: MinerTaskRunSubmission) -> dict[str, object]:
    total_tool_usage = submission.run.details.total_tool_usage
    llm_usage = total_tool_usage.llm
    score_breakdown = submission.run.details.score_breakdown
    error = submission.run.details.error
    return {
        "batch_id": str(submission.batch_id),
        "validator": {"uid": submission.validator_uid},
        "run": {
            "artifact_id": str(submission.run.artifact_id),
            "task_id": str(submission.run.task_id),
            "completed_at": submission.run.completed_at.isoformat(),
            "response": None if submission.run.response is None else {"text": submission.run.response.text},
        },
        "score": submission.score,
        "execution_log": TypeAdapter(tuple[ToolCall, ...]).dump_python(
            submission.execution_log,
            mode="json",
        ),
        "usage": {
            "total_prompt_tokens": submission.usage.total_prompt_tokens,
            "total_completion_tokens": submission.usage.total_completion_tokens,
            "total_tokens": submission.usage.total_tokens,
            "call_count": submission.usage.call_count,
            "by_provider": submission.usage.by_provider,
        },
        "session": {
            "session_id": str(submission.session.session_id),
            "uid": submission.session.uid,
            "status": submission.session.status.value,
            "issued_at": submission.session.issued_at.isoformat(),
            "expires_at": submission.session.expires_at.isoformat(),
        },
        "specifics": {
            "score_breakdown": (
                None
                if score_breakdown is None
                else {
                    "comparison_score": score_breakdown.comparison_score,
                    "total_score": score_breakdown.total_score,
                    "scoring_version": score_breakdown.scoring_version,
                    "reasoning": (
                        None
                        if score_breakdown.reasoning is None
                        else {
                            "text": score_breakdown.reasoning.text,
                            "reasoning_tokens": score_breakdown.reasoning.reasoning_tokens,
                        }
                    ),
                }
            ),
            "error": None if error is None else {"code": error.code, "message": error.message},
            "total_tool_usage": {
                "search_tool": {
                    "call_count": total_tool_usage.search_tool.call_count,
                    "cost": total_tool_usage.search_tool.cost,
                },
                "search_tool_cost": total_tool_usage.search_tool_cost,
                "llm": {
                    "call_count": llm_usage.call_count,
                    "prompt_tokens": llm_usage.prompt_tokens,
                    "completion_tokens": llm_usage.completion_tokens,
                    "total_tokens": llm_usage.total_tokens,
                    "cost": llm_usage.cost,
                    "providers": {
                        provider: {
                            model: {
                                "usage": {
                                    "prompt_tokens": model_cost.usage.prompt_tokens,
                                    "completion_tokens": model_cost.usage.completion_tokens,
                                    "total_tokens": model_cost.usage.total_tokens,
                                    "call_count": model_cost.usage.call_count,
                                },
                                "cost": model_cost.cost,
                            }
                            for model, model_cost in models.items()
                        }
                        for provider, models in llm_usage.providers.items()
                    },
                },
                "llm_cost": total_tool_usage.llm_cost,
            },
            "elapsed_ms": submission.run.details.elapsed_ms,
        },
    }


def test_status_and_runs_endpoints_split_summary_from_detail_page() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 1,
        "completed": 1,
        "remaining": 0,
        "tasks": (task,),
        "run_page_submissions": (submission,),
        "provider_evidence": (
            {
                "provider": "desearch",
                "model": "search_web",
                "total_calls": 12,
                "failed_calls": 5,
                "failure_reason": "provider timed out",
            },
        ),
    }

    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["status"] == "processing"
    assert body["total"] == 1
    assert body["completed"] == 1
    assert body["remaining"] == 0
    assert body["latest_sequence"] == 1
    assert "run_page_submissions" not in body
    assert body["provider_model_evidence"] == [
        {
            "provider": "desearch",
            "model": "search_web",
            "total_calls": 12,
            "failed_calls": 5,
            "failure_reason": "provider timed out",
        }
    ]

    runs_response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")
    assert runs_response.status_code == 200
    runs_body = runs_response.json()
    assert runs_body["batch_id"] == str(batch_id)
    assert runs_body["after_sequence"] == 0
    assert runs_body["latest_sequence"] == 1
    assert runs_body["next_after_sequence"] == 1
    assert runs_body["has_more"] is False
    item = runs_body["items"][0]
    assert item["sequence"] == 1
    run = item["submission"]
    assert run["score"] == pytest.approx(0.9)
    assert "query" not in run["run"]
    assert "reference_answer" not in run["run"]
    assert run["run"]["response"]["text"] == "The miner answer."
    assert run["run"]["response"].get("citations") is None

    specifics = run["specifics"]
    assert specifics["total_tool_usage"]["search_tool_cost"] == pytest.approx(0.005)
    assert specifics["total_tool_usage"]["llm_cost"] == pytest.approx(0.02)
    assert specifics["score_breakdown"]["total_score"] == pytest.approx(0.9)
    assert specifics["score_breakdown"]["reasoning"] == {
        "text": "miner-first trace\n\n---\n\nreference-first trace",
        "reasoning_tokens": 18,
    }
    assert specifics["elapsed_ms"] == pytest.approx(2500.0)
    execution_log = run["execution_log"]
    assert len(execution_log) == 1
    assert execution_log[0]["receipt_id"] == "receipt-1"
    assert execution_log[0]["session_id"] == str(submission.run.session_id)
    assert execution_log[0]["tool"] == "search_web"
    assert execution_log[0]["issued_at"] == "2026-03-08T00:00:01Z"
    assert execution_log[0]["details"]["request_payload"] == {
        "args": [],
        "kwargs": {"query": "what happened?"},
    }
    assert execution_log[0]["details"]["response_payload"] == {"results": [{"title": "Example"}]}
    assert execution_log[0]["details"]["cost_usd"] == pytest.approx(0.001)
    assert execution_log[0]["details"]["execution"] == {
        "elapsed_ms": pytest.approx(2000.0),
        "ttft_ms": None,
        "started_at": "2026-03-08T00:00:01Z",
        "finished_at": "2026-03-08T00:00:03Z",
    }


def test_failed_status_keeps_failure_detail_and_runs_page_keeps_sequence() -> None:
    batch_id = uuid4()
    first_task, first_submission = _make_task_submission(batch_id=batch_id)
    second_task, second_submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 2,
        "completed": 2,
        "remaining": 0,
        "tasks": (first_task, second_task),
        "run_page_submissions": (first_submission, second_submission),
        "provider_evidence": (),
    }

    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        lifecycle="failed",
        error_code="sandbox_invocation_failed",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="sandbox_invocation_failed",
            error_message="plain sandbox failure",
            occurred_at=datetime(2026, 3, 26, 21, 0, tzinfo=UTC),
            artifact_id=first_submission.run.artifact_id,
            task_id=first_submission.run.task_id,
            uid=first_submission.run.uid,
            exception_type="SandboxInvocationError",
            traceback="Traceback (most recent call last): ...",
        ),
    )
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["status"] == "failed"
    assert body["error_code"] == "sandbox_invocation_failed"
    assert body["failure_detail"] == {
        "error_code": "sandbox_invocation_failed",
        "error_message": "plain sandbox failure",
        "artifact_id": str(first_submission.run.artifact_id),
        "task_id": str(first_submission.run.task_id),
        "uid": first_submission.run.uid,
        "exception_type": "SandboxInvocationError",
        "traceback": "Traceback (most recent call last): ...",
        "occurred_at": "2026-03-26T21:00:00+00:00",
    }
    assert body["total"] == 2
    assert body["completed"] == 2
    assert body["remaining"] == 0
    runs_response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")
    assert runs_response.status_code == 200
    runs_body = runs_response.json()
    assert [item["submission"]["run"]["task_id"] for item in runs_body["items"]] == [
        str(first_task.task_id),
        str(second_task.task_id),
    ]
    assert [item["sequence"] for item in runs_body["items"]] == [1, 2]
    assert all("query" not in item["submission"]["run"] for item in runs_body["items"])


def test_runs_endpoint_preserves_real_record_sequence_without_artifact_sorting(tmp_path: Path) -> None:
    provider = RealAcceptBatchDependencyProvider(storage_root=tmp_path / "run-progress")
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    first_artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="hash-a", size_bytes=42)
    second_artifact = ScriptArtifactSpec(uid=8, artifact_id=uuid4(), content_hash="hash-b", size_bytes=43)
    batch = MinerTaskBatchSpec(
        batch_id=batch_id,
        cutoff_at="2026-03-08T00:00:00+00:00",
        created_at="2026-03-08T00:00:00+00:00",
        tasks=(task,),
        artifacts=(first_artifact, second_artifact),
    )
    first_submission = _make_submission_for_batch_pair(
        batch_id=batch_id,
        task=task,
        artifact=first_artifact,
    )
    second_submission = _make_submission_for_batch_pair(
        batch_id=batch_id,
        task=task,
        artifact=second_artifact,
    )

    provider.accept_batch.execute(batch)
    provider.progress_tracker.record(second_submission)
    provider.progress_tracker.record(first_submission)
    provider.accept_batch.mark_failed(
        batch_id,
        error_code="sandbox_invocation_failed",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="sandbox_invocation_failed",
            error_message="plain sandbox failure",
            occurred_at=datetime(2026, 3, 26, 21, 0, tzinfo=UTC),
            artifact_id=second_artifact.artifact_id,
            task_id=task.task_id,
            uid=second_artifact.uid,
            exception_type="SandboxInvocationError",
        ),
    )

    response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["latest_sequence"] == 2
    assert [item["submission"]["run"]["artifact_id"] for item in body["items"]] == [
        str(second_artifact.artifact_id),
        str(first_artifact.artifact_id),
    ]


def test_runs_endpoint_hydrates_file_backed_terminated_attempt_execution_log(
    tmp_path: Path,
) -> None:
    provider = RealAcceptBatchDependencyProvider(storage_root=tmp_path / "run-progress")
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="hash-a", size_bytes=42)
    batch = MinerTaskBatchSpec(
        batch_id=batch_id,
        cutoff_at="2026-03-08T00:00:00+00:00",
        created_at="2026-03-08T00:00:00+00:00",
        tasks=(task,),
        artifacts=(artifact,),
    )
    attempt = _make_attempt_for_batch_pair(batch_id=batch_id, task=task, artifact=artifact)
    submission = _make_submission_for_batch_pair(
        batch_id=batch_id,
        task=task,
        artifact=artifact,
    )

    provider.accept_batch.execute(batch)
    provider.progress_tracker.record_terminated_attempt(attempt)
    provider.progress_tracker.record(submission)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["latest_sequence"] == 2
    first, second = body["items"]
    assert first["sequence"] == 1
    assert first["kind"] == "terminated_attempt"
    assert first["submission"] is None
    assert first["attempt"]["attempt_number"] == 1
    assert first["attempt"]["error_summary_code"] == "timeout_miner_owned"
    assert first["attempt"]["execution_log"][0]["receipt_id"] == "receipt-1"
    assert first["attempt"]["execution_log"][0]["details"]["response_payload"] == {
        "results": [{"title": "Example"}]
    }
    assert second["sequence"] == 2
    assert second["kind"] == "completed_run"
    assert second["attempt"] is None
    assert second["submission"]["run"]["artifact_id"] == str(artifact.artifact_id)


def test_runs_endpoint_returns_empty_page_after_terminal_progress_cleanup(tmp_path: Path) -> None:
    provider = RealAcceptBatchDependencyProvider(storage_root=tmp_path / "run-progress")
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="What happened?"),
        reference_answer=ReferenceAnswer(text="The reference answer."),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="hash-a", size_bytes=42)
    batch = MinerTaskBatchSpec(
        batch_id=batch_id,
        cutoff_at="2026-03-08T00:00:00+00:00",
        created_at="2026-03-08T00:00:00+00:00",
        tasks=(task,),
        artifacts=(artifact,),
    )
    submission = _make_submission_for_batch_pair(
        batch_id=batch_id,
        task=task,
        artifact=artifact,
    )
    terminal_at = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)

    provider.accept_batch.execute(batch)
    assert provider.inbox.next() == batch
    provider.accept_batch.mark_processing(batch_id)
    provider.progress_tracker.record(submission)
    provider.accept_batch.mark_completed(batch_id, terminal_at=terminal_at)
    assert provider.accept_batch.prune_terminal_batch(
        batch_id,
        older_than=terminal_at,
        cleanup=provider.progress_tracker.discard_batch,
    )

    response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "batch_id": str(batch_id),
        "after_sequence": 0,
        "limit": 10,
        "latest_sequence": 0,
        "next_after_sequence": 0,
        "has_more": False,
        "items": [],
        "failure_detail": None,
    }


def test_runs_endpoint_openapi_declares_page_bounds() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)

    operation = app.openapi()["paths"]["/validator/miner-task-batches/{batch_id}/runs"]["get"]
    parameters = {parameter["name"]: parameter for parameter in operation["parameters"]}

    assert parameters["after_sequence"]["schema"]["minimum"] == 0
    assert parameters["limit"]["schema"]["minimum"] == 1
    assert parameters["limit"]["schema"]["maximum"] == 500


def test_runs_endpoint_includes_timeout_inconclusive_pair_result_when_batch_fails() -> None:
    batch_id = uuid4()
    task, submission = _make_failed_task_submission(
        batch_id=batch_id,
        error_code="timeout_inconclusive",
    )
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 1,
        "completed": 1,
        "remaining": 0,
        "tasks": (task,),
        "run_page_submissions": (submission,),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        lifecycle="failed",
        error_code="timeout_inconclusive",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="timeout_inconclusive",
            error_message="terminal timeout",
            occurred_at=datetime(2026, 3, 26, 21, 0, tzinfo=UTC),
            artifact_id=submission.run.artifact_id,
            task_id=submission.run.task_id,
            uid=submission.run.uid,
            exception_type="ReadTimeout",
        ),
    )
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["failure_detail"]["error_code"] == "timeout_inconclusive"
    runs_response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")
    assert runs_response.status_code == 200
    assert runs_response.json()["items"][0]["submission"]["specifics"]["error"]["code"] == "timeout_inconclusive"


def test_status_endpoint_replaces_blank_failure_message_at_transport_boundary() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 1,
        "completed": 1,
        "remaining": 0,
        "tasks": (task,),
        "run_page_submissions": (submission,),
        "provider_evidence": (),
    }

    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        lifecycle="failed",
        error_code="unexpected_validator_failure",
        failure_detail=ValidatorBatchFailureDetail(
            error_code="unexpected_validator_failure",
            error_message="",
            occurred_at=datetime(2026, 3, 31, 9, 26, tzinfo=UTC),
            artifact_id=submission.run.artifact_id,
            task_id=submission.run.task_id,
            uid=submission.run.uid,
            exception_type="ReadTimeout",
            traceback=None,
        ),
    )
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["status"] == "failed"
    assert body["error_code"] == "unexpected_validator_failure"
    assert body["failure_detail"] == {
        "error_code": "unexpected_validator_failure",
        "error_message": "ReadTimeout",
        "artifact_id": str(submission.run.artifact_id),
        "task_id": str(submission.run.task_id),
        "uid": submission.run.uid,
        "exception_type": "ReadTimeout",
        "traceback": None,
        "occurred_at": "2026-03-31T09:26:00+00:00",
    }
    assert body["total"] == 1
    assert body["completed"] == 1
    assert body["remaining"] == 0


def test_runs_endpoint_converts_page_failure_to_valid_failed_payload() -> None:
    batch_id = uuid4()
    _task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 2,
        "completed": 1,
        "remaining": 1,
        "tasks": (),
        "run_page_submissions": (submission,),
        "provider_evidence": (),
        "page_error": RuntimeError("page exploded"),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, lifecycle="processing")
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["failure_detail"]["error_code"] == "progress_snapshot_failed"
    assert body["failure_detail"]["exception_type"] == "RuntimeError"
    assert "page exploded" in body["failure_detail"]["error_message"]
    assert "RuntimeError: page exploded" in body["failure_detail"]["traceback"]
    assert body["items"] == []


def test_runs_endpoint_converts_restore_floor_stale_cursor_to_failed_payload() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 2,
        "completed": 1,
        "remaining": 1,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
        "page_error": ProgressCursorBeforeRestoreFloorError(
            "progress cursor is older than restored platform detail floor"
        ),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, lifecycle="processing")
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/runs?after_sequence=0&limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["batch_id"] == str(batch_id)
    assert body["failure_detail"]["error_code"] == "progress_snapshot_failed"
    assert body["failure_detail"]["exception_type"] == "ProgressCursorBeforeRestoreFloorError"
    assert "restored platform detail floor" in body["failure_detail"]["error_message"]
    assert body["items"] == []


def test_signed_validator_route_internal_error_returns_structured_payload_and_captures_sentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot: RunProgressFixture = {
        "batch_id": uuid4(),
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        status_provider=_ExplodingStatusProvider(),
    )
    captured: list[Exception] = []
    monkeypatch.setattr(routes_mod, "capture_exception", captured.append)
    client = TestClient(_create_test_app(provider), raise_server_exceptions=False)

    response = client.get(
        "/validator/status",
        headers={
            "Authorization": 'Bittensor ss58="5demo",sig="00"',
            "x-request-id": "req-123",
        },
    )

    assert response.status_code == 500
    assert response.json()["error_code"] == "internal_server_error"
    assert response.json()["exception_type"] == "RuntimeError"
    assert response.json()["request_id"] == "req-123"
    assert response.json()["error_message"] == "status exploded"
    assert "RuntimeError: status exploded" in response.json()["traceback"]
    assert len(captured) == 1
    assert str(captured[0]) == "status exploded"


def test_status_route_tolerates_resource_usage_sampling_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot: RunProgressFixture = {
        "batch_id": uuid4(),
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        resource_usage_provider=_ExplodingResourceUsageProvider(),
    )
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        routes_mod.logger,
        "exception",
        lambda *args, **kwargs: captured.append((args, kwargs)),
    )
    client = TestClient(_create_test_app(provider), raise_server_exceptions=False)

    response = client.get(
        "/validator/status",
        headers={"Authorization": 'Bittensor ss58="5demo",sig="00"'},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["hotkey"] == "5validator"
    assert response.json()["resource_usage"] is None
    assert len(captured) == 1


def test_validator_route_internal_error_before_auth_success_falls_back_to_generic_500() -> None:
    snapshot: RunProgressFixture = {
        "batch_id": uuid4(),
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(
        snapshot=snapshot,
        auth=_explode_auth,
        status_provider=_ExplodingStatusProvider(),
    )
    client = TestClient(_create_test_app(provider), raise_server_exceptions=False)

    response = client.get(
        "/validator/status",
        headers={"x-request-id": "req-auth-pre"},
    )

    assert response.status_code == 500
    assert "auth exploded" not in response.text


def test_accept_batch_endpoint_accepts_platform_json_payload() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at": "2026-03-08T00:00:00+00:00",
            "created_at": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": 7,
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                    "miner_hotkey_ss58": "miner-hotkey",
                    "task_retry_count": 0,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "batch_id": str(batch_id),
        "caller": "caller",
    }
    received = provider.accept_batch.received_batch
    assert received is not None
    assert received.batch_id == batch_id
    assert received.tasks[0].task_id == task_id
    assert received.tasks[0].query.text == "What happened?"
    assert received.artifacts[0].artifact_id == artifact_id
    assert received.cutoff_at == "2026-03-08T00:00:00+00:00"
    assert received.created_at == "2026-03-08T00:00:00+00:00"
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_rejects_restore_runs() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [_make_restore_run_payload(submission)],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_rejects_restore_runs_before_execution_log_parsing() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    restore_run = _make_restore_run_payload(submission)
    execution_log = list(restore_run["execution_log"])
    future_entry = dict(execution_log[0])
    future_entry["tool"] = "future_tool"
    restore_run["execution_log"] = [future_entry, *execution_log]

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [restore_run],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_rejects_restore_run_and_provider_evidence() -> None:
    batch_id = uuid4()
    task, submission = _make_failed_task_submission(
        batch_id=batch_id,
        error_code="timeout_inconclusive",
    )
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [_make_restore_run_payload(submission)],
            "restore_provider_evidence": [
                {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "total_calls": 3,
                    "failed_calls": 1,
                    "failure_reason": "rate limited",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()
    assert provider.accept_batch.received_restore_provider_evidence == ()


def test_accept_batch_endpoint_rejects_invalid_restore_session_status() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)
    restore_payload = _make_restore_run_payload(submission)
    session_payload = restore_payload["session"]
    if not isinstance(session_payload, dict):
        raise AssertionError("expected restore session payload dict")
    session_payload["status"] = "not_a_real_status"

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [restore_payload],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_rejects_restore_run_with_divergent_score_breakdown() -> None:
    batch_id = uuid4()
    task, submission = _make_task_submission(batch_id=batch_id)
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)
    restore_payload = _make_restore_run_payload(submission)
    specifics_payload = restore_payload["specifics"]
    if not isinstance(specifics_payload, dict):
        raise AssertionError("expected restore specifics payload dict")
    score_breakdown = specifics_payload["score_breakdown"]
    if not isinstance(score_breakdown, dict):
        raise AssertionError("expected restore score breakdown payload dict")
    score_breakdown["comparison_score"] = 0.8
    score_breakdown["total_score"] = 0.9

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            **_make_batch_payload(
                batch_id=batch_id,
                task_id=task.task_id,
                artifact_id=submission.run.artifact_id,
                query_text=task.query.text,
            ),
            "restore_runs": [restore_payload],
        },
    )

    assert response.status_code == 422
    assert "comparison_score" in response.text
    assert provider.accept_batch.received_batch is None
    assert provider.accept_batch.received_restore_runs == ()


def test_accept_batch_endpoint_rejects_legacy_iso_keys() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at_iso": "2026-03-08T00:00:00+00:00",
            "created_at_iso": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": 7,
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                    "miner_hotkey_ss58": "miner-hotkey",
                    "task_retry_count": 0,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None


def test_accept_batch_endpoint_rejects_non_strict_artifact_uid_payload() -> None:
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.post(
        "/validator/miner-task-batches/batch",
        json={
            "batch_id": str(batch_id),
            "cutoff_at": "2026-03-08T00:00:00+00:00",
            "created_at": "2026-03-08T00:00:00+00:00",
            "tasks": [
                {
                    "task_id": str(task_id),
                    "query": {"text": "What happened?"},
                    "reference_answer": {"text": "The reference answer."},
                    "budget_usd": 0.05,
                }
            ],
            "artifacts": [
                {
                    "uid": "7",
                    "artifact_id": str(artifact_id),
                    "content_hash": "hash-123",
                    "size_bytes": 42,
                    "miner_hotkey_ss58": "miner-hotkey",
                    "task_retry_count": 0,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert provider.accept_batch.received_batch is None


def test_accept_batch_endpoint_is_idempotent_for_exact_duplicate_replay(tmp_path: Path) -> None:
    provider = RealAcceptBatchDependencyProvider(storage_root=tmp_path / "run-progress")
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    payload = _make_batch_payload(batch_id=batch_id, task_id=task_id, artifact_id=artifact_id)

    first = client.post("/validator/miner-task-batches/batch", json=payload)
    second = client.post("/validator/miner-task-batches/batch", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert len(provider.inbox) == 0
    assert provider.status_provider.state.queued_batches == 0
    assert len(provider.restore_worker.received_decisions) == 1
    assert provider.restore_worker.received_decisions[0].should_start_restore is True
    summary = provider.progress_tracker.summary(batch_id)
    assert summary["total"] == 1
    assert summary["completed"] == 0
    assert summary["remaining"] == 1
    assert summary["latest_sequence"] == 0


def test_status_endpoint_returns_unknown_for_unaccepted_batch() -> None:
    batch_id = uuid4()
    snapshot: RunProgressFixture = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "tasks": (),
        "run_page_submissions": (),
        "provider_evidence": (),
    }
    provider = DemoControlDependencyProvider(snapshot=snapshot, lifecycle=None)
    app = _create_test_app(provider)
    client = TestClient(app)

    response = client.get(f"/validator/miner-task-batches/{batch_id}/status")

    assert response.status_code == 200
    assert response.json() == {
        "batch_id": str(batch_id),
        "status": "unknown",
        "error_code": None,
        "failure_detail": None,
        "total": 0,
        "completed": 0,
        "remaining": 0,
        "latest_sequence": 0,
        "provider_model_evidence": [],
        "activity_last_updated_at": None,
        "activity_stage": None,
        "active_artifact_count": 0,
        "active_task_session_count": 0,
    }


def test_accept_batch_endpoint_rejects_conflicting_duplicate_replay(tmp_path: Path) -> None:
    provider = RealAcceptBatchDependencyProvider(storage_root=tmp_path / "run-progress")
    app = _create_test_app(provider)
    client = TestClient(app)
    batch_id = uuid4()
    task_id = uuid4()
    artifact_id = uuid4()
    payload = _make_batch_payload(batch_id=batch_id, task_id=task_id, artifact_id=artifact_id)
    conflicting = _make_batch_payload(
        batch_id=batch_id,
        task_id=task_id,
        artifact_id=artifact_id,
        query_text="Different question?",
    )

    first = client.post("/validator/miner-task-batches/batch", json=payload)
    second = client.post("/validator/miner-task-batches/batch", json=conflicting)

    assert first.status_code == 200
    assert second.status_code == 400
    assert second.json() == {"detail": "batch_id already exists with different contents"}
    assert len(provider.inbox) == 0
    assert provider.status_provider.state.queued_batches == 0
