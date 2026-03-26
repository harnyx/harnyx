from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import Session, SessionFailureCode, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import ReceiptMetadata, SearchToolResult, ToolCall, ToolCallOutcome
from harnyx_commons.domain.tool_usage import SearchToolUsageSummary, ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, UsageSummarizer
from harnyx_validator.application.invoke_entrypoint import SandboxInvocationError
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    EvaluationRunner,
    ValidatorBatchFailedError,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


class _ClockSequence:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def __call__(self) -> datetime:
        if not self._values:
            raise AssertionError("clock sequence exhausted")
        return self._values.pop(0)


class _RecordingEvaluationStore:
    def __init__(self) -> None:
        self.records: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records.append(result)


def _record_receipt(
    receipt_log: FakeReceiptLog,
    *,
    session_id,
    uid: int,
    receipt_id: str,
    issued_at: datetime,
    cost_usd: float,
) -> None:
    receipt_log.record(
        ToolCall(
            receipt_id=receipt_id,
            session_id=session_id,
            uid=uid,
            tool="search_web",
            issued_at=issued_at,
            outcome=ToolCallOutcome.OK,
            metadata=ReceiptMetadata(
                request_hash=f"{receipt_id}-req",
                response_hash=f"{receipt_id}-res",
                cost_usd=cost_usd,
            ),
        )
    )


def _search_usage(receipt_log: FakeReceiptLog, session_id) -> ToolUsageSummary:
    receipts = tuple(receipt_log.for_session(session_id))
    total_cost = sum(float(receipt.metadata.cost_usd or 0.0) for receipt in receipts)
    return ToolUsageSummary(
        search_tool=SearchToolUsageSummary(
            call_count=len(receipts),
            cost=round(total_cost, 6),
        ),
        search_tool_cost=total_cost,
        llm=ToolUsageSummary.zero().llm,
        llm_cost=0.0,
    )


def _successful_outcome(
    request: MinerTaskRunRequest,
    *,
    score: float = 0.75,
) -> TaskRunOutcome:
    return TaskRunOutcome(
        run=MinerTaskRun(
            session_id=request.session_id,
            uid=request.uid,
            artifact_id=request.artifact_id,
            task_id=request.task.task_id,
            response=Response(text=f"answer {request.task.query.text}"),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=score,
                    similarity_score=score,
                    total_score=score,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ),
        usage=TokenUsageSummary.empty(),
    )


def test_usage_summarizer_falls_back_to_referenceable_result_count_when_search_cost_is_missing() -> None:
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(),
    )
    receipt = ToolCall(
        receipt_id="receipt-fallback",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        metadata=ReceiptMetadata(
            request_hash="req",
            response_hash="res",
            cost_usd=None,
            results=(
                SearchToolResult(index=0, result_id="result-1", url="https://a.example", note="A"),
                SearchToolResult(index=1, result_id="result-2", url="https://b.example", note="B"),
            ),
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))

    assert total_tool_usage.search_tool.call_count == 1
    assert total_tool_usage.search_tool.cost == pytest.approx(0.0002)
    assert total_tool_usage.search_tool_cost == pytest.approx(0.0002)


def _sandbox_invocation_error(
    message: str,
    *,
    status_code: int = 0,
    detail_code: str | None = None,
    detail_exception: str = "RuntimeError",
    detail_error: str | None = None,
) -> SandboxInvocationError:
    return SandboxInvocationError(
        message,
        status_code=status_code,
        detail_code=detail_code,
        detail_exception=detail_exception,
        detail_error=detail_error or message,
    )


class _ExhaustingOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log

    async def evaluate(self, request: MinerTaskRunRequest) -> None:
        session = self._sessions.get(request.session_id)
        assert session is not None
        self._sessions.update(session.mark_exhausted())
        self._receipt_log.record(
            ToolCall(
                receipt_id="receipt-1",
                session_id=request.session_id,
                uid=request.uid,
                tool="search_web",
                issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
                outcome=ToolCallOutcome.OK,
                metadata=ReceiptMetadata(
                    request_hash="req",
                    response_hash="res",
                    cost_usd=0.25,
                ),
            )
        )
        raise SessionBudgetExhaustedError("session exhausted during entrypoint invocation")


class _RetryThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        assert session.status is SessionStatus.ACTIVE
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"receipt-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
        )
        if self.calls == 1:
            raise _sandbox_invocation_error("transient sandbox failure")
        details = EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                similarity_score=0.5,
                total_score=0.75,
                scoring_version="v1",
            ),
            total_tool_usage=_search_usage(self._receipt_log, request.session_id),
        )
        self._receipt_log.clear_session(request.session_id)
        return TaskRunOutcome(
            run=MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text="answer"),
                details=details,
                completed_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
            ),
            usage=TokenUsageSummary.empty(),
        )


class _GenericFailureOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"generic-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            cost_usd=0.25,
        )
        raise RuntimeError("scoring failed")


class _RetryThenExhaustedOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        if self.calls == 1:
            _record_receipt(
                self._receipt_log,
                session_id=request.session_id,
                uid=request.uid,
                receipt_id="near-limit",
                issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
                cost_usd=0.04,
            )
            raise _sandbox_invocation_error("transient sandbox failure")
        self._sessions.update(session.mark_exhausted())
        raise SessionBudgetExhaustedError("session exhausted during retry")


class _ProviderMarkerThenSandboxFailureOrchestrator:
    def __init__(self, *, sessions: FakeSessionRegistry) -> None:
        self._sessions = sessions
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        if self.calls == 1:
            self._sessions.update(session.mark_failure_code(SessionFailureCode.TOOL_PROVIDER_FAILED))
            raise _sandbox_invocation_error("tool route failed")
        raise _sandbox_invocation_error("plain sandbox failure")


class _ProviderMarkerThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"provider-success-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
        )
        if self.calls == 1:
            self._sessions.update(session.mark_failure_code(SessionFailureCode.TOOL_PROVIDER_FAILED))
        details = EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                similarity_score=0.5,
                total_score=0.75,
                scoring_version="v1",
            ),
            total_tool_usage=_search_usage(self._receipt_log, request.session_id),
        )
        self._receipt_log.clear_session(request.session_id)
        return TaskRunOutcome(
            run=MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text="answer"),
                details=details,
                completed_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
            ),
            usage=TokenUsageSummary.empty(),
        )


class _RepeatedProviderMarkerOrchestrator:
    def __init__(self, *, sessions: FakeSessionRegistry, receipt_log: FakeReceiptLog) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"provider-repeat-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
        )
        self._sessions.update(session.mark_failure_code(SessionFailureCode.TOOL_PROVIDER_FAILED))
        details = EvaluationDetails(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                similarity_score=0.5,
                total_score=0.75,
                scoring_version="v1",
            ),
            total_tool_usage=_search_usage(self._receipt_log, request.session_id),
        )
        self._receipt_log.clear_session(request.session_id)
        return TaskRunOutcome(
            run=MinerTaskRun(
                session_id=request.session_id,
                uid=request.uid,
                artifact_id=request.artifact_id,
                task_id=request.task.task_id,
                response=Response(text="answer"),
                details=details,
                completed_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
            ),
            usage=TokenUsageSummary.empty(),
        )


class _RepeatedProviderMarkerSandboxFailureOrchestrator:
    def __init__(self, *, sessions: FakeSessionRegistry) -> None:
        self._sessions = sessions
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        self._sessions.update(session.mark_failure_code(SessionFailureCode.TOOL_PROVIDER_FAILED))
        raise _sandbox_invocation_error("tool route failed")


class _UnhandledMinerCrashOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="KeyError",
            detail_error="missing key",
        )


async def test_evaluation_runner_records_exhausted_submission() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="budget test"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    submissions = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(
            TaskRunOrchestrator,
            _ExhaustingOrchestrator(
                sessions=session_registry,
                receipt_log=receipt_log,
            ),
        ),
    )

    assert len(submissions) == 1
    submission = submissions[0]
    assert submission.validator_uid == 41
    assert submission.score == 0.0
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.response is None
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "session_budget_exhausted"
    assert submission.run.details.error.message == "session exhausted during entrypoint invocation"
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.25)
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_retries_transient_invocation_with_same_session_and_accumulated_usage() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="retry test"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _RetryThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    submissions = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 1
    assert len(submissions) == 1
    submission = submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[0]
    assert submission.score == pytest.approx(0.75)
    assert submission.run.details.total_tool_usage.search_tool.call_count == 2
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.5)
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_fails_batch_on_generic_post_invoke_failure() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="generic failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _GenericFailureOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    with pytest.raises(ValidatorBatchFailedError, match="scoring failed") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


async def test_evaluation_runner_records_budget_exhausted_when_retry_starts_near_limit() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="near budget"),
        reference_answer=ReferenceAnswer(text="reference"),
        budget_usd=0.05,
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _RetryThenExhaustedOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    submissions = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 1
    assert len(submissions) == 1
    submission = submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[0]
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.details.error == EvaluationError(
        code="session_budget_exhausted",
        message="session exhausted during retry",
    )
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.04)
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_retries_when_success_path_consumes_provider_marker() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider marker success"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderMarkerThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    submissions = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )

    assert orchestrator.calls == 2
    assert len(submissions) == 1
    assert submissions[0].score == pytest.approx(0.75)
    assert evaluation_store.records == submissions


async def test_evaluation_runner_returns_tool_provider_failed_after_retry_exhaustion() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider marker repeat"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _RepeatedProviderMarkerOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    with pytest.raises(ValidatorBatchFailedError, match="tool provider failed") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    assert exc_info.value.error_code == "tool_provider_failed"
    assert exc_info.value.failure_detail.error_code == "tool_provider_failed"
    assert exc_info.value.failure_detail.artifact_id == artifact.artifact_id
    assert exc_info.value.failure_detail.task_id == task.task_id
    assert exc_info.value.failure_detail.uid == artifact.uid
    assert orchestrator.calls == 2
    assert evaluation_store.records == []


async def test_evaluation_runner_prefers_provider_marker_over_sandbox_error() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider marker sandbox failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _RepeatedProviderMarkerSandboxFailureOrchestrator(sessions=session_registry)

    with pytest.raises(ValidatorBatchFailedError, match="tool provider failed") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    assert exc_info.value.error_code == "tool_provider_failed"
    assert orchestrator.calls == 2
    assert evaluation_store.records == []


async def test_evaluation_runner_records_zero_score_for_unhandled_miner_exception() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        ),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="miner crash"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    submissions = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _UnhandledMinerCrashOrchestrator()),
    )

    assert len(submissions) == 1
    submission = submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_does_not_let_stale_provider_marker_poison_later_attempt() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider marker"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderMarkerThenSandboxFailureOrchestrator(sessions=session_registry)

    with pytest.raises(ValidatorBatchFailedError, match="plain sandbox failure") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    assert exc_info.value.error_code == "sandbox_invocation_failed"
    assert exc_info.value.failure_detail.error_code == "sandbox_invocation_failed"
    assert exc_info.value.failure_detail.error_message == "plain sandbox failure"
    assert exc_info.value.failure_detail.artifact_id == artifact.artifact_id
    assert exc_info.value.failure_detail.task_id == task.task_id
    assert exc_info.value.failure_detail.uid == artifact.uid
    assert exc_info.value.failure_detail.exception_type == "SandboxInvocationError"
    assert orchestrator.calls == 2
    assert evaluation_store.records == []


async def test_evaluation_runner_uses_bounded_continuous_worker_pool() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"task-{index}"),
            reference_answer=ReferenceAnswer(text=f"reference-{index}"),
        )
        for index in range(6)
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _ContinuousPoolOrchestrator:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.max_active = 0
            self._active = 0
            self.first_wave_started = asyncio.Event()
            self.sixth_started = asyncio.Event()
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started.append(text)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if len(self.started) == 5:
                self.first_wave_started.set()
            if text == "task-5":
                self.sixth_started.set()
            await self.release_by_text[text].wait()
            self._active -= 1
            return _successful_outcome(request, score=1.0)

    orchestrator = _ContinuousPoolOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.first_wave_started.wait(), timeout=1.0)
        assert set(orchestrator.started) == {f"task-{index}" for index in range(5)}
        assert "task-5" not in orchestrator.started

        orchestrator.release_by_text["task-0"].set()
        await asyncio.wait_for(orchestrator.sixth_started.wait(), timeout=1.0)

        for task in tasks[1:]:
            orchestrator.release_by_text[task.query.text].set()

        submissions = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()

    assert orchestrator.max_active == 5
    assert [submission.run.task_id for submission in submissions] == [task.task_id for task in tasks]
    assert len(evaluation_store.records) == 6


async def test_evaluation_runner_keeps_miner_failures_local_and_preserves_input_order() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    tasks = (
        MinerTask(
            task_id=uuid4(),
            query=Query(text="task-one"),
            reference_answer=ReferenceAnswer(text="reference-one"),
        ),
        MinerTask(
            task_id=uuid4(),
            query=Query(text="task-two"),
            reference_answer=ReferenceAnswer(text="reference-two"),
        ),
        MinerTask(
            task_id=uuid4(),
            query=Query(text="task-three"),
            reference_answer=ReferenceAnswer(text="reference-three"),
        ),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _OutOfOrderMinerFailureOrchestrator:
        def __init__(self) -> None:
            self.entered: list[str] = []
            self.all_entered = asyncio.Event()
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.entered.append(text)
            if len(self.entered) == len(tasks):
                self.all_entered.set()
            await self.release_by_text[text].wait()
            if text == "task-two":
                raise _sandbox_invocation_error(
                    "miner crashed",
                    detail_code="UnhandledException",
                    detail_exception="RuntimeError",
                    detail_error="boom",
                )
            return _successful_outcome(request, score=1.0)

    orchestrator = _OutOfOrderMinerFailureOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.all_entered.wait(), timeout=1.0)
        orchestrator.release_by_text["task-three"].set()
        orchestrator.release_by_text["task-two"].set()
        orchestrator.release_by_text["task-one"].set()
        submissions = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()

    assert [submission.run.task_id for submission in submissions] == [task.task_id for task in tasks]
    assert [submission.score for submission in submissions] == [1.0, 0.0, 1.0]
    assert submissions[1].run.details.error == EvaluationError(
        code="miner_unhandled_exception",
        message="boom",
    )
    assert len(evaluation_store.records) == 3


async def test_evaluation_runner_stops_new_dispatch_after_shared_failure_and_drains_in_flight_work() -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
    )
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"task-{index}"),
            reference_answer=ReferenceAnswer(text=f"reference-{index}"),
        )
        for index in range(6)
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _SharedFailureOrchestrator:
        def __init__(self) -> None:
            self.started_distinct: set[str] = set()
            self.first_wave_started = asyncio.Event()
            self.failure_triggered = asyncio.Event()
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started_distinct.add(text)
            if len(self.started_distinct) == 5:
                self.first_wave_started.set()
            await self.release_by_text[text].wait()
            if text == "task-0":
                self.failure_triggered.set()
                raise LlmRetryExhaustedError("shared failure")
            return _successful_outcome(request, score=1.0)

    orchestrator = _SharedFailureOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.first_wave_started.wait(), timeout=1.0)
        orchestrator.release_by_text["task-0"].set()
        await asyncio.wait_for(orchestrator.failure_triggered.wait(), timeout=1.0)
        assert "task-5" not in orchestrator.started_distinct

        for task in tasks[1:5]:
            orchestrator.release_by_text[task.query.text].set()

        with pytest.raises(ValidatorBatchFailedError, match="shared failure") as exc_info:
            await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()

    assert exc_info.value.error_code == "scoring_llm_retry_exhausted"
    assert orchestrator.started_distinct == {f"task-{index}" for index in range(5)}
    assert [record.run.task_id for record in evaluation_store.records] == [task.task_id for task in tasks[1:5]]
