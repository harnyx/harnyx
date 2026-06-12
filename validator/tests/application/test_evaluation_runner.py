from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

import httpx
import pytest

import harnyx_validator.application.services.evaluation_runner as evaluation_runner_module
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionStatus, SessionUsage
from harnyx_commons.domain.tool_call import (
    SearchToolResult,
    ToolCall,
    ToolCallDetails,
    ToolCallOutcome,
    ToolExecutionFacts,
)
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
from harnyx_validator.application.invoke_entrypoint import (
    MinerResponseValidationError,
    SandboxInvocationError,
)
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import (
    TERMINAL_TIMEOUT_ERROR_MESSAGE,
    EvaluationRunner,
    UnexpectedArtifactExecutionError,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.domain.evaluation import MinerTaskRun
from harnyx_validator.infrastructure.scoring.vertex_embedding import VertexEmbeddingRetryExhaustedError
from harnyx_validator.infrastructure.state.run_progress import FileBackedRunProgress
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry
from validator.tests.fixtures.subtensor import FakeSubtensorClient

pytestmark = pytest.mark.anyio("asyncio")


def _progress(tmp_path: Path) -> FileBackedRunProgress:
    return FileBackedRunProgress(storage_root=tmp_path / "run-progress")


class _ClockSequence:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)
        self._last_value = values[-1] if values else None

    def __call__(self) -> datetime:
        if not self._values:
            if self._last_value is None:
                raise AssertionError("clock sequence exhausted")
            return self._last_value
        self._last_value = self._values.pop(0)
        return self._last_value


class _RecordingEvaluationStore:
    def __init__(self) -> None:
        self.records: list[MinerTaskRunSubmission] = []

    def record(self, result: MinerTaskRunSubmission) -> None:
        self.records.append(result)


class _FailOnNthRecordEvaluationStore(_RecordingEvaluationStore):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def record(self, result: MinerTaskRunSubmission) -> None:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError("evaluation record write failed")
        super().record(result)


def _record_receipt(
    receipt_log: FakeReceiptLog,
    *,
    session_id,
    uid: int,
    receipt_id: str,
    issued_at: datetime,
    cost_usd: float,
    tool: str = "search_web",
    outcome: ToolCallOutcome = ToolCallOutcome.OK,
    response_payload: dict[str, object] | None = None,
    execution: ToolExecutionFacts | None = None,
    request_payload: dict[str, object] | None = None,
    active_attempt: int | None = 1,
    extra: dict[str, str] | None = None,
) -> None:
    receipt_extra = {} if extra is None else dict(extra)
    if active_attempt is not None:
        receipt_extra["session_active_attempt"] = str(active_attempt)
    receipt_log.record(
        ToolCall(
            receipt_id=receipt_id,
            session_id=session_id,
            uid=uid,
            tool=tool,
            issued_at=issued_at,
            outcome=outcome,
            details=ToolCallDetails(
                request_hash=f"{receipt_id}-req",
                request_payload=request_payload,
                response_hash=f"{receipt_id}-res",
                cost_usd=cost_usd,
                response_payload=response_payload,
                execution=execution,
                extra=receipt_extra or None,
            ),
        )
    )


def _search_usage(receipt_log: FakeReceiptLog, session_id) -> ToolUsageSummary:
    receipts = tuple(receipt_log.for_session(session_id))
    total_cost = sum(float(receipt.details.cost_usd or 0.0) for receipt in receipts)
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
                    total_score=score,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ),
        usage=TokenUsageSummary.empty(),
    )


def _submission_for_task(
    *,
    batch_id,
    validator_uid: int,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    error: EvaluationError | None = None,
) -> MinerTaskRunSubmission:
    issued_at = datetime(2025, 10, 17, 12, 0, tzinfo=UTC)
    session_id = uuid4()
    session = Session(
        session_id=session_id,
        uid=artifact.uid,
        task_id=task.task_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(minutes=5),
        budget_usd=task.budget_usd,
        usage=SessionUsage(),
        status=SessionStatus.ERROR if error is not None else SessionStatus.COMPLETED,
    )
    if error is None:
        run = MinerTaskRun(
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            response=Response(text=f"answer {task.query.text}"),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    total_score=1.0,
                    scoring_version="v1",
                ),
                total_tool_usage=ToolUsageSummary.zero(),
            ),
            completed_at=issued_at,
        )
        return MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=validator_uid,
            run=run,
            score=1.0,
            usage=TokenUsageSummary.empty(),
            session=session,
        )

    run = MinerTaskRun(
        session_id=session_id,
        uid=artifact.uid,
        artifact_id=artifact.artifact_id,
        task_id=task.task_id,
        details=EvaluationDetails(
            error=error,
            total_tool_usage=ToolUsageSummary.zero(),
        ),
        completed_at=issued_at,
    )
    return MinerTaskRunSubmission(
        batch_id=batch_id,
        validator_uid=validator_uid,
        run=run,
        score=0.0,
        usage=TokenUsageSummary.empty(),
        session=session,
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
        receipt_id="receipt-missing-cost",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
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


def test_usage_summarizer_ignores_failed_search_receipts() -> None:
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(),
    )
    successful = ToolCall(
        receipt_id="successful-search",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="successful-search-req",
            response_hash="successful-search-res",
            cost_usd=0.02,
            response_payload={"data": [{"link": "https://example.com"}]},
        ),
    )
    failed = ToolCall(
        receipt_id="failed-search",
        session_id=session.session_id,
        uid=session.uid,
        tool="search_web",
        issued_at=datetime(2025, 10, 17, 12, 2, tzinfo=UTC),
        outcome=ToolCallOutcome.PROVIDER_ERROR,
        details=ToolCallDetails(
            request_hash="failed-search-req",
            request_payload={"args": [], "kwargs": {"query": "Task B"}},
            response_hash=None,
            response_payload=None,
            cost_usd=None,
            extra={"error_type": "ToolProviderError", "error_message": "tool provider failed"},
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (successful, failed))
    assert total_tool_usage.search_tool.call_count == 1
    assert total_tool_usage.search_tool.cost == pytest.approx(0.02)
    assert total_tool_usage.search_tool_cost == pytest.approx(0.02)


def test_usage_summarizer_preserves_reasoning_tokens_in_llm_summary() -> None:
    model = "deepseek-ai/DeepSeek-V3.2-TEE"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=22,
        reasoning_tokens=7,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "chutes": {
                    model: totals,
                },
            },
        ),
    )

    receipt = ToolCall(
        receipt_id="llm-chutes-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-chutes-summary-req",
            request_payload={"kwargs": {"provider": "chutes", "model": model}},
            response_hash="llm-chutes-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0123,
            actual_cost_usd=0.0123,
            actual_cost_provider="chutes",
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))

    model_usage = total_tool_usage.llm.providers["chutes"][model]
    assert total_tool_usage.llm.reasoning_tokens == 7
    assert model_usage.usage.reasoning_tokens == 7
    assert model_usage.cost == pytest.approx(0.0123)


def test_usage_summarizer_rejects_successful_llm_receipt_without_provider() -> None:
    model = "deepseek-ai/DeepSeek-V3.2-TEE"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "chutes": {
                    model: totals,
                },
            },
        ),
    )
    receipt = ToolCall(
        receipt_id="llm-missing-provider-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-missing-provider-summary-req",
            request_payload={"kwargs": {"model": model}},
            response_hash="llm-missing-provider-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0123,
            actual_cost_usd=0.0123,
            actual_cost_provider="chutes",
        ),
    )

    with pytest.raises(ValueError, match="llm_chat receipt requires request provider"):
        UsageSummarizer().summarize(session, (receipt,))


def test_usage_summarizer_prices_openrouter_native_llm_summary() -> None:
    model = "deepseek/deepseek-v3.2"
    totals = LlmUsageTotals(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        call_count=1,
    )
    session = Session(
        session_id=uuid4(),
        uid=7,
        task_id=uuid4(),
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=1.0,
        usage=SessionUsage(
            llm_usage_totals={
                "openrouter": {
                    model: totals,
                },
            },
        ),
    )

    receipt = ToolCall(
        receipt_id="llm-openrouter-summary",
        session_id=session.session_id,
        uid=session.uid,
        tool="llm_chat",
        issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-openrouter-summary-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="llm-openrouter-summary-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.0042,
            actual_cost_usd=0.0042,
            actual_cost_provider="openrouter",
        ),
    )

    _, total_tool_usage = UsageSummarizer().summarize(session, (receipt,))

    model_usage = total_tool_usage.llm.providers["openrouter"][model]
    assert total_tool_usage.llm.call_count == 1
    assert total_tool_usage.llm_cost == pytest.approx(0.0042)
    assert model_usage.cost == pytest.approx(0.0042)


def test_receipt_usage_uses_actual_llm_provider_from_platform_tool_proxy_receipt() -> None:
    model = "openai/gpt-oss-120b"
    receipt = ToolCall(
        receipt_id="llm-openrouter",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-openrouter-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="llm-openrouter-res",
            response_payload={
                "message": {"role": "assistant", "content": "ok"},
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            cost_usd=0.02,
            actual_cost_usd=0.02,
            actual_cost_provider="openrouter",
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))
    assert set(usage.llm_usage_totals) == {"openrouter"}
    assert usage.llm_usage_totals["openrouter"][model].total_tokens == 5
    assert usage.total_cost_usd == pytest.approx(0.02)
    assert usage.cost_by_provider == {"openrouter": pytest.approx(0.02)}
    assert usage.reference_total_cost_usd == pytest.approx(0.02)
    assert usage.reference_cost_by_provider == {"openrouter": pytest.approx(0.02)}
    assert usage.actual_total_cost_usd == pytest.approx(0.02)
    assert usage.actual_cost_by_provider == {"openrouter": pytest.approx(0.02)}


def test_receipt_usage_rejects_successful_llm_receipt_without_provider() -> None:
    model = "openai/gpt-oss-120b"
    receipt = ToolCall(
        receipt_id="llm-missing-provider",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="llm-missing-provider-req",
            request_payload={"kwargs": {"model": model}},
            response_hash="llm-missing-provider-res",
            response_payload={
                "message": {"role": "assistant", "content": "ok"},
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            cost_usd=0.02,
            actual_cost_usd=0.02,
            actual_cost_provider="chutes",
        ),
    )

    with pytest.raises(ValueError, match="llm_chat receipt requires request provider"):
        evaluation_runner_module._usage_from_receipts((receipt,))


def test_receipt_usage_rejects_divergent_search_receipt_costs() -> None:
    receipt = ToolCall(
        receipt_id="search-parallel",
        session_id=uuid4(),
        uid=7,
        tool="search_web",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="search-parallel-req",
            request_payload={"kwargs": {"provider": "parallel", "search_queries": ["harnyx"]}},
            response_hash="search-parallel-res",
            response_payload={"data": []},
            cost_usd=0.0,
            actual_cost_usd=0.005,
            actual_cost_provider="parallel",
        ),
    )
    object.__setattr__(receipt.details, "cost_usd", 0.0)

    with pytest.raises(ValueError, match="must match"):
        evaluation_runner_module._usage_from_receipts((receipt,))


def test_receipt_usage_uses_cost_usd_when_actual_cost_is_missing() -> None:
    receipt = ToolCall(
        receipt_id="cost-only",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="cost-only-req",
            request_payload={"kwargs": {"provider": "chutes"}},
            response_hash="cost-only-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.02,
            reference_cost_usd=0.01,
            actual_cost_usd=None,
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    assert usage.total_cost_usd == pytest.approx(0.02)
    assert usage.reference_total_cost_usd == pytest.approx(0.02)
    assert usage.cost_by_provider == {"chutes": pytest.approx(0.02)}
    assert usage.reference_cost_by_provider == {"chutes": pytest.approx(0.02)}


def test_receipt_replay_records_zero_token_llm_usage_for_priced_receipt() -> None:
    model = "deepseek/deepseek-v3.2"
    receipt = ToolCall(
        receipt_id="priced-zero-token-llm",
        session_id=uuid4(),
        uid=7,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 30, 12, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="priced-zero-token-llm-req",
            request_payload={"kwargs": {"provider": "openrouter", "model": model}},
            response_hash="priced-zero-token-llm-res",
            response_payload={"message": {"role": "assistant", "content": "ok"}},
            cost_usd=0.003,
            reference_cost_usd=0.003,
            actual_cost_usd=0.003,
            actual_cost_provider="openrouter",
        ),
    )

    usage = evaluation_runner_module._usage_from_receipts((receipt,))

    totals = usage.llm_usage_totals["openrouter"][model]
    assert totals.call_count == 1
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0
    assert totals.total_tokens == 0
    assert totals.reasoning_tokens == 0
    assert usage.total_cost_usd == pytest.approx(0.003)


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


def _provider_tool_failure_error() -> SandboxInvocationError:
    return _sandbox_invocation_error(
        "tool route failed",
        status_code=500,
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
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
                details=ToolCallDetails(
                    request_hash="req",
                    response_hash="res",
                    cost_usd=0.25,
                ),
            )
        )
        raise SessionBudgetExhaustedError("session exhausted during entrypoint invocation")


class _ExhaustingAfterPlatformToolProxyControlReceiptOrchestrator:
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
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id="platform-tool-proxy-control",
            issued_at=datetime(2025, 10, 17, 12, 1, tzinfo=UTC),
            cost_usd=0.0,
            outcome=ToolCallOutcome.INTERNAL_ERROR,
            active_attempt=session.active_attempt,
            extra={
                "platform_tool_proxy_error_code": "platform_tool_proxy_denied",
                "platform_tool_proxy_status_code": "403",
                "error_type": "PlatformToolProxyInvocationError",
                "error_message": "platform tool proxy denied",
            },
        )
        self._sessions.update(session.mark_exhausted())
        raise SessionBudgetExhaustedError("session exhausted after proxy control failure")


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
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
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
                comparison_score=0.75,
                total_score=0.75,
                scoring_version="v1",
            ),
            total_tool_usage=_search_usage(self._receipt_log, request.session_id),
        )
        tool_receipts = tuple(self._receipt_log.for_session(request.session_id))
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
            tool_receipts=tool_receipts,
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


class _ScoringTimeoutThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
    ) -> None:
        self._sessions = sessions
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        if self.calls == 1:
            raise httpx.ReadTimeout(
                "embedding timed out",
                request=httpx.Request("POST", "https://validator.invalid/scoring"),
            )
        return _successful_outcome(request)


class _AlwaysMinerTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
        total_tokens: int | None = None,
        elapsed_ms: float | None = None,
        status_code: int = 504,
        detail_exception: str = "TimeoutError",
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self._total_tokens = total_tokens
        self._elapsed_ms = elapsed_ms
        self._status_code = status_code
        self._detail_exception = detail_exception
        self.calls = 0
        self.session_ids: list = []
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        if self._total_tokens is not None and self._elapsed_ms is not None:
            _record_receipt(
                self._receipt_log,
                session_id=request.session_id,
                uid=request.uid,
                receipt_id=f"timeout-{self.calls}",
                issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
                cost_usd=0.0,
                tool="llm_chat",
                request_payload={
                    "args": [],
                    "kwargs": {
                        "provider": "chutes",
                        "model": "google/gemma-4-31B-turbo-TEE",
                    },
                },
                response_payload={"usage": {"total_tokens": self._total_tokens}},
                execution=ToolExecutionFacts(elapsed_ms=self._elapsed_ms),
            )
        raise _sandbox_invocation_error(
            "sandbox entrypoint request timed out",
            status_code=self._status_code,
            detail_exception=self._detail_exception,
            detail_error="sandbox entrypoint request timed out",
        )


class _AlwaysScoringTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
    ) -> None:
        self._sessions = sessions
        self.calls = 0
        self.session_ids: list = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        self.session_ids.append(request.session_id)
        session = self._sessions.get(request.session_id)
        assert session is not None
        raise httpx.ReadTimeout(
            "embedding timed out",
            request=httpx.Request("POST", "https://validator.invalid/scoring"),
        )


class _MinerTimeoutWithNonQualifyingReceiptsOrchestrator:
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
            receipt_id=f"search-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.25,
            tool="search_web",
        )
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"failed-llm-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.0,
            tool="llm_chat",
            outcome=ToolCallOutcome.PROVIDER_ERROR,
            request_payload={
                "args": [],
                "kwargs": {
                    "provider": "chutes",
                    "model": "google/gemma-4-31B-turbo-TEE",
                },
            },
            response_payload={"usage": {"total_tokens": 500}},
            execution=ToolExecutionFacts(elapsed_ms=500.0),
        )
        raise _sandbox_invocation_error(
            "sandbox entrypoint request timed out",
            status_code=504,
            detail_exception="TimeoutError",
            detail_error="sandbox entrypoint request timed out",
        )


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


def _record_provider_failure(
    progress: FileBackedRunProgress,
    *,
    request: MinerTaskRunRequest,
    provider: str = "desearch",
    model: str = "search_web",
    reason: str = "http_402: subscription usage cap exceeded",
) -> None:
    progress.record_provider_call(
        session_id=request.session_id,
        provider=provider,
        model=model,
    )
    progress.record_provider_failure(
        session_id=request.session_id,
        provider=provider,
        model=model,
        reason=reason,
    )


def _record_platform_tool_proxy_timeout_receipt(
    receipt_log: FakeReceiptLog,
    *,
    request: MinerTaskRunRequest,
    active_attempt: int,
    receipt_id: str,
) -> None:
    _record_receipt(
        receipt_log,
        session_id=request.session_id,
        uid=request.uid,
        receipt_id=receipt_id,
        issued_at=datetime(2025, 10, 17, 12, active_attempt, tzinfo=UTC),
        cost_usd=0.0,
        outcome=ToolCallOutcome.TIMEOUT,
        active_attempt=active_attempt,
        extra={
            "platform_tool_proxy_error_code": "tool_timeout",
            "platform_tool_proxy_status_code": "0",
            "error_type": "PlatformToolProxyToolTimeoutError",
            "error_message": "platform tool proxy execution timed out while awaiting tool result",
        },
    )


def _seed_provider_evidence(
    progress: FileBackedRunProgress,
    *,
    batch_id,
    provider: str,
    model: str,
    total_calls: int,
    failed_calls: int,
) -> None:
    for index in range(total_calls):
        session_id = uuid4()
        progress.register_task_session(
            batch_id=batch_id,
            session_id=session_id,
        )
        progress.record_provider_call(
            session_id=session_id,
            provider=provider,
            model=model,
        )
        if index < failed_calls:
            progress.record_provider_failure(
                session_id=session_id,
                provider=provider,
                model=model,
                reason="http_402: subscription usage cap exceeded",
            )
        progress.clear_task_session(session_id)


class _ProviderFailureThenSandboxFailureOrchestrator:
    def __init__(self, *, progress: FileBackedRunProgress) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        if self.calls == 1:
            _record_provider_failure(self._progress, request=request)
            raise _sandbox_invocation_error("tool route failed")
        raise _sandbox_invocation_error("plain sandbox failure")


class _ProviderFailureThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        progress: FileBackedRunProgress,
    ) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        _record_provider_failure(self._progress, request=request)
        return _successful_outcome(request)


class _ProviderBatchFailureOrchestrator:
    def __init__(self, *, progress: FileBackedRunProgress) -> None:
        self._progress = progress
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        _record_provider_failure(self._progress, request=request)
        raise _provider_tool_failure_error()


class _PlatformToolProxyReceiptOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
        error_code: str,
        caught_by_miner: bool,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self._error_code = error_code
        self._caught_by_miner = caught_by_miner
        self.calls = 0

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        _record_receipt(
            self._receipt_log,
            session_id=request.session_id,
            uid=request.uid,
            receipt_id=f"platform-tool-proxy-{self.calls}",
            issued_at=datetime(2025, 10, 17, 12, self.calls, tzinfo=UTC),
            cost_usd=0.0,
            outcome=ToolCallOutcome.INTERNAL_ERROR,
            active_attempt=session.active_attempt,
            extra={
                "platform_tool_proxy_error_code": self._error_code,
                "platform_tool_proxy_status_code": "400",
                "error_type": "PlatformToolProxyInvocationError",
                "error_message": f"{self._error_code} message",
            },
        )
        if self._caught_by_miner:
            return _successful_outcome(request)
        raise _sandbox_invocation_error(
            f"{self._error_code} message",
            status_code=400,
            detail_code="UnhandledException",
            detail_exception="ToolInvocationError",
            detail_error=f"{self._error_code} message",
        )


class _PlatformToolProxyTimeoutThenSuccessOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        if self.calls == 1:
            _record_platform_tool_proxy_timeout_receipt(
                self._receipt_log,
                request=request,
                active_attempt=session.active_attempt,
                receipt_id="platform-tool-proxy-timeout-1",
            )
            raise _provider_tool_failure_error()
        receipts = tuple(self._receipt_log.for_session(request.session_id))
        self._receipt_log.clear_session(request.session_id)
        outcome = _successful_outcome(request)
        return TaskRunOutcome(
            run=outcome.run,
            tool_receipts=receipts,
            usage=outcome.usage,
        )


class _AlwaysPlatformToolProxyTimeoutOrchestrator:
    def __init__(
        self,
        *,
        sessions: FakeSessionRegistry,
        receipt_log: FakeReceiptLog,
    ) -> None:
        self._sessions = sessions
        self._receipt_log = receipt_log
        self.calls = 0
        self.active_attempts: list[int] = []

    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        self.calls += 1
        session = self._sessions.get(request.session_id)
        assert session is not None
        self.active_attempts.append(session.active_attempt)
        _record_platform_tool_proxy_timeout_receipt(
            self._receipt_log,
            request=request,
            active_attempt=session.active_attempt,
            receipt_id=f"platform-tool-proxy-timeout-{self.calls}",
        )
        raise _provider_tool_failure_error()


class _PlatformToolProxyTimeoutThenDifferentMinerExceptionOrchestrator:
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
        _record_platform_tool_proxy_timeout_receipt(
            self._receipt_log,
            request=request,
            active_attempt=session.active_attempt,
            receipt_id="platform-tool-proxy-timeout-before-crash",
        )
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="KeyError",
            detail_error="missing key",
        )


class _UnhandledMinerCrashOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="KeyError",
            detail_error="missing key",
        )


class _UnhandledMinerTypeErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox invocation failed (...)",
            status_code=500,
            detail_code="UnhandledException",
            detail_exception="TypeError",
            detail_error="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
        )


class _MissingEntrypointOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "sandbox entrypoint missing",
            status_code=404,
            detail_code="MissingEntrypoint",
            detail_exception="KeyError",
            detail_error="'query'",
        )


class _PreloadContractFailureOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload contract failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="TypeError",
            detail_error="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
        )


class _PreloadRuntimeErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload runtime failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="RuntimeError",
            detail_error="agent import failed",
        )


class _PreloadImportErrorOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload import failed",
            status_code=500,
            detail_code="PreloadFailed",
            detail_exception="ImportError",
            detail_error="missing miner dependency",
        )


class _PreloadInfrastructureFailureOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "preload infrastructure failed",
            status_code=500,
            detail_code="PreloadInfrastructureFailed",
            detail_exception="RuntimeError",
            detail_error="AGENT_PATH is required",
        )


class _EntrypointUnavailableOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise _sandbox_invocation_error(
            "entrypoint unavailable",
            status_code=500,
            detail_code="EntrypointUnavailable",
            detail_exception="KeyError",
            detail_error="'query'",
        )


class _MinerResponseValidationOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise MinerResponseValidationError("miner returned invalid response payload")


class _ScoringRetryExhaustedOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise LlmRetryExhaustedError("embedding retries exhausted")


class _EmbeddingRetryExhaustedOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        raise VertexEmbeddingRetryExhaustedError("embedding retries exhausted")


class _SuccessfulOrchestrator:
    async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
        return _successful_outcome(request)


async def test_evaluation_runner_records_exhausted_submission(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
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
        progress=progress,
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

    result = await runner.evaluate_artifact(
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
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.validator_uid == 41
    assert submission.score == 0.0
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.response is None
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "session_budget_exhausted"
    assert submission.run.details.error.message == "session exhausted during entrypoint invocation"
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.25)
    assert tuple(receipt.receipt_id for receipt in submission.execution_log) == ("receipt-1",)
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_proxy_control_receipt_overrides_later_budget_exhaustion(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="proxy control before budget exhaustion"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ExhaustingAfterPlatformToolProxyControlReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    with pytest.raises(ValidatorBatchFailedError, match="platform tool proxy control failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert exc_info.value.failure_detail.exception_type == "PlatformToolProxyInvocationError"
    assert exc_info.value.failure_detail.error_message == (
        "platform tool proxy control failure: platform_tool_proxy_denied"
    )
    assert evaluation_store.records == []


async def test_evaluation_runner_retries_transient_invocation_with_new_session_and_accumulated_usage(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
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
        task_retry_count=1,
    )
    orchestrator = _RetryThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[-1]
    assert submission.score == pytest.approx(0.75)
    assert submission.run.details.total_tool_usage.search_tool.call_count == 1
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.25)
    assert sorted(receipt.receipt_id for receipt in submission.execution_log) == ["receipt-2"]
    assert receipt_log.for_session(submission.run.session_id) == ()
    assert evaluation_store.records == [submission]
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert attempt_logs == [("receipt-1",), ("receipt-2",)]


async def test_evaluation_runner_fails_batch_on_generic_post_invoke_failure(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
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


async def test_evaluation_runner_retries_scoring_timeout_with_new_session_before_success(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="scoring timeout retry"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _ScoringTimeoutThenSuccessOrchestrator(
        sessions=session_registry,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert len(result.submissions) == 1
    assert result.submissions[0].run.session_id == orchestrator.session_ids[-1]
    assert result.submissions[0].score == pytest.approx(0.75)
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_sandbox_timeout_terminalizes_without_scheduler_requeue(
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout retry"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=2000.0,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )
    assert evaluation_store.records == list(result.submissions)
    assert session_registry.get(orchestrator.session_ids[0]) is None
    assert tuple(receipt_log.for_session(orchestrator.session_ids[0])) == ()


async def test_evaluation_runner_sandbox_timeout_uses_configured_attempt_budget_with_new_session_per_attempt(
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout retry count three"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=3,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 4
    assert len(set(orchestrator.session_ids)) == 4
    assert orchestrator.active_attempts == [1, 1, 1, 1]
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_fails_batch_after_scoring_timeout_retry_exhaustion(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _AlwaysScoringTimeoutOrchestrator(
        sessions=session_registry,
    )

    with pytest.raises(ValidatorBatchFailedError, match="embedding timed out") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "validator_internal_timeout"
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert evaluation_store.records == []


async def test_evaluation_runner_records_sandbox_boundary_timeout_as_miner_owned(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout within threshold"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=1200.0,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_treats_http_client_timeoutexception_as_sandbox_timeout(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="http client timeout exception"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=1200.0,
        detail_exception="TimeoutException",
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_current_attempt_log_after_sandbox_timeout_exhaustion(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout miner owned"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=2,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        total_tokens=100,
        elapsed_ms=4000.0,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact_with_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 3
    assert len(result.submissions) == 1
    assert evaluation_store.records == list(result.submissions)
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )
    assert result.submissions[0].session.status is SessionStatus.ERROR
    assert tuple(receipt.receipt_id for receipt in result.submissions[0].execution_log) == ("timeout-3",)
    assert all(
        receipt.details.extra == {"session_active_attempt": "1"} for receipt in result.submissions[0].execution_log
    )
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert attempt_logs == [("timeout-1",), ("timeout-2",), ("timeout-3",)]


async def test_evaluation_runner_does_not_treat_non_504_timeouterror_as_sandbox_timeout(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="non-boundary timeout-like sandbox error"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        status_code=500,
        detail_exception="TimeoutError",
    )

    with pytest.raises(
        ValidatorBatchFailedError,
        match="sandbox entrypoint request timed out",
    ) as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.completed_submissions is not None
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="sandbox entrypoint request timed out",
    )
    assert exc.remaining_tasks == ()


async def test_evaluation_runner_terminalizes_sandbox_timeout_without_receipts(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout miner owned without evidence"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _AlwaysMinerTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_current_attempt_receipts_for_terminal_timeout(
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout evidence filter"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=2,
    )
    orchestrator = _MinerTimeoutWithNonQualifyingReceiptsOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_zero_score_for_invalid_miner_response(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="invalid miner response"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _MinerResponseValidationOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="miner_response_invalid",
        message="miner returned invalid response payload",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_zero_score_for_scoring_retry_exhaustion(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="scoring retry exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(ValidatorBatchFailedError, match="embedding retries exhausted") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
        )
    assert exc_info.value.error_code == MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED
    assert exc_info.value.completed_submissions is not None
    submission = exc_info.value.completed_submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_logs_session_summary_for_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluation_runner_module.measurement_logger, "info", capture_info)

    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="successful session log"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    batch_id = uuid4()

    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _SuccessfulOrchestrator()),
    )
    assert len(result.submissions) == 1
    session_logs = [extra for message, extra in captured_logs if message == "miner-task session finished"]
    assert len(session_logs) == 1
    payload = session_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["session_id"] == str(result.submissions[0].run.session_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == artifact.uid
    assert payload["attempt_count"] == 1
    assert payload["session_ms"] >= 0.0
    assert payload["terminal_outcome"] == "submission"
    assert payload["error_code"] is None


async def test_evaluation_runner_logs_session_summary_for_scoring_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluation_runner_module.measurement_logger, "info", capture_info)

    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="scoring retry exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    batch_id = uuid4()

    with pytest.raises(ValidatorBatchFailedError, match="embedding retries exhausted") as exc_info:
        await runner.evaluate_artifact(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _ScoringRetryExhaustedOrchestrator()),
        )
    assert exc_info.value.completed_submissions is not None
    submission = exc_info.value.completed_submissions[0]
    session_logs = [extra for message, extra in captured_logs if message == "miner-task session finished"]
    assert len(session_logs) == 1
    payload = session_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["session_id"] == str(submission.run.session_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == artifact.uid
    assert payload["attempt_count"] == 1
    assert payload["session_ms"] >= 0.0
    assert payload["terminal_outcome"] == "submission"
    assert payload["error_code"] == "scoring_llm_retry_exhausted"


async def test_evaluation_runner_logs_session_summary_for_timeout_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    captured_logs: list[tuple[str, dict[str, object]]] = []

    def capture_info(message: str, *args, **kwargs) -> None:
        captured_logs.append((message, dict(kwargs["extra"]["data"])))

    monkeypatch.setattr(evaluation_runner_module.measurement_logger, "info", capture_info)

    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="timeout miner owned"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=2,
    )
    batch_id = uuid4()

    result = await runner.evaluate_artifact_with_state(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(
            TaskRunOrchestrator,
            _AlwaysMinerTimeoutOrchestrator(
                sessions=session_registry,
                receipt_log=receipt_log,
                total_tokens=100,
                elapsed_ms=4000.0,
            ),
        ),
    )

    session_logs = [extra for message, extra in captured_logs if message == "miner-task session finished"]
    assert len(session_logs) == 1
    payload = session_logs[0]
    assert payload["batch_id"] == str(batch_id)
    assert payload["artifact_id"] == str(artifact.artifact_id)
    assert payload["task_id"] == str(task.task_id)
    assert payload["uid"] == artifact.uid
    assert payload["attempt_count"] == 3
    assert payload["session_ms"] >= 0.0
    assert payload["terminal_outcome"] == "submission"
    assert payload["error_code"] == "timeout_miner_owned"
    assert result.submissions[0].run.details.error == EvaluationError(
        code="timeout_miner_owned",
        message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
    )


async def test_evaluation_runner_records_embedding_retry_exhausted_submission(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
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

    with pytest.raises(ValidatorBatchFailedError, match="embedding retries exhausted") as exc_info:
        await runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _EmbeddingRetryExhaustedOrchestrator()),
        )
    assert exc_info.value.error_code == MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED
    assert exc_info.value.completed_submissions is not None
    submission = exc_info.value.completed_submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code="scoring_llm_retry_exhausted",
        message="embedding retries exhausted",
    )
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_budget_exhausted_when_retry_starts_near_limit(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
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
        task_retry_count=1,
    )
    orchestrator = _RetryThenExhaustedOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert len(set(orchestrator.session_ids)) == 2
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.session_id == orchestrator.session_ids[-1]
    assert submission.session.status is SessionStatus.EXHAUSTED
    assert submission.run.details.error == EvaluationError(
        code="session_budget_exhausted",
        message="session exhausted during retry",
    )
    assert submission.run.details.total_tool_usage.search_tool.call_count == 0
    assert submission.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.0)
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_keeps_valid_response_when_provider_failure_stays_below_threshold(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    batch_id = uuid4()
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure success"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderFailureThenSuccessOrchestrator(
        progress=progress,
    )

    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    assert result.submissions[0].score == pytest.approx(0.75)
    assert evaluation_store.records == list(result.submissions)
    assert progress.provider_evidence(batch_id) == (
        {
            "provider": "desearch",
            "model": "search_web",
            "total_calls": 1,
            "failed_calls": 1,
            "failure_reason": "http_402: subscription usage cap exceeded",
        },
    )


@pytest.mark.parametrize(
    "error_code",
    [
        "platform_tool_proxy_denied",
        "platform_tool_proxy_grant_failed",
        "platform_tool_proxy_execution_failed",
        "platform_error",
    ],
)
async def test_evaluation_runner_platform_proxy_control_categories_fail_validator(
    tmp_path: Path,
    error_code: str,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} caught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=True,
    )

    with pytest.raises(ValidatorBatchFailedError, match="platform tool proxy control failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "unexpected_validator_failure"
    assert exc_info.value.failure_detail.exception_type == "PlatformToolProxyInvocationError"
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


@pytest.mark.parametrize(
    "error_code",
    [
        "provider_failed",
        "tool_timeout",
        "platform_interrupted",
        "budget_exhausted",
        "concurrency_exhausted",
        "miner_credential_missing",
        "unsupported_provider",
        "unsupported_model",
        "invalid_request",
        "duplicate_call",
    ],
)
async def test_evaluation_runner_platform_proxy_miner_owned_categories_do_not_fail_validator(
    tmp_path: Path,
    error_code: str,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text=f"{error_code} caught"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="artifact-hash", size_bytes=128)
    orchestrator = _PlatformToolProxyReceiptOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
        error_code=error_code,
        caught_by_miner=True,
    )

    result = await runner.evaluate_artifact_with_state(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert len(result.submissions) == 1
    assert result.submissions[0].run.details.error is None
    assert evaluation_store.records == list(result.submissions)


async def test_evaluation_runner_retries_platform_tool_proxy_timeout_with_task_retry_count(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform tool proxy timeout retry"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _PlatformToolProxyTimeoutThenSuccessOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert orchestrator.active_attempts == [1, 1]
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == pytest.approx(0.75)
    assert submission.run.details.error is None
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_errors = [
        item["attempt"].error_code
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert "tool_timeout" in attempt_errors
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert ("platform-tool-proxy-timeout-1",) in attempt_logs
    assert any(item["kind"] == "completed_run" for item in page["items"])
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_records_platform_tool_proxy_timeout_as_miner_owned_after_retry_exhaustion(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform tool proxy timeout exhausted"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _AlwaysPlatformToolProxyTimeoutOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    batch_id = uuid4()
    result = await runner.evaluate_artifact(
        batch_id=batch_id,
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 2
    assert orchestrator.active_attempts == [1, 1]
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "timeout_miner_owned"
    page = progress.completed_run_page(batch_id, after_sequence=0, limit=10)
    attempt_errors = [
        item["attempt"].error_code
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert attempt_errors == ["tool_timeout", "tool_timeout"]
    attempt_logs = [
        tuple(receipt.receipt_id for receipt in item["attempt"].execution_log)
        for item in page["items"]
        if item["kind"] == "terminated_attempt" and item["attempt"] is not None
    ]
    assert attempt_logs == [("platform-tool-proxy-timeout-1",), ("platform-tool-proxy-timeout-2",)]
    assert len(submission.execution_log) == 1
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_does_not_rewrite_different_miner_exception_after_timeout_receipt(
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="platform proxy timeout before miner crash"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _PlatformToolProxyTimeoutThenDifferentMinerExceptionOrchestrator(
        sessions=session_registry,
        receipt_log=receipt_log,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, orchestrator),
    )
    assert orchestrator.calls == 1
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_fails_batch_when_successful_fallback_crosses_provider_threshold(
    tmp_path: Path,
) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    _seed_provider_evidence(
        progress,
        batch_id=batch_id,
        provider="desearch",
        model="search_web",
        total_calls=9,
        failed_calls=9,
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure fallback"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderFailureThenSuccessOrchestrator(
        progress=progress,
    )

    with pytest.raises(ValidatorBatchFailedError, match="provider failure threshold reached") as exc_info:
        await runner.evaluate_artifact(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "provider_batch_failure"
    assert exc_info.value.failure_detail.error_message == (
        "provider failure threshold reached "
        "(provider=desearch model=search_web failed_calls=10 total_calls=10 "
        "reason=http_402: subscription usage cap exceeded)"
    )
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


async def test_evaluation_runner_escalates_provider_failure_only_after_batch_threshold(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    _seed_provider_evidence(
        progress,
        batch_id=batch_id,
        provider="desearch",
        model="search_web",
        total_calls=9,
        failed_calls=9,
    )
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider threshold"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=8,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    orchestrator = _ProviderBatchFailureOrchestrator(
        progress=progress,
    )

    with pytest.raises(ValidatorBatchFailedError, match="provider failure threshold reached") as exc_info:
        await runner.evaluate_artifact(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    assert exc_info.value.error_code == "provider_batch_failure"
    assert exc_info.value.failure_detail.error_code == "provider_batch_failure"
    assert exc_info.value.failure_detail.artifact_id == artifact.artifact_id
    assert exc_info.value.failure_detail.task_id == task.task_id
    assert exc_info.value.failure_detail.uid == artifact.uid
    assert "reason=http_402: subscription usage cap exceeded" in exc_info.value.failure_detail.error_message
    assert orchestrator.calls == 1
    assert evaluation_store.records == []


async def test_evaluation_runner_records_zero_score_for_unhandled_miner_exception(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
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

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _UnhandledMinerCrashOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error is not None
    assert submission.run.details.error.code == "miner_unhandled_exception"
    assert evaluation_store.records == [submission]


async def test_evaluation_runner_keeps_query_runtime_type_error_as_miner_unhandled_exception(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="runtime type error"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=cast(TaskRunOrchestrator, _UnhandledMinerTypeErrorOrchestrator()),
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.run.details.error == EvaluationError(
        code="miner_unhandled_exception",
        message="query entrypoint parameter must be annotated as harnyx_miner_sdk.query.Query",
    )


@pytest.mark.parametrize(
    ("orchestrator", "error_code"),
    (
        (_MissingEntrypointOrchestrator(), "script_validation_failed"),
        (_PreloadContractFailureOrchestrator(), "script_validation_failed"),
        (_PreloadRuntimeErrorOrchestrator(), "script_validation_failed"),
        (_PreloadImportErrorOrchestrator(), "script_validation_failed"),
    ),
)
async def test_evaluation_runner_records_zero_score_for_script_validation_failures(
    orchestrator: TaskRunOrchestrator,
    error_code: str,
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="script invalid"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    result = await runner.evaluate_artifact(
        batch_id=uuid4(),
        artifact=artifact,
        tasks=(task,),
        orchestrator=orchestrator,
    )
    assert len(result.submissions) == 1
    submission = result.submissions[0]
    assert submission.score == 0.0
    assert submission.run.details.error == EvaluationError(
        code=error_code,
        message=submission.run.details.error.message,
    )
    assert evaluation_store.records == [submission]


@pytest.mark.parametrize(
    "orchestrator",
    (
        _PreloadInfrastructureFailureOrchestrator(),
        _EntrypointUnavailableOrchestrator(),
    ),
)
async def test_evaluate_artifact_with_state_preserves_sandbox_infrastructure_failures(
    orchestrator: TaskRunOrchestrator,
    tmp_path: Path,
) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="sandbox infra"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(ValidatorBatchFailedError) as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(task,),
            orchestrator=orchestrator,
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.failure_detail.error_code == "sandbox_invocation_failed"


async def test_evaluation_runner_does_not_let_stale_provider_marker_poison_later_attempt(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _RecordingEvaluationStore()
    receipt_log = FakeReceiptLog()
    progress = _progress(tmp_path)
    batch_id = uuid4()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=progress,
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="provider failure then sandbox failure"),
        reference_answer=ReferenceAnswer(text="reference"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
        task_retry_count=1,
    )
    orchestrator = _ProviderFailureThenSandboxFailureOrchestrator(progress=progress)

    with pytest.raises(ValidatorBatchFailedError, match="plain sandbox failure") as exc_info:
        await runner.evaluate_artifact(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert orchestrator.calls == 2
    assert exc.completed_submissions is not None
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="plain sandbox failure",
    )
    assert evaluation_store.records == list(exc.completed_submissions)


async def test_evaluation_runner_uses_bounded_continuous_worker_pool(tmp_path: Path) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=5,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
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

        result = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()
    assert orchestrator.max_active == 5
    assert [submission.run.task_id for submission in result.submissions] == [task.task_id for task in tasks]
    assert len(evaluation_store.records) == 6


async def test_evaluation_runner_keeps_miner_failures_local_and_preserves_input_order(tmp_path: Path) -> None:
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
        progress=_progress(tmp_path),
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
        result = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()
    assert [submission.run.task_id for submission in result.submissions] == [task.task_id for task in tasks]
    assert [submission.score for submission in result.submissions] == [1.0, 0.0, 1.0]
    assert result.submissions[1].run.details.error == EvaluationError(
        code="miner_unhandled_exception",
        message="boom",
    )
    assert len(evaluation_store.records) == 3


async def test_evaluation_runner_fails_batch_after_first_conclusive_validator_owned_submission(tmp_path: Path) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=5,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
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
        task_retry_count=1,
    )

    class _FailFastOrchestrator:
        def __init__(self) -> None:
            self.started_distinct: set[str] = set()
            self.first_wave_started = asyncio.Event()
            self.conclusive_failure_recorded = asyncio.Event()
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}
            self.second_attempt_release_by_text = {task.query.text: asyncio.Event() for task in tasks[:2]}
            self.attempts_by_text: dict[str, int] = {}

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started_distinct.add(text)
            if len(self.started_distinct) == 5:
                self.first_wave_started.set()
            attempt_number = self.attempts_by_text.get(text, 0) + 1
            self.attempts_by_text[text] = attempt_number
            await self.release_by_text[text].wait()
            if text in {"task-0", "task-1"}:
                if attempt_number == 2:
                    await self.second_attempt_release_by_text[text].wait()
                if attempt_number == 2:
                    self.conclusive_failure_recorded.set()
                raise _sandbox_invocation_error("shared sandbox failure")
            return _successful_outcome(request, score=1.0)

    orchestrator = _FailFastOrchestrator()
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
        orchestrator.release_by_text["task-1"].set()
        while orchestrator.attempts_by_text.get("task-0", 0) < 2 or orchestrator.attempts_by_text.get("task-1", 0) < 2:
            await asyncio.sleep(0)
        orchestrator.second_attempt_release_by_text["task-0"].set()
        orchestrator.second_attempt_release_by_text["task-1"].set()
        await asyncio.wait_for(orchestrator.conclusive_failure_recorded.wait(), timeout=1.0)

        for task in tasks[1:]:
            orchestrator.release_by_text[task.query.text].set()

        with pytest.raises(
            ValidatorBatchFailedError,
            match="shared sandbox failure",
        ) as exc_info:
            await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.completed_submissions is not None
    assert [submission.run.task_id for submission in exc.completed_submissions] == [task.task_id for task in tasks[:5]]
    assert exc.remaining_tasks == (tasks[5],)
    recorded_ids = [record.run.task_id for record in evaluation_store.records]
    assert recorded_ids[:5] == [task.task_id for task in tasks[:5]]
    assert tasks[0].task_id in recorded_ids
    assert tasks[1].task_id in recorded_ids


async def test_evaluate_artifact_with_state_preserves_earlier_submissions_for_conclusive_failure(
    tmp_path: Path,
) -> None:
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
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    earlier_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="earlier success"),
        reference_answer=ReferenceAnswer(text="reference earlier"),
    )
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="conclusive later round"),
        reference_answer=ReferenceAnswer(text="reference later"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    earlier_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=earlier_task,
    )

    class _AlwaysSandboxFailureOrchestrator:
        async def evaluate(self, _request: MinerTaskRunRequest) -> TaskRunOutcome:
            raise _sandbox_invocation_error("shared sandbox failure")

    with pytest.raises(ValidatorBatchFailedError, match="shared sandbox failure") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(task,),
            orchestrator=cast(TaskRunOrchestrator, _AlwaysSandboxFailureOrchestrator()),
            earlier_submissions=(earlier_submission,),
        )

    exc = exc_info.value
    assert exc.error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert exc.completed_submissions is not None
    assert exc.completed_submissions[0] == earlier_submission
    assert exc.completed_submissions[1].run.details.error == EvaluationError(
        code="sandbox_invocation_failed",
        message="shared sandbox failure",
    )
    assert exc.remaining_tasks == ()


async def test_evaluate_artifact_with_state_preserves_partial_submissions_for_validator_batch_failure(
    tmp_path: Path,
) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    completed_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="completed"),
        reference_answer=ReferenceAnswer(text="reference completed"),
    )
    pending_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="pending"),
        reference_answer=ReferenceAnswer(text="reference pending"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )

    async def run_artifact_worker(**kwargs) -> None:
        dispatch = kwargs["dispatch"]
        dispatch.submissions_by_index[0] = completed_submission
        dispatch.validator_failure = ValidatorBatchFailedError(
            error_code="validator_internal_timeout",
            message="validator timeout",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="validator_internal_timeout",
                error_message="validator timeout",
                occurred_at=datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
            ),
        )

    runner._run_artifact_worker = run_artifact_worker  # type: ignore[method-assign]

    with pytest.raises(ValidatorBatchFailedError, match="validator timeout") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(completed_task, pending_task),
            orchestrator=cast(TaskRunOrchestrator, object()),
        )

    exc = exc_info.value
    assert exc.completed_submissions == (completed_submission,)
    assert exc.remaining_tasks == (pending_task,)


async def test_evaluate_artifact_with_state_preserves_partial_submissions_for_unexpected_failure(
    tmp_path: Path,
) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    batch_id = uuid4()
    completed_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="completed"),
        reference_answer=ReferenceAnswer(text="reference completed"),
    )
    pending_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="pending"),
        reference_answer=ReferenceAnswer(text="reference pending"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )
    completed_submission = _submission_for_task(
        batch_id=batch_id,
        validator_uid=41,
        artifact=artifact,
        task=completed_task,
    )

    async def record_then_fail(**kwargs) -> None:
        dispatch = kwargs["dispatch"]
        dispatch.submissions_by_index[0] = completed_submission
        dispatch.unexpected_failure = RuntimeError("progress store failed")

    runner._run_artifact_worker = record_then_fail  # type: ignore[method-assign]

    with pytest.raises(UnexpectedArtifactExecutionError, match="progress store failed") as exc_info:
        await runner.evaluate_artifact_with_state(
            batch_id=batch_id,
            artifact=artifact,
            tasks=(completed_task, pending_task),
            orchestrator=cast(TaskRunOrchestrator, object()),
        )

    exc = exc_info.value
    assert exc.completed_submissions == (completed_submission,)
    assert exc.remaining_tasks == (pending_task,)
    assert isinstance(exc.cause, RuntimeError)


async def test_record_failure_for_artifact_preserves_partial_submissions_when_recording_fails(tmp_path: Path) -> None:
    subtensor = FakeSubtensorClient()
    subtensor.validator_metadata = ValidatorNodeInfo(uid=41, version_key=None)
    session_registry = FakeSessionRegistry()
    session_manager = SessionManager(session_registry, InMemoryTokenRegistry())
    evaluation_store = _FailOnNthRecordEvaluationStore(fail_on_call=2)
    receipt_log = FakeReceiptLog()
    runner = EvaluationRunner(
        subtensor_client=subtensor,
        session_manager=session_manager,
        evaluation_records=evaluation_store,
        receipt_log=receipt_log,
        config=SchedulerConfig(token_secret_bytes=8, session_ttl=timedelta(minutes=5)),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    first_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="first"),
        reference_answer=ReferenceAnswer(text="reference first"),
    )
    second_task = MinerTask(
        task_id=uuid4(),
        query=Query(text="second"),
        reference_answer=ReferenceAnswer(text="reference second"),
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    with pytest.raises(UnexpectedArtifactExecutionError, match="evaluation record write failed") as exc_info:
        await runner.record_failure_for_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=(first_task, second_task),
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            error_message="artifact setup failed",
        )

    exc = exc_info.value
    assert len(exc.completed_submissions) == 1
    assert exc.completed_submissions[0].run.task_id == first_task.task_id
    assert exc.remaining_tasks == (second_task,)
    assert isinstance(exc.cause, RuntimeError)


async def test_evaluation_runner_supports_serialized_artifact_execution(tmp_path: Path) -> None:
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
        config=SchedulerConfig(
            token_secret_bytes=8,
            session_ttl=timedelta(minutes=5),
            artifact_task_parallelism=1,
        ),
        clock=lambda: datetime(2025, 10, 17, 12, 0, tzinfo=UTC),
        progress=_progress(tmp_path),
    )
    tasks = tuple(
        MinerTask(
            task_id=uuid4(),
            query=Query(text=f"task-{index}"),
            reference_answer=ReferenceAnswer(text=f"reference-{index}"),
        )
        for index in range(3)
    )
    artifact = ScriptArtifactSpec(
        uid=7,
        artifact_id=uuid4(),
        content_hash="artifact-hash",
        size_bytes=128,
    )

    class _SerializedOrchestrator:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.max_active = 0
            self._active = 0
            self.release_by_text = {task.query.text: asyncio.Event() for task in tasks}
            self.first_started = asyncio.Event()
            self.second_started = asyncio.Event()

        async def evaluate(self, request: MinerTaskRunRequest) -> TaskRunOutcome:
            text = request.task.query.text
            self.started.append(text)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if len(self.started) == 1:
                self.first_started.set()
            if len(self.started) == 2:
                self.second_started.set()
            await self.release_by_text[text].wait()
            self._active -= 1
            return _successful_outcome(request, score=1.0)

    orchestrator = _SerializedOrchestrator()
    execution = asyncio.create_task(
        runner.evaluate_artifact(
            batch_id=uuid4(),
            artifact=artifact,
            tasks=tasks,
            orchestrator=cast(TaskRunOrchestrator, orchestrator),
        )
    )

    try:
        await asyncio.wait_for(orchestrator.first_started.wait(), timeout=1.0)
        assert orchestrator.started == ["task-0"]
        assert not orchestrator.second_started.is_set()

        orchestrator.release_by_text["task-0"].set()
        await asyncio.wait_for(orchestrator.second_started.wait(), timeout=1.0)

        orchestrator.release_by_text["task-1"].set()
        orchestrator.release_by_text["task-2"].set()
        result = await asyncio.wait_for(execution, timeout=1.0)
    finally:
        for release_event in orchestrator.release_by_text.values():
            release_event.set()
    assert orchestrator.max_active == 1
    assert [submission.run.task_id for submission in result.submissions] == [task.task_id for task in tasks]
    assert len(evaluation_store.records) == 3
