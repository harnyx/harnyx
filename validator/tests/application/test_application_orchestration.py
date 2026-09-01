from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

import harnyx_validator.application.evaluate_task_run as evaluate_task_run_module
from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.judge_usage import JudgeModelUsage, JudgeUsageSummary
from harnyx_commons.domain.miner_task import (
    EvaluationError,
    EvaluationTrace,
    FastScoreEvidence,
    FastScoreExpectedComponent,
    MinerTask,
    MinerTaskErrorCode,
    Query,
    ReferenceAnswer,
    Response,
    ScoreBreakdown,
)
from harnyx_commons.domain.session import LlmUsageTotals, Session, SessionStatus
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.miner_task_scoring import EvaluationScoringResult
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor, ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.usage_tracker import UsageTracker
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    PlatformOwnedTaskExecution,
    TokenUsageSummary,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, score_platform_execution
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker
from validator.tests.fixtures.fakes import FakeReceiptLog, FakeSessionRegistry

pytestmark = pytest.mark.anyio("asyncio")

TEST_SESSION_TOKEN = uuid4().hex


async def test_miner_task_run_request_requires_execution_time_limit() -> None:
    with pytest.raises(ValidationError, match="execution_time_limit_seconds"):
        MinerTaskRunRequest(
            batch_id=uuid4(),
            session_id=uuid4(),
            token=TEST_SESSION_TOKEN,
            uid=7,
            artifact_id=uuid4(),
            task=MinerTask(
                task_id=uuid4(),
                query=Query(text="Harnyx Subnet demo"),
                reference_answer=ReferenceAnswer(text="A direct answer"),
            ),
        )


class StubSandboxClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object], dict[str, object], str, UUID]] = []
        self.response: dict[str, object] | None = None
        self.on_invoke: Callable[[UUID], None] | None = None

    def set_response(self, response: dict[str, object]) -> None:
        self.response = response

    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: dict[str, object],
        context: dict[str, object],
        token: str,
        session_id: UUID,
    ) -> dict[str, object]:
        self.requests.append((entrypoint, payload, context, token, session_id))
        if self.on_invoke is not None:
            self.on_invoke(session_id)
        if self.response is None:
            raise RuntimeError("sandbox response not configured")
        return self.response


class EchoToolInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def invoke(
        self,
        tool_name: str,
        *,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        context: ToolInvocationContext | None = None,
    ) -> ToolInvocationOutput:
        self.calls.append((tool_name, args, kwargs))
        return ToolInvocationOutput(
            public_payload={
                "data": [
                    {
                        "link": "https://example.com",
                        "title": "Example",
                        "snippet": "ref",
                    },
                ],
                "cost_usd": 0.01,
            },
            actual_cost_usd=0.01,
            actual_cost_provider="parallel",
        )


class StubScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        assert task.query.text == "Harnyx Subnet demo"
        assert response.text == "A direct answer"
        return ScoreBreakdown(
            comparison_score=1.0,
            total_score=1.0,
            scoring_version="v1",
        )


class TracingScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> EvaluationScoringResult:
        assert task.query.text == "Harnyx Subnet demo"
        assert response.text == "A direct answer"
        return EvaluationScoringResult(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                total_score=1.0,
                scoring_version="v1",
            ),
            judge_usage=JudgeUsageSummary(
                call_count=2,
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                reasoning_tokens=3,
                actual_cost_usd=None,
                models=(
                    JudgeModelUsage(
                        provider="chutes",
                        model="judge-model",
                        call_count=2,
                        prompt_tokens=20,
                        completion_tokens=10,
                        total_tokens=30,
                        reasoning_tokens=3,
                        actual_cost_usd=None,
                        actual_cost_source="unavailable",
                    ),
                ),
            ),
            evaluation_trace=EvaluationTrace(
                scoring_judge_selected_routes=("chutes/judge-model",),
                scoring_judge_attempt_count=3,
                scoring_judge_retry_count=1,
                scoring_judge_retry_reasons=("transport_error",),
                scoring_judge_duration_ms=250.0,
                scoring_judge_status="ok",
            ),
        )


class FastEvidenceScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> EvaluationScoringResult:
        assert task.query.fast is True
        assert response.text == "A direct answer"
        evidence = FastScoreEvidence(
            expected_components=(FastScoreExpectedComponent(component_id="answer", is_correct=True),),
            excessive_components=(),
            precision=1.0,
            recall=1.0,
        )
        return EvaluationScoringResult(
            score_breakdown=ScoreBreakdown(
                comparison_score=1.0,
                total_score=1.0,
                scoring_version="deepsearchqa-f1-v1",
                fast_score_evidence=evidence,
            ),
            judge_usage=JudgeUsageSummary(
                call_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                reasoning_tokens=0,
                actual_cost_usd=0.0,
                models=(),
            ),
        )


class TrackingScoringService:
    def __init__(self) -> None:
        self.calls = 0

    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        self.calls += 1
        raise AssertionError("scoring should not run for exhausted sessions")


class FailingScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        raise RuntimeError("scoring failed")


class RetryExhaustedScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        raise LlmRetryExhaustedError("scoring retries exhausted")


class RetryExhaustedWithJudgeUsageScoringService:
    async def score(
        self,
        *,
        task: MinerTask,
        response: Response,
    ) -> ScoreBreakdown:
        _ = task, response
        exc = LlmRetryExhaustedError("scoring retries exhausted")
        exc.__dict__["judge_usage"] = _judge_usage()
        raise exc


class _ClockSequence:
    def __init__(self, *values: datetime) -> None:
        self._values = list(values)

    def now(self) -> datetime:
        if not self._values:
            raise AssertionError("clock sequence exhausted")
        return self._values.pop(0)


class _MonotonicSequence:
    def __init__(self, *values: float) -> None:
        self._values = list(values)

    def monotonic(self) -> float:
        if not self._values:
            raise AssertionError("monotonic sequence exhausted")
        return self._values.pop(0)


def _judge_usage() -> JudgeUsageSummary:
    return JudgeUsageSummary(
        call_count=1,
        prompt_tokens=70,
        completion_tokens=30,
        total_tokens=100,
        reasoning_tokens=5,
        actual_cost_usd=0.019,
        models=(
            JudgeModelUsage(
                provider="chutes",
                model="judge-model",
                call_count=1,
                prompt_tokens=70,
                completion_tokens=30,
                total_tokens=100,
                reasoning_tokens=5,
                actual_cost_usd=0.019,
                actual_cost_source="provider_actual",
            ),
        ),
    )


def _platform_execution(task: MinerTask) -> PlatformOwnedTaskExecution:
    session_id = uuid4()
    issued_at = datetime(2025, 10, 17, 12, tzinfo=UTC)
    return PlatformOwnedTaskExecution(
        batch_id=uuid4(),
        artifact_id=uuid4(),
        task_id=task.task_id,
        task=task,
        attempt_number=1,
        max_attempts=1,
        validator_session_id=session_id,
        uid=7,
        miner_hotkey_ss58="miner-hotkey",
        started_at=issued_at,
        execution_completed_at=issued_at,
        response=Response(text="A direct answer"),
        session=Session(
            session_id=session_id,
            uid=7,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=issued_at.replace(hour=13),
            budget_usd=0.5,
            status=SessionStatus.COMPLETED,
        ),
        usage=TokenUsageSummary.empty(),
        total_tool_usage=ToolUsageSummary.zero(),
    )


class TailObservedUsageSummarizer(evaluate_task_run_module.UsageSummarizer):
    def summarize(
        self,
        session: Session,
        receipts: Sequence[ToolCall],
    ) -> tuple[TokenUsageSummary, ToolUsageSummary]:
        evaluate_task_run_module.time.monotonic()
        return super().summarize(session, receipts)


async def test_score_platform_execution_explicit_failed_result_is_response_free() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )

    result = await score_platform_execution(
        RetryExhaustedWithJudgeUsageScoringService(),
        _platform_execution(task),
        convert_scoring_error=True,
    )

    assert result.result is not None
    assert result.result.run.response is None
    assert result.result.run.details.error == EvaluationError(
        code=MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
        message="scoring retries exhausted",
    )
    assert result.result.run.details.scoring_judge_usage == _judge_usage()


async def test_score_platform_execution_does_not_convert_unexpected_scoring_failure() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )

    with pytest.raises(RuntimeError, match="scoring failed"):
        await score_platform_execution(
            FailingScoringService(),
            _platform_execution(task),
            convert_scoring_error=True,
        )


async def test_score_platform_execution_preserves_fast_score_evidence() -> None:
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo", fast=True),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )

    outcome = await score_platform_execution(
        FastEvidenceScoringService(),
        _platform_execution(task),
        convert_scoring_error=True,
    )

    assert outcome.result is not None
    breakdown = outcome.result.run.details.score_breakdown
    assert breakdown is not None
    assert breakdown.fast_score_evidence is not None
    assert breakdown.fast_score_evidence.expected_components[0].component_id == "answer"


async def test_application_use_cases_cooperate_for_single_task_run() -> None:
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()

    session_manager = SessionManager(session_registry, token_registry)

    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)

    tool_invoker = EchoToolInvoker()
    usage_tracker = UsageTracker()

    executor = ToolExecutor(
        session_registry=session_registry,
        receipt_log=receipt_log,
        usage_tracker=usage_tracker,
        tool_invoker=tool_invoker,
        token_registry=token_registry,
        clock=lambda: datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
    )

    await executor.execute(
        ToolInvocationRequest(
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            tool="search_web",
            args=("harnyx subnet",),
            kwargs={"query": "harnyx subnet"},
        ),
    )

    session = session_registry.get(session_request.session_id)
    assert session is not None
    session_registry.update(
        session.with_usage(
            session.usage.update(
                llm_usage_totals={
                    "chutes": {
                        "openai/gpt-oss-20b-TEE": LlmUsageTotals(
                            prompt_tokens=10,
                            completion_tokens=15,
                            total_tokens=25,
                            call_count=1,
                        ),
                    },
                },
                llm_tokens_last_call=25,
            ),
        ),
    )

    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})

    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )

    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=StubScoringService(),
        session_registry=session_registry,
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ).now,
    )

    outcome = await orchestrator.evaluate(
        MinerTaskRunRequest(
            batch_id=uuid4(),
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            uid=7,
            artifact_id=uuid4(),
            execution_time_limit_seconds=300.0,
            task=task,
        ),
    )

    assert outcome.run.response == Response(text="A direct answer")
    assert outcome.run.details.score_breakdown is not None
    assert outcome.run.details.score_breakdown.total_score == pytest.approx(1.0)
    assert outcome.run.details.elapsed_ms == pytest.approx(300000.0)
    assert outcome.run.completed_at == datetime(2025, 10, 17, 12, 10, tzinfo=UTC)
    assert outcome.run.details.total_tool_usage.search_tool_cost == pytest.approx(0.01)
    assert sandbox.requests == [
        (
            "query",
            {"text": "Harnyx Subnet demo"},
            {
                "cost_budget": {
                    "session_budget_usd": 0.5,
                    "session_hard_limit_usd": 0.5,
                    "session_used_budget_usd": 0.01,
                    "session_remaining_budget_usd": 0.49,
                },
                "time_budget": {"limit_seconds": 300.0},
            },
            TEST_SESSION_TOKEN,
            session_request.session_id,
        ),
    ]


async def test_task_orchestration_success_persists_consolidated_evaluation_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic = _MonotonicSequence(
        10.0,  # orchestration starts
        11.0,  # entrypoint invocation completes
        20.0,  # usage summarization observes post-execution time
        23.0,  # execution trace is finalized
        24.0,  # scoring starts
        27.0,  # scoring completes
    )
    monkeypatch.setattr(evaluate_task_run_module, "time", SimpleNamespace(monotonic=monotonic.monotonic))
    session_registry = FakeSessionRegistry()
    receipt_log = FakeReceiptLog()
    token_registry = InMemoryTokenRegistry()
    session_manager = SessionManager(session_registry, token_registry)
    task = MinerTask(
        task_id=uuid4(),
        query=Query(text="Harnyx Subnet demo"),
        reference_answer=ReferenceAnswer(text="A direct answer"),
    )
    session_request = SessionTokenRequest(
        session_id=uuid4(),
        uid=7,
        task_id=task.task_id,
        issued_at=datetime(2025, 10, 17, 12, tzinfo=UTC),
        expires_at=datetime(2025, 10, 17, 13, tzinfo=UTC),
        budget_usd=0.5,
        token=TEST_SESSION_TOKEN,
    )
    session_manager.issue(session_request)
    sandbox = StubSandboxClient()
    sandbox.set_response({"text": "A direct answer"})
    invoker = EntrypointInvoker(
        session_registry=session_registry,
        sandbox_client=sandbox,
        token_registry=token_registry,
        receipt_log=receipt_log,
    )
    orchestrator = TaskRunOrchestrator(
        entrypoint_invoker=invoker,
        receipt_log=receipt_log,
        scoring_service=TracingScoringService(),
        session_registry=session_registry,
        usage_summarizer=TailObservedUsageSummarizer(),
        clock=_ClockSequence(
            datetime(2025, 10, 17, 12, 5, tzinfo=UTC),
            datetime(2025, 10, 17, 12, 10, tzinfo=UTC),
        ).now,
    )

    outcome = await orchestrator.evaluate(
        MinerTaskRunRequest(
            batch_id=uuid4(),
            session_id=session_request.session_id,
            token=TEST_SESSION_TOKEN,
            uid=7,
            artifact_id=uuid4(),
            execution_time_limit_seconds=300.0,
            task=task,
        ),
    )

    trace = outcome.run.details.trace
    assert trace is not None
    assert trace.entrypoint_invocation_ms == pytest.approx(1000.0)
    assert trace.scoring_ms == pytest.approx(3000.0)
    assert trace.orchestration_ms == pytest.approx(4000.0)
    assert trace.scoring_judge_selected_routes == ("chutes/judge-model",)
    assert trace.scoring_judge_attempt_count == 3
    assert trace.scoring_judge_retry_count == 1
    assert trace.scoring_judge_retry_reasons == ("transport_error",)
    assert trace.scoring_judge_duration_ms == pytest.approx(250.0)
    assert trace.scoring_judge_status == "ok"
