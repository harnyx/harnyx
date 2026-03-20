from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain.session import SessionStatus
from harnyx_commons.domain.tool_call import ReceiptMetadata, ToolCall, ToolCallOutcome
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.subtensor import ValidatorNodeInfo
from harnyx_validator.application.scheduler import SchedulerConfig
from harnyx_validator.application.services.evaluation_runner import EvaluationRunner
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
