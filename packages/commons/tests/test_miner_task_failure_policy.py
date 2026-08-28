from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from harnyx_commons.domain.miner_task import (
    EvaluationError,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome
from harnyx_commons.miner_task_failure_policy import (
    delivery_exclusion_from_completed_pair_results,
    is_platform_tool_proxy_timeout_receipt,
    is_provider_caused_terminal_failure,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    is_uncaught_platform_tool_proxy_timeout_sandbox_invocation,
)

TEST_MODEL = "google/gemma-4-31B-turbo-TEE"


@dataclass(frozen=True, slots=True)
class _Run:
    completed_at: datetime | None
    artifact_id: UUID
    task_id: UUID


@dataclass(frozen=True, slots=True)
class _Specifics:
    error: EvaluationError | None


@dataclass(frozen=True, slots=True)
class _Submission:
    run: _Run
    specifics: _Specifics


def test_delivery_exclusion_selects_first_validator_owned_completed_pair_failure() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)
    completed_at = datetime(2026, 4, 29, 4, 1, tzinfo=UTC)
    validator_owned_artifact_id = uuid4()
    validator_owned_task_id = uuid4()

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                completed_at=completed_at,
                artifact_id=validator_owned_artifact_id,
                task_id=validator_owned_task_id,
            ),
            _submission(
                code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                completed_at=completed_at,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.error.code is MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert decision.artifact_id == validator_owned_artifact_id
    assert decision.task_id == validator_owned_task_id
    assert decision.occurred_at == completed_at


def test_delivery_exclusion_uses_observed_at_when_completed_at_is_missing() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
                completed_at=None,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.occurred_at == observed_at


def test_delivery_exclusion_ignores_historical_timeout_inconclusive_pair_failures() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (_submission(code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE),),
        observed_at=observed_at,
    )

    assert decision is None


def test_delivery_exclusion_ignores_miner_owned_pair_failures() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED),
            _submission(code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED),
        ),
        observed_at=observed_at,
    )

    assert decision is None


def test_platform_tool_proxy_control_failures_disqualify_validator_delivery() -> None:
    assert is_delivery_disqualifying_validator_pair_error(MinerTaskErrorCode.PLATFORM_TOOL_PROXY_DENIED)
    assert is_delivery_disqualifying_validator_pair_error(MinerTaskErrorCode.PLATFORM_TOOL_PROXY_GRANT_FAILED)


def test_sandbox_failure_shape_classifiers_expose_validator_attribution_policy() -> None:
    assert is_timeout_sandbox_invocation(status_code=504, detail_exception="TimeoutError")
    assert is_script_validation_sandbox_invocation(detail_code="MissingEntrypoint")
    assert is_provider_caused_terminal_failure(
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
    )


def test_platform_tool_proxy_timeout_receipt_is_miner_owned_timeout_evidence() -> None:
    receipt = _tool_call(
        outcome=ToolCallOutcome.TIMEOUT,
        extra={"platform_tool_proxy_error_code": "tool_timeout"},
    )

    assert is_platform_tool_proxy_timeout_receipt(receipt)


def test_platform_tool_proxy_control_receipt_is_not_timeout_evidence() -> None:
    receipt = _tool_call(
        outcome=ToolCallOutcome.INTERNAL_ERROR,
        extra={"platform_tool_proxy_error_code": "platform_tool_proxy_denied"},
    )

    assert not is_platform_tool_proxy_timeout_receipt(receipt)


def test_platform_tool_proxy_interrupted_receipt_is_not_timeout_evidence() -> None:
    receipt = _tool_call(
        outcome=ToolCallOutcome.INTERNAL_ERROR,
        extra={"platform_tool_proxy_error_code": "platform_interrupted"},
    )

    assert not is_platform_tool_proxy_timeout_receipt(receipt)


def test_uncaught_platform_tool_proxy_timeout_requires_timeout_receipt_and_tool_failure_shape() -> None:
    assert is_uncaught_platform_tool_proxy_timeout_sandbox_invocation(
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
        latest_current_attempt_platform_tool_proxy_receipt_is_timeout=True,
    )
    assert not is_uncaught_platform_tool_proxy_timeout_sandbox_invocation(
        detail_code="UnhandledException",
        detail_exception="KeyError",
        detail_error="missing key",
        latest_current_attempt_platform_tool_proxy_receipt_is_timeout=True,
    )
    assert not is_uncaught_platform_tool_proxy_timeout_sandbox_invocation(
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
        latest_current_attempt_platform_tool_proxy_receipt_is_timeout=False,
    )


def _tool_call(
    *,
    outcome: ToolCallOutcome,
    extra: dict[str, str] | None = None,
) -> ToolCall:
    return ToolCall(
        receipt_id="receipt-1",
        session_id=uuid4(),
        uid=1,
        tool="search_web",
        issued_at=datetime(2026, 5, 13, tzinfo=UTC),
        outcome=outcome,
        details=ToolCallDetails(
            request_hash="request-hash",
            response_hash="response-hash",
            extra=extra,
        ),
    )


def _submission(
    *,
    code: MinerTaskErrorCode,
    completed_at: datetime | None = None,
    artifact_id: UUID | None = None,
    task_id: UUID | None = None,
) -> _Submission:
    return _Submission(
        run=_Run(
            completed_at=completed_at,
            artifact_id=artifact_id or uuid4(),
            task_id=task_id or uuid4(),
        ),
        specifics=_Specifics(error=EvaluationError(code=code, message=f"{code.value} happened")),
    )
