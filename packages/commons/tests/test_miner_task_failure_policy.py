from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from harnyx_commons.domain.miner_task import EvaluationError, MinerTaskErrorCode
from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolExecutionFacts
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.miner_task_failure_policy import (
    ProviderFailureEvidence,
    SuccessfulLlmSample,
    TimeoutAttributionKind,
    TimeoutObservationEvidence,
    ValidatorLlmSpeedBaseline,
    ValidatorModelLlmBaseline,
    classify_timeout_attribution,
    delivery_exclusion_from_completed_pair_results,
    is_platform_tool_proxy_timeout_receipt,
    is_provider_caused_terminal_failure,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    is_uncaught_platform_tool_proxy_timeout_sandbox_invocation,
    provider_batch_failure_evidence,
    provider_batch_failure_message,
    successful_llm_samples,
)

TEST_MODEL = "google/gemma-4-31B-turbo-TEE"
OTHER_MODEL = "Qwen/Qwen3.6-27B-TEE"


@dataclass(frozen=True, slots=True)
class _Run:
    completed_at: datetime | None
    artifact_id: UUID
    task_id: UUID


@dataclass(frozen=True, slots=True)
class _Validator:
    uid: int


@dataclass(frozen=True, slots=True)
class _Specifics:
    error: EvaluationError | None


@dataclass(frozen=True, slots=True)
class _Submission:
    run: _Run
    validator: _Validator
    specifics: _Specifics


def test_provider_batch_failure_requires_minimum_calls_and_failure_rate() -> None:
    below_calls: ProviderFailureEvidence = {
        "provider": "chutes",
        "model": "model",
        "total_calls": 9,
        "failed_calls": 9,
    }
    threshold_met: ProviderFailureEvidence = {
        "provider": "chutes",
        "model": "model",
        "total_calls": 20,
        "failed_calls": 20,
    }

    assert provider_batch_failure_evidence((below_calls, threshold_met)) == threshold_met


def test_provider_batch_failure_message_includes_reason_when_available() -> None:
    evidence: ProviderFailureEvidence = {
        "provider": "desearch",
        "model": "search_web",
        "total_calls": 10,
        "failed_calls": 10,
        "failure_reason": "http_402: subscription usage cap exceeded",
    }

    assert provider_batch_failure_message(evidence) == (
        "provider failure threshold reached "
        "(provider=desearch model=search_web failed_calls=10 total_calls=10 "
        "reason=http_402: subscription usage cap exceeded)"
    )


def test_delivery_exclusion_selects_first_validator_owned_completed_pair_failure() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)
    completed_at = datetime(2026, 4, 29, 4, 1, tzinfo=UTC)
    validator_owned_artifact_id = uuid4()
    validator_owned_task_id = uuid4()
    validator_uid = 42

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                completed_at=completed_at,
                artifact_id=validator_owned_artifact_id,
                task_id=validator_owned_task_id,
                validator_uid=validator_uid,
            ),
            _submission(
                code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                completed_at=completed_at,
                validator_uid=validator_uid,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.error.code is MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert decision.artifact_id == validator_owned_artifact_id
    assert decision.task_id == validator_owned_task_id
    assert decision.uid == validator_uid
    assert decision.occurred_at == completed_at


def test_delivery_exclusion_uses_observed_at_when_completed_at_is_missing() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                completed_at=None,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.occurred_at == observed_at


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


def test_timeout_attribution_marks_fast_llm_sample_as_miner_owned_at_budget_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_timeout_attribution_slow_completed_sample_blocks_fast_sample_before_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(
            _llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            _llm_sample(model=TEST_MODEL, ingestion_tps=40.0, generation_tps=80.0),
        ),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(),
        )
        is None
    )


def test_timeout_attribution_prior_slow_sample_blocks_current_fast_sample_at_exhaustion() -> None:
    prior_observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=30.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )
    current_observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=current_observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(prior_observation, prior_observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.NOT_MINER_OWNED
    )


def test_timeout_attribution_uses_model_specific_validator_baseline() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=40.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(OTHER_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(observation, observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_timeout_attribution_waits_until_observations_are_exhausted_without_baseline() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=ValidatorModelLlmBaseline.empty(),
            prior_timeout_observations=(observation,),
        )
        is None
    )
    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=ValidatorModelLlmBaseline.empty(),
            prior_timeout_observations=(observation, observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_timeout_attribution_observation_limit_does_not_terminalize_before_budget_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=ValidatorModelLlmBaseline.empty(),
            prior_timeout_observations=(observation, observation),
        )
        is None
    )


def test_timeout_attribution_fast_sample_does_not_terminalize_before_budget_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(),
        )
        is None
    )


def test_timeout_attribution_keeps_unknown_inflight_unresolved_before_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
        unknown_inflight_llm_count=1,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=ValidatorModelLlmBaseline.empty(),
            prior_timeout_observations=(),
        )
        is None
    )


def test_timeout_attribution_defaults_unknown_inflight_to_miner_owned_at_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
        unknown_inflight_llm_count=1,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=ValidatorModelLlmBaseline.empty(),
            prior_timeout_observations=(observation, observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_timeout_attribution_slow_completed_sample_beats_unknown_at_exhaustion() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, ingestion_tps=40.0, generation_tps=80.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
        unknown_inflight_llm_count=1,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, ingestion_tps=100.0, generation_tps=80.0),
            prior_timeout_observations=(observation, observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.NOT_MINER_OWNED
    )


def test_successful_llm_samples_include_model_identity() -> None:
    session_id = uuid4()
    receipt = ToolCall(
        receipt_id="receipt-1",
        session_id=session_id,
        uid=1,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 13, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="request-hash",
            request_payload={
                "args": [],
                "kwargs": {"model": TEST_MODEL},
            },
            response_hash="response-hash",
            response_payload={
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 50,
                    "reasoning_tokens": 10,
                    "total_tokens": 100,
                }
            },
            execution=ToolExecutionFacts(elapsed_ms=1000.0, ttft_ms=200.0),
        ),
    )

    samples = successful_llm_samples((receipt,))

    assert samples == (
        SuccessfulLlmSample(
            model=TEST_MODEL,
            elapsed_ms=1000.0,
            total_tokens=100,
            ttft_ms=200.0,
            generation_elapsed_ms=800.0,
            prompt_tokens=40,
            output_tokens=60,
            ingestion_tps=200.0,
            generation_tps=75.0,
            legacy_total_tps=100.0,
        ),
    )


def test_successful_llm_samples_accept_openrouter_native_model_identity() -> None:
    model = "deepseek/deepseek-v3.2"
    receipt = _llm_receipt(
        model=model,
        response_usage={
            "prompt_tokens": 40,
            "completion_tokens": 50,
            "reasoning_tokens": 10,
            "total_tokens": 100,
        },
        execution=ToolExecutionFacts(elapsed_ms=1000.0, ttft_ms=200.0),
    )

    samples = successful_llm_samples((receipt,))

    assert samples == (
        SuccessfulLlmSample(
            model=model,
            elapsed_ms=1000.0,
            total_tokens=100,
            ttft_ms=200.0,
            generation_elapsed_ms=800.0,
            prompt_tokens=40,
            output_tokens=60,
            ingestion_tps=200.0,
            generation_tps=75.0,
            legacy_total_tps=100.0,
        ),
    )


def test_successful_llm_samples_fall_back_to_legacy_total_tps_without_valid_ttft() -> None:
    receipt = _llm_receipt(
        response_usage={"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100},
        execution=ToolExecutionFacts(elapsed_ms=1000.0),
    )

    samples = successful_llm_samples((receipt,))

    assert samples == (
        SuccessfulLlmSample(
            model=TEST_MODEL,
            elapsed_ms=1000.0,
            total_tokens=100,
            legacy_total_tps=100.0,
        ),
    )


def test_validator_llm_baseline_keeps_split_and_legacy_slowest_speeds() -> None:
    baseline = ValidatorModelLlmBaseline.from_samples(
        (
            _llm_sample(
                model=TEST_MODEL,
                ingestion_tps=120.0,
                generation_tps=100.0,
                legacy_total_tps=90.0,
            ),
            _llm_sample(
                model=TEST_MODEL,
                ingestion_tps=80.0,
                generation_tps=140.0,
                legacy_total_tps=70.0,
            ),
        )
    )

    assert baseline.slowest_speed_by_model[TEST_MODEL] == ValidatorLlmSpeedBaseline(
        ingestion_tps=80.0,
        generation_tps=100.0,
        legacy_total_tps=70.0,
    )


def test_timeout_attribution_uses_legacy_total_tps_when_ttft_is_not_comparable() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(_llm_sample(model=TEST_MODEL, legacy_total_tps=40.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            validator_model_llm_baseline=_baseline(TEST_MODEL, legacy_total_tps=100.0),
            prior_timeout_observations=(observation, observation),
            attempt_budget_exhausted=True,
        )
        is TimeoutAttributionKind.NOT_MINER_OWNED
    )


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


def _llm_sample(
    *,
    model: str,
    ingestion_tps: float | None = None,
    generation_tps: float | None = None,
    legacy_total_tps: float | None = None,
) -> SuccessfulLlmSample:
    total_tokens = int(legacy_total_tps or ingestion_tps or generation_tps or 1)
    return SuccessfulLlmSample(
        model=model,
        elapsed_ms=1000.0,
        total_tokens=total_tokens,
        ttft_ms=100.0 if ingestion_tps is not None else None,
        generation_elapsed_ms=900.0 if generation_tps is not None else None,
        prompt_tokens=10 if ingestion_tps is not None else None,
        output_tokens=10 if generation_tps is not None else None,
        ingestion_tps=ingestion_tps,
        generation_tps=generation_tps,
        legacy_total_tps=legacy_total_tps,
    )


def _baseline(
    model: str,
    *,
    ingestion_tps: float | None = None,
    generation_tps: float | None = None,
    legacy_total_tps: float | None = None,
) -> ValidatorModelLlmBaseline:
    return ValidatorModelLlmBaseline(
        slowest_speed_by_model={
            model: ValidatorLlmSpeedBaseline(
                ingestion_tps=ingestion_tps,
                generation_tps=generation_tps,
                legacy_total_tps=legacy_total_tps,
            )
        }
    )


def _llm_receipt(
    *,
    response_usage: dict[str, int],
    execution: ToolExecutionFacts,
    model: str = TEST_MODEL,
) -> ToolCall:
    return ToolCall(
        receipt_id="receipt-1",
        session_id=uuid4(),
        uid=1,
        tool="llm_chat",
        issued_at=datetime(2026, 5, 13, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="request-hash",
            request_payload={
                "args": [],
                "kwargs": {"model": model},
            },
            response_hash="response-hash",
            response_payload={"usage": response_usage},
            execution=execution,
        ),
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
    validator_uid: int = 1,
) -> _Submission:
    return _Submission(
        run=_Run(
            completed_at=completed_at,
            artifact_id=artifact_id or uuid4(),
            task_id=task_id or uuid4(),
        ),
        validator=_Validator(uid=validator_uid),
        specifics=_Specifics(error=EvaluationError(code=code, message=f"{code.value} happened")),
    )
