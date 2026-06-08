"""Miner-task failure attribution policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NotRequired, Protocol, TypedDict
from uuid import UUID

from harnyx_commons.domain.miner_task import (
    EvaluationError,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.tool_call import IN_FLIGHT_LLM_UNKNOWN_EVIDENCE, ToolCall, ToolCallOutcome
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.json_types import JsonValue

PROVIDER_BATCH_MIN_TOTAL_CALLS = 10
PROVIDER_BATCH_MIN_FAILURE_RATE = 0.95
TIMEOUT_REVIEW_MAX_OBSERVATIONS = 3
TIMEOUT_TPS_SLOWDOWN_FACTOR = 2.0
TERMINAL_TIMEOUT_ERROR_MESSAGE = "terminal timeout"
PLATFORM_TOOL_PROXY_TIMEOUT_ERROR_CODE = "tool_timeout"

SANDBOX_TIMEOUT_EXCEPTIONS = frozenset({"TimeoutError", "TimeoutException"})
SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION = "UnhandledException"
SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT = "MissingEntrypoint"
SANDBOX_DETAIL_CODE_PRELOAD_FAILED = "PreloadFailed"


class ProviderFailureEvidence(TypedDict):
    provider: str
    model: str
    total_calls: int
    failed_calls: int
    failure_reason: NotRequired[str]


class TimeoutAttributionKind(StrEnum):
    MINER_OWNED = "miner_owned"
    NOT_MINER_OWNED = "not_miner_owned"


@dataclass(frozen=True, slots=True)
class SuccessfulLlmSample:
    model: str
    elapsed_ms: float
    total_tokens: int
    ttft_ms: float | None = None
    generation_elapsed_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ingestion_tps: float | None = None
    generation_tps: float | None = None
    legacy_total_tps: float | None = None


@dataclass(frozen=True, slots=True)
class ValidatorLlmSpeedBaseline:
    ingestion_tps: float | None = None
    generation_tps: float | None = None
    legacy_total_tps: float | None = None

    @classmethod
    def from_sample(cls, sample: SuccessfulLlmSample) -> ValidatorLlmSpeedBaseline:
        return cls(
            ingestion_tps=sample.ingestion_tps,
            generation_tps=sample.generation_tps,
            legacy_total_tps=sample.legacy_total_tps,
        )

    def merge(self, other: ValidatorLlmSpeedBaseline) -> ValidatorLlmSpeedBaseline:
        return ValidatorLlmSpeedBaseline(
            ingestion_tps=_slowest_optional_tps(self.ingestion_tps, other.ingestion_tps),
            generation_tps=_slowest_optional_tps(self.generation_tps, other.generation_tps),
            legacy_total_tps=_slowest_optional_tps(self.legacy_total_tps, other.legacy_total_tps),
        )

    def threshold(self) -> ValidatorLlmSpeedBaseline:
        return ValidatorLlmSpeedBaseline(
            ingestion_tps=_threshold_optional_tps(self.ingestion_tps),
            generation_tps=_threshold_optional_tps(self.generation_tps),
            legacy_total_tps=_threshold_optional_tps(self.legacy_total_tps),
        )


@dataclass(frozen=True, slots=True)
class ValidatorModelLlmBaseline:
    slowest_speed_by_model: Mapping[str, ValidatorLlmSpeedBaseline]

    @classmethod
    def empty(cls) -> ValidatorModelLlmBaseline:
        return cls(slowest_speed_by_model={})

    @classmethod
    def from_samples(cls, samples: Sequence[SuccessfulLlmSample]) -> ValidatorModelLlmBaseline:
        slowest_by_model: dict[str, ValidatorLlmSpeedBaseline] = {}
        for sample in samples:
            sample_baseline = ValidatorLlmSpeedBaseline.from_sample(sample)
            current = slowest_by_model.get(sample.model)
            slowest_by_model[sample.model] = sample_baseline if current is None else current.merge(sample_baseline)
        return cls(slowest_speed_by_model=slowest_by_model)

    def threshold_for(self, model: str) -> ValidatorLlmSpeedBaseline | None:
        baseline = self.slowest_speed_by_model.get(model)
        if baseline is None:
            return None
        return baseline.threshold()

    def merge(self, other: ValidatorModelLlmBaseline) -> ValidatorModelLlmBaseline:
        merged = dict(self.slowest_speed_by_model)
        for model, speed in other.slowest_speed_by_model.items():
            current = merged.get(model)
            merged[model] = speed if current is None else current.merge(speed)
        return ValidatorModelLlmBaseline(slowest_speed_by_model=merged)


@dataclass(frozen=True, slots=True)
class TimeoutObservationEvidence:
    successful_llm_samples: tuple[SuccessfulLlmSample, ...]
    session_summary: ToolUsageSummary
    session_elapsed_ms: float
    execution_log: tuple[ToolCall, ...] = ()
    unknown_inflight_llm_count: int = 0


@dataclass(frozen=True, slots=True)
class _ReceiptLlmUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None


class DeliveryRunInput(Protocol):
    @property
    def completed_at(self) -> datetime | None: ...

    @property
    def artifact_id(self) -> UUID: ...

    @property
    def task_id(self) -> UUID: ...


class DeliveryValidatorInput(Protocol):
    @property
    def uid(self) -> int: ...


class DeliverySpecificsInput(Protocol):
    @property
    def error(self) -> EvaluationError | None: ...


class DeliverySubmissionInput(Protocol):
    @property
    def run(self) -> DeliveryRunInput: ...

    @property
    def validator(self) -> DeliveryValidatorInput: ...

    @property
    def specifics(self) -> DeliverySpecificsInput: ...


@dataclass(frozen=True, slots=True)
class ValidatorDeliveryExclusion:
    error: EvaluationError
    occurred_at: datetime
    artifact_id: UUID
    task_id: UUID
    uid: int


def delivery_exclusion_from_completed_pair_results(
    submissions: Sequence[DeliverySubmissionInput],
    *,
    observed_at: datetime,
) -> ValidatorDeliveryExclusion | None:
    for submission in submissions:
        error = submission.specifics.error
        if error is None:
            continue
        if not is_delivery_disqualifying_validator_pair_error(error.code):
            continue
        return ValidatorDeliveryExclusion(
            error=error,
            occurred_at=submission.run.completed_at or observed_at,
            artifact_id=submission.run.artifact_id,
            task_id=submission.run.task_id,
            uid=submission.validator.uid,
        )
    return None


def provider_batch_failure_evidence(
    provider_failures: tuple[ProviderFailureEvidence, ...],
) -> ProviderFailureEvidence | None:
    for evidence in provider_failures:
        if evidence["total_calls"] < PROVIDER_BATCH_MIN_TOTAL_CALLS:
            continue
        if evidence["failed_calls"] / evidence["total_calls"] <= PROVIDER_BATCH_MIN_FAILURE_RATE:
            continue
        return evidence
    return None


def provider_batch_failure_message(evidence: ProviderFailureEvidence) -> str:
    reason = evidence.get("failure_reason")
    if reason:
        return (
            "provider failure threshold reached "
            f"(provider={evidence['provider']} model={evidence['model']} "
            f"failed_calls={evidence['failed_calls']} total_calls={evidence['total_calls']} "
            f"reason={reason})"
        )
    return (
        "provider failure threshold reached "
        f"(provider={evidence['provider']} model={evidence['model']} "
        f"failed_calls={evidence['failed_calls']} total_calls={evidence['total_calls']})"
    )


def is_provider_caused_terminal_failure(
    *,
    detail_code: str | None,
    detail_exception: str | None,
    detail_error: str | None,
) -> bool:
    if detail_code != SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION:
        return False
    if detail_exception != "ToolInvocationError":
        return False
    return detail_error == "tool invocation failed with 400: tool execution failed"


def is_platform_tool_proxy_timeout_receipt(receipt: ToolCall) -> bool:
    if receipt.outcome is not ToolCallOutcome.TIMEOUT:
        return False
    extra = receipt.details.extra or {}
    return extra.get("platform_tool_proxy_error_code") == PLATFORM_TOOL_PROXY_TIMEOUT_ERROR_CODE


def is_uncaught_platform_tool_proxy_timeout_sandbox_invocation(
    *,
    detail_code: str | None,
    detail_exception: str | None,
    detail_error: str | None,
    latest_current_attempt_platform_tool_proxy_receipt_is_timeout: bool,
) -> bool:
    if not latest_current_attempt_platform_tool_proxy_receipt_is_timeout:
        return False
    return is_provider_caused_terminal_failure(
        detail_code=detail_code,
        detail_exception=detail_exception,
        detail_error=detail_error,
    )


def is_timeout_sandbox_invocation(
    *,
    status_code: int | None,
    detail_exception: str | None,
) -> bool:
    return status_code == 504 and detail_exception in SANDBOX_TIMEOUT_EXCEPTIONS


def is_script_validation_sandbox_invocation(*, detail_code: str | None) -> bool:
    return detail_code in {
        SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT,
        SANDBOX_DETAIL_CODE_PRELOAD_FAILED,
    }


def successful_llm_samples(receipts: Sequence[ToolCall]) -> tuple[SuccessfulLlmSample, ...]:
    samples: list[SuccessfulLlmSample] = []
    for receipt in receipts:
        if not receipt.is_successful() or receipt.tool != "llm_chat":
            continue
        model = _receipt_llm_model(receipt)
        if model is None:
            continue
        execution = receipt.details.execution
        if execution is None or execution.elapsed_ms is None or execution.elapsed_ms <= 0:
            continue
        usage = _receipt_llm_usage(receipt)
        if usage is None:
            continue
        sample = _successful_llm_sample(
            model=model,
            elapsed_ms=execution.elapsed_ms,
            ttft_ms=execution.ttft_ms,
            usage=usage,
        )
        if sample is None:
            continue
        samples.append(sample)
    return tuple(samples)


def _successful_llm_sample(
    *,
    model: str,
    elapsed_ms: float,
    ttft_ms: float | None,
    usage: _ReceiptLlmUsage,
) -> SuccessfulLlmSample | None:
    elapsed_seconds = elapsed_ms / 1000.0
    prompt_tokens = _positive_int(usage.prompt_tokens)
    output_tokens = _positive_token_sum(usage.completion_tokens, usage.reasoning_tokens)
    total_tokens = _positive_int(usage.total_tokens)
    legacy_total_tps = None if total_tokens is None else total_tokens / elapsed_seconds
    split_sample = _split_llm_speed_sample(
        model=model,
        elapsed_ms=elapsed_ms,
        ttft_ms=ttft_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        legacy_total_tps=legacy_total_tps,
    )
    if split_sample is not None:
        return split_sample
    if total_tokens is None or legacy_total_tps is None:
        return None
    return SuccessfulLlmSample(
        model=model,
        elapsed_ms=elapsed_ms,
        total_tokens=total_tokens,
        legacy_total_tps=legacy_total_tps,
    )


def _split_llm_speed_sample(
    *,
    model: str,
    elapsed_ms: float,
    ttft_ms: float | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    legacy_total_tps: float | None,
) -> SuccessfulLlmSample | None:
    if ttft_ms is None or ttft_ms <= 0 or ttft_ms >= elapsed_ms:
        return None
    if prompt_tokens is None or output_tokens is None:
        return None
    generation_elapsed_ms = elapsed_ms - ttft_ms
    sample_total_tokens = total_tokens or prompt_tokens + output_tokens
    return SuccessfulLlmSample(
        model=model,
        elapsed_ms=elapsed_ms,
        total_tokens=sample_total_tokens,
        ttft_ms=ttft_ms,
        generation_elapsed_ms=generation_elapsed_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ingestion_tps=prompt_tokens / (ttft_ms / 1000.0),
        generation_tps=output_tokens / (generation_elapsed_ms / 1000.0),
        legacy_total_tps=legacy_total_tps,
    )


def _positive_token_sum(*values: int | None) -> int | None:
    total = sum(value for value in values if isinstance(value, int) and not isinstance(value, bool) and value > 0)
    return total if total > 0 else None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _slowest_optional_tps(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _threshold_optional_tps(value: float | None) -> float | None:
    if value is None:
        return None
    return value / TIMEOUT_TPS_SLOWDOWN_FACTOR


def classify_timeout_attribution(
    *,
    observation: TimeoutObservationEvidence,
    validator_model_llm_baseline: ValidatorModelLlmBaseline,
    prior_timeout_observations: tuple[TimeoutObservationEvidence, ...],
    attempt_budget_exhausted: bool = False,
) -> TimeoutAttributionKind | None:
    comparable_samples = tuple(
        sample
        for timeout_observation in (*prior_timeout_observations, observation)
        for sample in timeout_observation.successful_llm_samples
        if _sample_has_comparable_speed(sample, validator_model_llm_baseline)
    )
    if not attempt_budget_exhausted:
        return None
    if any(_is_slow_llm_sample(sample, validator_model_llm_baseline) for sample in comparable_samples):
        return TimeoutAttributionKind.NOT_MINER_OWNED
    unknown_inflight_count = sum(
        timeout_observation.unknown_inflight_llm_count
        for timeout_observation in (*prior_timeout_observations, observation)
    )
    if unknown_inflight_count:
        return TimeoutAttributionKind.MINER_OWNED
    if any(_is_fast_llm_sample(sample, validator_model_llm_baseline) for sample in comparable_samples):
        return TimeoutAttributionKind.MINER_OWNED
    return TimeoutAttributionKind.MINER_OWNED


def validator_model_llm_baseline(receipts: Sequence[ToolCall]) -> ValidatorModelLlmBaseline:
    return ValidatorModelLlmBaseline.from_samples(successful_llm_samples(receipts))


def unknown_inflight_llm_count(receipts: Sequence[ToolCall]) -> int:
    return sum(1 for receipt in receipts if _is_unknown_inflight_llm_receipt(receipt))


def _is_slow_llm_sample(
    sample: SuccessfulLlmSample,
    baseline: ValidatorModelLlmBaseline,
) -> bool:
    threshold = baseline.threshold_for(sample.model)
    if threshold is None:
        return False
    if _has_split_speed_comparison(sample, threshold):
        return (
            sample.ingestion_tps is not None
            and sample.generation_tps is not None
            and threshold.ingestion_tps is not None
            and threshold.generation_tps is not None
            and (
                sample.ingestion_tps < threshold.ingestion_tps
                or sample.generation_tps < threshold.generation_tps
            )
        )
    if sample.legacy_total_tps is None or threshold.legacy_total_tps is None:
        return False
    return sample.legacy_total_tps < threshold.legacy_total_tps


def _is_fast_llm_sample(
    sample: SuccessfulLlmSample,
    baseline: ValidatorModelLlmBaseline,
) -> bool:
    threshold = baseline.threshold_for(sample.model)
    if threshold is None:
        return False
    if _has_split_speed_comparison(sample, threshold):
        return (
            sample.ingestion_tps is not None
            and sample.generation_tps is not None
            and threshold.ingestion_tps is not None
            and threshold.generation_tps is not None
            and sample.ingestion_tps >= threshold.ingestion_tps
            and sample.generation_tps >= threshold.generation_tps
        )
    if sample.legacy_total_tps is None or threshold.legacy_total_tps is None:
        return False
    return sample.legacy_total_tps >= threshold.legacy_total_tps


def _sample_has_comparable_speed(
    sample: SuccessfulLlmSample,
    baseline: ValidatorModelLlmBaseline,
) -> bool:
    threshold = baseline.threshold_for(sample.model)
    if threshold is None:
        return False
    if _has_split_speed_comparison(sample, threshold):
        return True
    return sample.legacy_total_tps is not None and threshold.legacy_total_tps is not None


def _has_split_speed_comparison(
    sample: SuccessfulLlmSample,
    threshold: ValidatorLlmSpeedBaseline,
) -> bool:
    return (
        sample.ingestion_tps is not None
        and sample.generation_tps is not None
        and threshold.ingestion_tps is not None
        and threshold.generation_tps is not None
    )


def _receipt_llm_usage(receipt: ToolCall) -> _ReceiptLlmUsage | None:
    response_payload = receipt.details.response_payload
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return _ReceiptLlmUsage(
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        reasoning_tokens=_optional_int(usage.get("reasoning_tokens")),
    )


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _receipt_llm_model(receipt: ToolCall) -> str | None:
    request_payload = receipt.details.request_payload
    raw_model = _raw_model_from_request_payload(request_payload)
    if raw_model is None:
        return None
    return raw_model.strip()


def _is_unknown_inflight_llm_receipt(receipt: ToolCall) -> bool:
    if receipt.tool != "llm_chat":
        return False
    if receipt.is_successful():
        return False
    extra = receipt.details.extra
    return (
        extra is not None
        and extra.get("timeout_attribution_evidence") == IN_FLIGHT_LLM_UNKNOWN_EVIDENCE
    )


def _raw_model_from_request_payload(payload: JsonValue | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct_model = payload.get("model")
    if isinstance(direct_model, str):
        return direct_model

    kwargs = payload.get("kwargs")
    if isinstance(kwargs, dict):
        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str):
            return kwargs_model

    args = payload.get("args")
    if isinstance(args, list) and args:
        first_arg = args[0]
        if isinstance(first_arg, dict):
            arg_model = first_arg.get("model")
            if isinstance(arg_model, str):
                return arg_model
    return None


__all__ = [
    "PROVIDER_BATCH_MIN_FAILURE_RATE",
    "PROVIDER_BATCH_MIN_TOTAL_CALLS",
    "SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT",
    "SANDBOX_DETAIL_CODE_PRELOAD_FAILED",
    "SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION",
    "SANDBOX_TIMEOUT_EXCEPTIONS",
    "TERMINAL_TIMEOUT_ERROR_MESSAGE",
    "TIMEOUT_REVIEW_MAX_OBSERVATIONS",
    "TIMEOUT_TPS_SLOWDOWN_FACTOR",
    "DeliveryRunInput",
    "DeliverySpecificsInput",
    "DeliverySubmissionInput",
    "DeliveryValidatorInput",
    "ProviderFailureEvidence",
    "SuccessfulLlmSample",
    "TimeoutAttributionKind",
    "TimeoutObservationEvidence",
    "ValidatorLlmSpeedBaseline",
    "ValidatorModelLlmBaseline",
    "ValidatorDeliveryExclusion",
    "classify_timeout_attribution",
    "delivery_exclusion_from_completed_pair_results",
    "is_platform_tool_proxy_timeout_receipt",
    "is_provider_caused_terminal_failure",
    "is_script_validation_sandbox_invocation",
    "is_timeout_sandbox_invocation",
    "is_uncaught_platform_tool_proxy_timeout_sandbox_invocation",
    "provider_batch_failure_evidence",
    "provider_batch_failure_message",
    "successful_llm_samples",
    "unknown_inflight_llm_count",
    "validator_model_llm_baseline",
]
