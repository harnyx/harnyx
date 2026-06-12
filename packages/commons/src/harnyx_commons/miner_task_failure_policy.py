"""Miner-task failure attribution policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import NotRequired, Protocol, TypedDict
from uuid import UUID

from harnyx_commons.domain.miner_task import (
    EvaluationError,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.tool_call import ToolCall, ToolCallOutcome

PROVIDER_BATCH_MIN_TOTAL_CALLS = 10
PROVIDER_BATCH_MIN_FAILURE_RATE = 0.95
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


__all__ = [
    "PROVIDER_BATCH_MIN_FAILURE_RATE",
    "PROVIDER_BATCH_MIN_TOTAL_CALLS",
    "SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT",
    "SANDBOX_DETAIL_CODE_PRELOAD_FAILED",
    "SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION",
    "SANDBOX_TIMEOUT_EXCEPTIONS",
    "TERMINAL_TIMEOUT_ERROR_MESSAGE",
    "DeliveryRunInput",
    "DeliverySpecificsInput",
    "DeliverySubmissionInput",
    "DeliveryValidatorInput",
    "ProviderFailureEvidence",
    "ValidatorDeliveryExclusion",
    "delivery_exclusion_from_completed_pair_results",
    "is_platform_tool_proxy_timeout_receipt",
    "is_provider_caused_terminal_failure",
    "is_script_validation_sandbox_invocation",
    "is_timeout_sandbox_invocation",
    "is_uncaught_platform_tool_proxy_timeout_sandbox_invocation",
    "provider_batch_failure_evidence",
    "provider_batch_failure_message",
]
