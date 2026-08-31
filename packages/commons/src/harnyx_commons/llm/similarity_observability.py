"""Content-free stream progress events for similarity-judge LLM attempts."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse

SIMILARITY_JUDGE_USE_CASE = "miner_task_similarity_judge"
_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
_IDENTITY_FIELDS = (
    "similarity_invocation_id",
    "batch_id",
    "candidate_artifact_id",
    "reference_artifact_id",
    "candidate_model",
    "candidate_position",
    "candidate_count",
)
_LOGGER = logging.getLogger("harnyx_commons.llm.calls")


def similarity_judge_observability_metadata(
    *,
    similarity_invocation_id: str,
    batch_id: object,
    candidate_artifact_id: object,
    reference_artifact_id: object,
    candidate_model: str,
    candidate_position: int,
    candidate_count: int,
) -> dict[str, object]:
    """Build the content-free identity shared by candidate and provider events."""
    return {
        "similarity_invocation_id": similarity_invocation_id,
        "batch_id": str(batch_id),
        "candidate_artifact_id": str(candidate_artifact_id),
        "reference_artifact_id": str(reference_artifact_id),
        "candidate_model": candidate_model,
        "candidate_position": candidate_position,
        "candidate_count": candidate_count,
    }


@dataclass(slots=True)
class SimilarityLlmAttemptObservation:
    """Track one actual provider attempt without retaining streamed content."""

    provider: str
    model: str
    attempt_number: int
    attempt_limit: int
    identity: Mapping[str, object]
    logger: logging.Logger = _LOGGER
    clock: Callable[[], float] = time.perf_counter
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    _started_at: float = field(init=False)
    _headers_received_at: float | None = field(default=None, init=False)
    _first_output_at: float | None = field(default=None, init=False)
    _last_stream_event_at: float | None = field(default=None, init=False)
    _last_progress_log_at: float | None = field(default=None, init=False)
    _max_stream_event_gap_ms: float = field(default=0.0, init=False)
    _stream_event_count: int = field(default=0, init=False)
    _output_event_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._started_at = self.clock()
        self._log("started", self._common_data())

    def headers_received(self) -> None:
        now = self.clock()
        if self._headers_received_at is None:
            self._headers_received_at = now
            self._log(
                "headers_received",
                self._common_data() | {"headers_received_ms": self._elapsed_ms(now)},
            )

    def stream_event(self, *, saw_output: bool) -> None:
        now = self.clock()
        if self._last_stream_event_at is not None:
            gap_ms = (now - self._last_stream_event_at) * 1000
            self._max_stream_event_gap_ms = max(self._max_stream_event_gap_ms, gap_ms)
        self._last_stream_event_at = now
        self._stream_event_count += 1
        first_output_observed = saw_output and self._first_output_at is None
        if saw_output:
            self._output_event_count += 1
            if first_output_observed:
                self._first_output_at = now
        if (
            first_output_observed
            or self._last_progress_log_at is None
            or now - self._last_progress_log_at >= _PROGRESS_LOG_INTERVAL_SECONDS
        ):
            self._last_progress_log_at = now
            self._log("progress", self._common_data() | self._progress_data(now))

    def finish_response(self, response: LlmResponse) -> None:
        usage = response.usage
        data = self._common_data() | self._terminal_data("response_received")
        data |= {
            "response_id": response.id,
            "finish_reason": response.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage is not None else None,
            "completion_tokens": usage.completion_tokens if usage is not None else None,
            "reasoning_tokens": usage.reasoning_tokens if usage is not None else None,
            "total_tokens": usage.total_tokens if usage is not None else None,
        }
        self._log("finished", data)

    def finish_failure(self, exc: Exception, *, retryable: bool) -> None:
        self._log(
            "finished",
            self._common_data()
            | self._terminal_data("failed")
            | {"exception_type": type(exc).__name__, "retryable": retryable},
        )

    def finish_cancelled(self) -> None:
        self._log("finished", self._common_data() | self._terminal_data("cancelled"))

    def _common_data(self) -> dict[str, object]:
        return {
            "use_case": SIMILARITY_JUDGE_USE_CASE,
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "model": self.model,
            "attempt_number": self.attempt_number,
            "attempt_limit": self.attempt_limit,
        } | _safe_identity(self.identity)

    def _terminal_data(self, outcome: str) -> dict[str, object]:
        now = self.clock()
        return {"outcome": outcome, "elapsed_ms": self._elapsed_ms(now)} | self._progress_data(now)

    def _progress_data(self, now: float) -> dict[str, object]:
        return {
            "headers_received_ms": self._since_start_ms(self._headers_received_at),
            "first_output_ms": self._since_start_ms(self._first_output_at),
            "last_stream_event_ms": self._since_start_ms(self._last_stream_event_at),
            "time_since_last_stream_event_ms": (
                round((now - self._last_stream_event_at) * 1000, 2) if self._last_stream_event_at is not None else None
            ),
            "max_stream_event_gap_ms": round(self._max_stream_event_gap_ms, 2),
            "stream_event_count": self._stream_event_count,
            "output_event_count": self._output_event_count,
        }

    def _since_start_ms(self, observed_at: float | None) -> float | None:
        return self._elapsed_ms(observed_at) if observed_at is not None else None

    def _elapsed_ms(self, observed_at: float) -> float:
        return round((observed_at - self._started_at) * 1000, 2)

    def _log(self, phase: str, data: Mapping[str, object]) -> None:
        try:
            self.logger.info(
                f"similarity_judge.llm_attempt.{phase}",
                extra={"data": dict(data)},
            )
        except Exception:
            # Monitoring must never alter the provider attempt it observes.
            return


_ACTIVE_ATTEMPT: ContextVar[SimilarityLlmAttemptObservation | None] = ContextVar(
    "similarity_llm_attempt",
    default=None,
)


@contextmanager
def observe_similarity_llm_attempt(
    *,
    provider: str,
    request: AbstractLlmRequest,
    attempt_number: int,
    attempt_limit: int,
) -> Iterator[SimilarityLlmAttemptObservation | None]:
    """Activate attempt tracking only for the similarity-judge use case."""
    if request.use_case != SIMILARITY_JUDGE_USE_CASE:
        yield None
        return
    observation = SimilarityLlmAttemptObservation(
        provider=provider,
        model=request.model,
        attempt_number=attempt_number,
        attempt_limit=attempt_limit,
        identity=request.internal_metadata or {},
    )
    token = _ACTIVE_ATTEMPT.set(observation)
    try:
        yield observation
    finally:
        _ACTIVE_ATTEMPT.reset(token)


def record_similarity_stream_headers_received() -> None:
    observation = _ACTIVE_ATTEMPT.get()
    if observation is not None:
        observation.headers_received()


def record_similarity_stream_event(*, saw_output: bool) -> None:
    observation = _ACTIVE_ATTEMPT.get()
    if observation is not None:
        observation.stream_event(saw_output=saw_output)


def _safe_identity(metadata: Mapping[str, Any]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key in _IDENTITY_FIELDS:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, str | int):
            safe[key] = value
    return safe


__all__ = [
    "SIMILARITY_JUDGE_USE_CASE",
    "observe_similarity_llm_attempt",
    "record_similarity_stream_event",
    "record_similarity_stream_headers_received",
    "similarity_judge_observability_metadata",
]
