"""Content-free similarity diagnostics and negotiated response envelopes."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

SIMILARITY_STATUS_MAX_BYTES = 8 * 1024
SIMILARITY_TERMINAL_MAX_BYTES = 16 * 1024 * 1024
SIMILARITY_DIAGNOSTICS_MAX_BYTES = 128 * 1024
SIMILARITY_HISTORY_LIMIT = 64
SIMILARITY_DIAGNOSTIC_STRING_LIMIT = 256

TruncatedField = Literal[
    "similarity_invocation_id", "attempt_id", "candidate_model", "model", "provider",
    "upstream_provider", "outcome", "exception_type",
]
DiagnosticText = Annotated[str, Field(max_length=SIMILARITY_DIAGNOSTIC_STRING_LIMIT)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]


class SimilarityAttemptStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    event: Literal["started", "headers_received", "progress", "finished", "candidate_succeeded", "candidate_failed"]
    similarity_invocation_id: DiagnosticText | None = None
    attempt_id: DiagnosticText | None = None
    candidate_model: DiagnosticText | None = None
    candidate_position: Count | None = None
    candidate_count: Count | None = None
    model: DiagnosticText | None = None
    provider: DiagnosticText | None = None
    upstream_provider: DiagnosticText | None = None
    attempt_number: Count | None = None
    attempt_limit: Count | None = None
    outcome: DiagnosticText | None = None
    exception_type: DiagnosticText | None = None
    timeout_phase: Literal["prefill", "inactivity", "total"] | None = None
    timeout_total_seconds: Milliseconds | None = None
    timeout_prefill_seconds: Milliseconds | None = None
    timeout_inactivity_seconds: Milliseconds | None = None
    elapsed_ms: Milliseconds | None = None
    headers_received_ms: Milliseconds | None = None
    first_output_ms: Milliseconds | None = None
    last_output_ms: Milliseconds | None = None
    last_stream_event_ms: Milliseconds | None = None
    time_since_last_stream_event_ms: Milliseconds | None = None
    stream_event_count: Count | None = None
    output_event_count: Count | None = None
    truncated_fields: tuple[TruncatedField, ...] = Field(default=(), max_length=8)


class SimilarityDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    attempts: tuple[SimilarityAttemptStatus, ...] = Field(default=(), max_length=SIMILARITY_HISTORY_LIMIT)
    omitted_attempts: Count = 0

    def bounded(self, max_bytes: int = SIMILARITY_DIAGNOSTICS_MAX_BYTES) -> SimilarityDiagnostics:
        """Retain recent evidence within the same budget on both sides of HTTP."""
        attempts = self.attempts
        omitted = self.omitted_attempts
        while True:
            result = SimilarityDiagnostics(attempts=attempts, omitted_attempts=omitted)
            if len(result.model_dump_json().encode()) <= max_bytes:
                return result
            attempts = attempts[1:]
            omitted += 1


class SimilarityStatusFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["status"] = "status"
    elapsed_ms: Milliseconds
    heartbeat: bool = False
    status: SimilarityAttemptStatus | None = None


class SimilarityResultFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["result"] = "result"
    # Each hop validates this using its existing judgment boundary model.
    result: JsonValue
    diagnostics: SimilarityDiagnostics


class SimilarityErrorFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["error"] = "error"
    status_code: int = Field(ge=400, le=599)
    error: JsonValue
    diagnostics: SimilarityDiagnostics


SimilarityFrame = Annotated[
    SimilarityStatusFrame | SimilarityResultFrame | SimilarityErrorFrame,
    Field(discriminator="type"),
]
SIMILARITY_FRAME_ADAPTER = TypeAdapter(SimilarityFrame)
