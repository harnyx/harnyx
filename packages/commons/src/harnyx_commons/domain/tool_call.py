"""Tool call receipts recorded during evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from uuid import UUID

from harnyx_commons.json_types import JsonValue
from harnyx_commons.tools.types import TOOL_NAMES, ToolName


class ToolCallOutcome(StrEnum):
    """High-level outcome for a tool invocation."""

    OK = "ok"
    PROVIDER_ERROR = "provider_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL_ERROR = "internal_error"
    TIMEOUT = "timeout"


class ToolResultPolicy(StrEnum):
    """Indicates whether tool results can be cited."""

    REFERENCEABLE = "referenceable"
    LOG_ONLY = "log_only"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured representation of a tool result for auditing."""

    index: int
    result_id: str
    raw: JsonValue | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must be non-negative")
        if not self.result_id.strip():
            raise ValueError("result_id must not be empty")


@dataclass(frozen=True, slots=True)
class SearchToolResult(ToolResult):
    """Normalized search result that miners may cite."""

    url: str = ""
    note: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        ToolResult.__post_init__(self)
        if not self.url.strip():
            raise ValueError("url must not be empty")


@dataclass(frozen=True, slots=True)
class ToolExecutionFacts:
    """Execution facts captured at the private tool runtime boundary."""

    elapsed_ms: float | None = None
    ttft_ms: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ToolCallDetails:
    """Supplemental details stored alongside a tool call receipt."""

    request_hash: str
    request_payload: JsonValue | None = None
    response_hash: str | None = None
    response_payload: JsonValue | None = None
    results: tuple[ToolResult, ...] = ()
    result_policy: ToolResultPolicy = ToolResultPolicy.LOG_ONLY
    cost_usd: float | None = None
    reference_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    actual_cost_provider: str | None = None
    extra: Mapping[str, str] | None = None
    execution: ToolExecutionFacts | None = None

    def __post_init__(self) -> None:
        if not self.request_hash.strip():
            raise ValueError("request_hash must not be empty")
        if self.response_hash == "":
            raise ValueError("response_hash must not be empty when supplied")
        if self.reference_cost_usd is None and self.cost_usd is not None:
            object.__setattr__(self, "reference_cost_usd", self.cost_usd)
        if self.cost_usd is None and self.reference_cost_usd is not None:
            object.__setattr__(self, "cost_usd", self.reference_cost_usd)
        if self.cost_usd is not None and self.cost_usd < 0.0:
            raise ValueError("cost_usd must be non-negative when supplied")
        if self.reference_cost_usd is not None and self.reference_cost_usd < 0.0:
            raise ValueError("reference_cost_usd must be non-negative when supplied")
        if self.actual_cost_usd is not None and self.actual_cost_usd < 0.0:
            raise ValueError("actual_cost_usd must be non-negative when supplied")
        if self.actual_cost_provider is not None and not self.actual_cost_provider.strip():
            raise ValueError("actual_cost_provider must not be empty when supplied")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Immutable audit trail for a tool invocation."""

    receipt_id: str
    session_id: UUID
    uid: int
    tool: ToolName
    issued_at: datetime
    outcome: ToolCallOutcome
    details: ToolCallDetails

    def __post_init__(self) -> None:
        if not self.receipt_id.strip():
            raise ValueError("receipt_id must not be empty")
        if self.uid <= 0:
            raise ValueError("uid must be positive")
        if self.tool not in TOOL_NAMES:
            raise ValueError(f"unsupported tool {self.tool!r}")

    def is_successful(self) -> bool:
        """Return True when the tool invocation succeeded."""
        return self.outcome == ToolCallOutcome.OK


@dataclass(frozen=True, slots=True)
class StartedToolCall:
    """A hosted tool invocation that has started and can become a final receipt."""

    receipt_id: str
    session_id: UUID
    session_active_attempt: int
    uid: int
    tool: ToolName
    issued_at: datetime
    request_payload: JsonValue | None
    result_policy: ToolResultPolicy
    execution: ToolExecutionFacts

    def __post_init__(self) -> None:
        if not self.receipt_id.strip():
            raise ValueError("receipt_id must not be empty")
        if self.session_active_attempt < 0:
            raise ValueError("session_active_attempt must be non-negative")
        if self.uid <= 0:
            raise ValueError("uid must be positive")
        if self.tool not in TOOL_NAMES:
            raise ValueError(f"unsupported tool {self.tool!r}")

    def materialize(
        self,
        *,
        outcome: ToolCallOutcome,
        response_payload: JsonValue | None = None,
        results: tuple[ToolResult, ...] = (),
        result_policy: ToolResultPolicy | None = None,
        cost_usd: float | None = None,
        reference_cost_usd: float | None = None,
        actual_cost_usd: float | None = None,
        actual_cost_provider: str | None = None,
        extra: Mapping[str, str] | None = None,
        execution: ToolExecutionFacts | None = None,
    ) -> ToolCall:
        """Build the final receipt for this started invocation."""
        effective_reference_cost_usd = (
            cost_usd if reference_cost_usd is None else reference_cost_usd
        )
        merged_extra = {
            "issued_at": self.issued_at.isoformat(),
            "session_active_attempt": str(self.session_active_attempt),
        }
        if cost_usd is not None:
            merged_extra["cost_usd"] = f"{cost_usd:.6f}"
        if effective_reference_cost_usd is not None:
            merged_extra["reference_cost_usd"] = f"{effective_reference_cost_usd:.6f}"
        if actual_cost_usd is not None:
            merged_extra["actual_cost_usd"] = f"{actual_cost_usd:.6f}"
        if actual_cost_provider is not None:
            merged_extra["actual_cost_provider"] = actual_cost_provider
        if extra is not None:
            merged_extra.update(extra)

        return ToolCall(
            receipt_id=self.receipt_id,
            session_id=self.session_id,
            uid=self.uid,
            tool=self.tool,
            issued_at=self.issued_at,
            outcome=outcome,
            details=ToolCallDetails(
                request_hash=_hash_json_payload(self.request_payload),
                request_payload=self.request_payload,
                response_hash=(
                    None if response_payload is None else _hash_json_payload(response_payload)
                ),
                response_payload=response_payload,
                results=results,
                result_policy=(
                    self.result_policy if result_policy is None else result_policy
                ),
                cost_usd=cost_usd,
                reference_cost_usd=effective_reference_cost_usd,
                actual_cost_usd=actual_cost_usd,
                actual_cost_provider=actual_cost_provider,
                extra=merged_extra,
                execution=execution or self.execution,
            ),
        )


def _hash_json_payload(payload: JsonValue | None) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(serialized).hexdigest()


__all__ = [
    "StartedToolCall",
    "ToolCallDetails",
    "ToolExecutionFacts",
    "SearchToolResult",
    "ToolCall",
    "ToolCallOutcome",
    "ToolResult",
    "ToolResultPolicy",
]
