"""Tool call receipts recorded during evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from caster_commons.json_types import JsonValue
from caster_commons.tools.types import TOOL_NAMES, ToolName


class ToolCallOutcome(str, Enum):
    """High-level outcome for a tool invocation."""

    OK = "ok"
    PROVIDER_ERROR = "provider_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"


class ToolResultPolicy(str, Enum):
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
        if self.note == "":
            raise ValueError("note must not be empty when supplied")


@dataclass(frozen=True, slots=True)
class ReceiptMetadata:
    """Supplemental metadata stored alongside a tool call receipt."""

    request_hash: str
    response_hash: str | None = None
    response_payload: JsonValue | None = None
    results: tuple[ToolResult, ...] = ()
    result_policy: ToolResultPolicy = ToolResultPolicy.LOG_ONLY
    cost_usd: float | None = None
    extra: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.request_hash.strip():
            raise ValueError("request_hash must not be empty")
        if self.response_hash == "":
            raise ValueError("response_hash must not be empty when supplied")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Immutable audit trail for a tool invocation."""

    receipt_id: str
    session_id: UUID
    uid: int
    tool: ToolName
    issued_at: datetime
    outcome: ToolCallOutcome
    metadata: ReceiptMetadata

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


__all__ = [
    "ReceiptMetadata",
    "SearchToolResult",
    "ToolCall",
    "ToolCallOutcome",
    "ToolResult",
    "ToolResultPolicy",
]
