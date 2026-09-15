"""Canonical signed endpoint answer and assignment-owned receipt evidence."""

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.domain.tool_call import SearchToolResult, ToolCall, ToolResultPolicy
from harnyx_commons.tools.types import ToolName


class EndpointReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    receipt_id: str
    assignment_id: UUID
    tool: ToolName
    issued_at: datetime
    results: tuple[SearchToolResult, ...]

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> Self:
        if not call.is_successful() or call.details.result_policy is not ToolResultPolicy.REFERENCEABLE:
            raise ValueError("rating receipt must be successful and referenceable")
        results = tuple(result for result in call.details.results if isinstance(result, SearchToolResult))
        if len(results) != len(call.details.results):
            raise ValueError("rating receipt requires normalized search results")
        return cls(
            receipt_id=call.receipt_id,
            assignment_id=call.session_id,
            tool=call.tool,
            issued_at=call.issued_at,
            results=results,
        )


class EndpointAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    assignment_id: UUID
    expected_hotkey: str = Field(min_length=1)
    callback_body_utf8: str
    signature_hex: str = Field(min_length=1)
    signed_callback_path: str = Field(pattern=r"^/")
    receipt_logs: tuple[EndpointReceipt, ...]
