"""Incremental public LLM output and hosted-tool stream envelopes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from harnyx_miner_sdk.tools.http_models import ToolExecuteResponseDTO

TResult = TypeVar("TResult")


class TextUpdate(BaseModel):
    """Append to or replace the current field; an empty replacement clears it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    operation: Literal["append", "replace"]
    value: str


class ToolUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    index: int
    id: TextUpdate | None = None
    name: TextUpdate | None = None
    arguments: TextUpdate | None = None
    type: str | None = None


class LlmChatChunk(BaseModel):
    """Provisional changes to one choice, in delivery order.

    Missing fields are unchanged. Reasoning details append in order. Tool arguments
    may contain incomplete JSON. A finish reason does not signify settled success.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    choice_index: int
    text: TextUpdate | None = None
    reasoning: TextUpdate | None = None
    reasoning_details: list[dict[str, JsonValue]] | None = None
    tool_calls: list[ToolUpdate] | None = None
    finish_reason: str | None = None


LlmChatOutputSink = Callable[[LlmChatChunk], Awaitable[None]]


class LlmChatOutputFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["output"] = "output"
    chunk: LlmChatChunk


class LlmChatCompletedFrame(BaseModel, Generic[TResult]):
    """The owning hop validates result against its existing response schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["completed"] = "completed"
    result: TResult


class LlmChatErrorFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["error"] = "error"
    detail: str
    # Only the authenticated platform hop sends private billing error details.
    error: dict[str, JsonValue] | None = None


LlmChatFrame = Annotated[
    LlmChatOutputFrame | LlmChatCompletedFrame[JsonValue] | LlmChatErrorFrame,
    Field(discriminator="type"),
]
LLM_CHAT_FRAME_ADAPTER = TypeAdapter(LlmChatFrame)


class LlmChatPublicErrorFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["error"] = "error"
    detail: str


LLM_CHAT_PUBLIC_FRAME_ADAPTER = TypeAdapter(Annotated[
    LlmChatOutputFrame | LlmChatCompletedFrame[ToolExecuteResponseDTO] | LlmChatPublicErrorFrame,
    Field(discriminator="type"),
])
