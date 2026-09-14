"""Bounded JSON messages on a single query's private connection."""

from __future__ import annotations

import asyncio
import json
import math
import struct
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

MAX_FRAME_BYTES = 64 * 1024 * 1024


class WorkerResultProtocolError(RuntimeError):
    """A worker sent an invalid or incomplete frame."""


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Invocation(Message):
    entrypoint: str
    payload: dict[str, JsonValue]
    context: dict[str, JsonValue]


class ToolCall(Message):
    kind: Literal["tool"] = "tool"
    call_id: int = Field(ge=0)
    method: str
    args: list[JsonValue]
    kwargs: dict[str, JsonValue]


class WorkerError(Message):
    code: str
    exception: str
    message: str


class QueryResult(Message):
    kind: Literal["result"] = "result"
    result: JsonValue = None
    error: WorkerError | None = None


class ToolReply(Message):
    call_id: int
    result: JsonValue = None
    error: str | None = None


WorkerMessage = Annotated[ToolCall | QueryResult, Field(discriminator="kind")]
WORKER_MESSAGE = TypeAdapter(WorkerMessage)


def load_json(payload: bytes) -> object:
    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise WorkerResultProtocolError("JSON number exceeds finite float range")
        return result

    def reject_constant(value: str) -> object:
        raise WorkerResultProtocolError(f"invalid JSON constant: {value}")

    def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerResultProtocolError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(payload, parse_float=finite_float, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def decode_worker_message(payload: bytes) -> WorkerMessage:
    return WORKER_MESSAGE.validate_python(load_json(payload))


def encode(message: BaseModel) -> bytes:
    payload = json.dumps(message.model_dump(mode="json"), allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME_BYTES:
        raise WorkerResultProtocolError("worker frame exceeds size limit")
    return struct.pack(">Q", len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    try:
        length = struct.unpack(">Q", await reader.readexactly(8))[0]
        if length > MAX_FRAME_BYTES:
            raise WorkerResultProtocolError("worker frame exceeds size limit")
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise WorkerResultProtocolError("worker connection closed without a complete frame") from exc


async def send(writer: asyncio.StreamWriter, message: BaseModel) -> None:
    writer.write(encode(message))
    await writer.drain()
