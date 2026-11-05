"""Closeable, single-consumer LLM output with a settled result after exhaustion."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Protocol

from harnyx_miner_sdk.tools.llm_chat_stream_models import (
    LlmChatChunk,
    LlmChatCompletedFrame,
    LlmChatErrorFrame,
    LlmChatFrame,
    LlmChatOutputFrame,
)
from harnyx_miner_sdk.tools.proxy import ToolInvocationError

if TYPE_CHECKING:
    from harnyx_miner_sdk.api import LlmChatResult


class LlmChatFrameStream(Protocol):
    def __aiter__(self) -> AsyncIterator[LlmChatFrame]: ...
    async def __anext__(self) -> LlmChatFrame: ...
    async def aclose(self) -> None: ...


class LlmChatStream:
    """Yield output only; use async with or aclose when ending consumption early."""

    def __init__(self, frames: LlmChatFrameStream, decode_result: Callable[[object], LlmChatResult]) -> None:
        self._frames = frames
        self._decode_result = decode_result
        self._result: LlmChatResult | None = None
        self._closed = False
        self._reading = False

    @property
    def result(self) -> LlmChatResult:
        if self._result is None:
            raise RuntimeError("llm_chat result is available only after successful stream exhaustion")
        return self._result

    def __aiter__(self) -> LlmChatStream:
        return self

    async def __anext__(self) -> LlmChatChunk:
        if self._closed:
            raise StopAsyncIteration
        if self._reading:
            raise RuntimeError("llm_chat stream supports a single consumer")
        self._reading = True
        try:
            frame = await self._frames.__anext__()
            if self._closed:
                raise ToolInvocationError("llm_chat stream closed during consumption")
            match frame:
                case LlmChatOutputFrame():
                    return frame.chunk
                case LlmChatCompletedFrame():
                    result = self._decode_result(frame.result)
                    await self.aclose()
                    self._result = result
                case LlmChatErrorFrame():
                    raise ToolInvocationError(frame.detail)
        except StopAsyncIteration as exc:
            await self.aclose()
            raise ToolInvocationError("llm_chat stream ended without a completed result") from exc
        except ValueError as exc:
            await self.aclose()
            raise ToolInvocationError("invalid llm_chat completed result") from exc
        except BaseException:
            await self.aclose()
            raise
        finally:
            self._reading = False
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._frames.aclose()

    async def __aenter__(self) -> LlmChatStream:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
