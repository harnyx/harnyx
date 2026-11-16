"""Own live similarity diagnostics and final delivery for one HTTP request."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress

import anyio
from starlette.responses import JSONResponse, StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from harnyx_commons.llm.judge_observability import observe_judge_events
from harnyx_commons.miner_task_similarity_stream import (
    SIMILARITY_DIAGNOSTIC_STRING_LIMIT,
    SIMILARITY_DIAGNOSTICS_MAX_BYTES,
    SIMILARITY_HISTORY_LIMIT,
    SIMILARITY_STATUS_MAX_BYTES,
    SIMILARITY_TERMINAL_MAX_BYTES,
    SimilarityAttemptStatus,
    SimilarityDiagnostics,
    SimilarityErrorFrame,
    SimilarityResultFrame,
    SimilarityStatusFrame,
)
from harnyx_commons.task_ownership import wait_for_owned_task

_HEARTBEAT_SECONDS = 30.0
_SEND_TIMEOUT_SECONDS = 60.0


class SimilarityStreamingResponse(StreamingResponse):
    def __init__(self, execute: Callable[[], Awaitable[JSONResponse]]) -> None:
        self._execute = execute
        self._started = asyncio.get_running_loop().time()
        self._changed = asyncio.Event()
        self._latest: SimilarityAttemptStatus | None = None
        self._attempts: OrderedDict[str, SimilarityAttemptStatus] = OrderedDict()
        self._omitted_attempts = 0
        self._terminal: SimilarityResultFrame | SimilarityErrorFrame | None = None
        self._failure: Exception | None = None
        super().__init__(
            self._body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    def _observe(self, event: str, data: Mapping[str, object]) -> None:
        if event not in {
            "started", "headers_received", "progress", "finished", "candidate_succeeded", "candidate_failed"
        }:
            return
        # The candidate transition has no provider-attempt id. It updates only
        # the latest attempt from the same invocation and candidate position.
        previous = self._latest
        same_candidate = previous is not None and (
            previous.similarity_invocation_id == data.get("similarity_invocation_id")
            and previous.candidate_position == data.get("candidate_position")
        )
        same_attempt = previous is not None and previous.attempt_id == data.get("attempt_id")
        values: dict[str, object] = {}
        if previous is not None and (same_attempt or (event.startswith("candidate_") and same_candidate)):
            values.update(previous.model_dump())
        truncated: set[str] = set(previous.truncated_fields) if values and previous is not None else set()
        for name, value in data.items():
            if name not in SimilarityAttemptStatus.model_fields or name in {"event", "truncated_fields"}:
                continue
            if isinstance(value, str):
                original = value
                value = value[:SIMILARITY_DIAGNOSTIC_STRING_LIMIT]
                # Escapes and multibyte characters also consume wire capacity.
                while len(json.dumps(value, ensure_ascii=False).encode()) > SIMILARITY_DIAGNOSTIC_STRING_LIMIT + 2:
                    value = value[:len(value) // 2]
                if value != original:
                    truncated.add(name)
            if event.startswith("candidate_") and same_candidate and name in {"model", "provider"}:
                continue
            values[name] = value
        timeout = data.get("timeout")
        if isinstance(timeout, Mapping):
            for phase in ("total", "prefill", "inactivity"):
                values[f"timeout_{phase}_seconds"] = dict(timeout).get(phase)
        values["event"] = event
        values["truncated_fields"] = tuple(sorted(truncated))
        snapshot = SimilarityAttemptStatus.model_validate(values)
        self._latest = snapshot
        key = snapshot.attempt_id or f"{snapshot.similarity_invocation_id}:{snapshot.candidate_position}"
        self._attempts[key] = snapshot
        if len(self._attempts) > SIMILARITY_HISTORY_LIMIT:
            self._attempts.popitem(last=False)
            self._omitted_attempts += 1
        self._changed.set()

    def _diagnostics(self) -> SimilarityDiagnostics:
        return SimilarityDiagnostics(
            attempts=tuple(self._attempts.values()), omitted_attempts=self._omitted_attempts
        ).bounded(SIMILARITY_DIAGNOSTICS_MAX_BYTES)

    async def _produce(self) -> None:
        try:
            with observe_judge_events(self._observe):
                response = await self._execute()
            payload = json.loads(bytes(response.body))
            diagnostics = self._diagnostics()
            if response.status_code < 400:
                self._terminal = SimilarityResultFrame(result=payload, diagnostics=diagnostics)
            else:
                self._terminal = SimilarityErrorFrame(
                    status_code=response.status_code, error=payload, diagnostics=diagnostics
                )
        except Exception as exc:
            self._failure = exc
        finally:
            self._changed.set()

    async def _body(self) -> AsyncIterator[bytes]:
        heartbeat = True
        while True:
            self._changed.clear()
            if self._failure is not None:
                raise self._failure
            terminal = self._terminal
            if terminal is not None:
                encoded = terminal.model_dump_json().encode()
                if len(encoded) > SIMILARITY_TERMINAL_MAX_BYTES:
                    raise ValueError("similarity terminal response exceeds the stream limit")
                yield b"event: " + terminal.type.encode() + b"\ndata: " + encoded + b"\n\n"
                return
            frame = SimilarityStatusFrame(
                elapsed_ms=(asyncio.get_running_loop().time() - self._started) * 1000,
                heartbeat=heartbeat,
                status=self._latest,
            )
            encoded = frame.model_dump_json().encode()
            if len(encoded) > SIMILARITY_STATUS_MAX_BYTES:
                raise ValueError("similarity status exceeds the stream limit")
            yield b"event: status\ndata: " + encoded + b"\n\n"
            try:
                async with asyncio.timeout(_HEARTBEAT_SECONDS):
                    await self._changed.wait()
                heartbeat = False
            except TimeoutError:
                heartbeat = True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        producer = asyncio.create_task(self._produce(), name="similarity-response-judge")

        async def bounded_send(message: Message) -> None:
            async with asyncio.timeout(_SEND_TIMEOUT_SECONDS):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            if not producer.done() and not producer.cancelling():
                producer.cancel()
            with anyio.CancelScope(shield=True):
                cancellation = await wait_for_owned_task(producer)
                with suppress(BaseException):
                    producer.result()
            if cancellation is not None:
                raise cancellation
