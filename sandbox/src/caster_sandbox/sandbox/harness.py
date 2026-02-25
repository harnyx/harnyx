"""Utilities for binding agent entrypoints to a FastAPI sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import inspect
import logging
import multiprocessing
import os
import traceback
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Protocol, cast

import pyseccomp as seccomp
from fastapi import APIRouter, HTTPException, Request

from caster_miner_sdk._internal.tool_invoker import bind_tool_invoker
from caster_miner_sdk.decorators import (
    EntrypointRegistry,
    get_entrypoint,
    get_entrypoint_registry,
)
from caster_sandbox.context.snapshot import ContextSnapshot

ToolConfig = Mapping[str, Any] | None
ToolHeaders = Mapping[str, str]
ToolFactory = Callable[[ToolConfig, ToolHeaders], Any]


@dataclass
class EntrypointRequest:
    payload: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] | None = None


class MpContext(Protocol):
    def Pipe(self, duplex: bool = True) -> tuple[Connection, Connection]: ...  # noqa: N802 - mirror multiprocessing

    def Process(  # noqa: N802 - mirror multiprocessing
        self,
        *,
        target: Callable[..., Any] | None = None,
        args: tuple[Any, ...] = ...,
    ) -> multiprocessing.Process: ...


logger = logging.getLogger("caster_sandbox.sandbox")

ENTRYPOINT_TIMEOUT_ENV_VAR = "ENTRYPOINT_TIMEOUT_SECONDS"


def _load_entrypoint_timeout_seconds() -> float:
    raw_timeout = os.getenv(ENTRYPOINT_TIMEOUT_ENV_VAR, "120")
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            f"{ENTRYPOINT_TIMEOUT_ENV_VAR} must be a number > 0, got {raw_timeout!r}",
        ) from exc
    if timeout_seconds <= 0:
        raise ValueError(
            f"{ENTRYPOINT_TIMEOUT_ENV_VAR} must be > 0, got {raw_timeout!r}",
        )
    return timeout_seconds


ENTRYPOINT_TIMEOUT_SECONDS = _load_entrypoint_timeout_seconds()
WORKER_KILL_GRACE_SECONDS = 1.0


def _default_mp_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix platforms
        return multiprocessing.get_context()


class SandboxHarness:
    """Coordinates entrypoint invocation for sandboxed agents."""

    def __init__(
        self,
        *,
        registry: EntrypointRegistry | None = None,
        tool_factory: ToolFactory | None = None,
        preload: Callable[[], None] | None = None,
    ) -> None:
        self._registry = registry or get_entrypoint_registry()
        self._tool_factory = tool_factory
        self._preload = preload
        self._mp: MpContext = cast(MpContext, _default_mp_context())

    async def invoke(
        self,
        entrypoint_name: str,
        body: EntrypointRequest,
        *,
        headers: ToolHeaders | None = None,
    ) -> Any:
        request_payload = body.payload
        tool_config = body.tool_config
        context_snapshot = ContextSnapshot(body.context or {})

        call_kwargs = {
            "entrypoint_name": entrypoint_name,
            "request_payload": request_payload,
            "context": context_snapshot.to_dict(),
            "tool_config": tool_config,
            "headers": dict(headers or {}),
            "preload": self._preload,
        }

        return await self._invoke_with_worker(call_kwargs)

    def create_router(self) -> APIRouter:
        """Return a FastAPI router exposing entrypoint invocation endpoints."""
        router = APIRouter()

        @router.post(
            "/{entrypoint_name}",
            tags=["entrypoints"],
            description="Invoke a registered entrypoint by name in a sandboxed worker process.",
        )
        async def dispatch(
            entrypoint_name: str,
            body: EntrypointRequest,
            request: Request,
        ) -> dict[str, Any]:
            headers = request.headers
            try:
                result = await self.invoke(entrypoint_name, body, headers=headers)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except HTTPException:
                raise
            except Exception as exc:
                session_id = headers.get("x-caster-session-id")
                logger.exception(
                    "sandbox entrypoint failed",
                    extra={
                        "entrypoint": entrypoint_name,
                        "session_id": session_id,
                    },
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(exc),
                        "exception": exc.__class__.__name__,
                    },
                ) from exc
            return {"ok": True, "result": result}

        return router

    @staticmethod
    def _build_call_kwargs(
        func: Callable[..., Any],
        request_payload: Any,
        context_snapshot: ContextSnapshot,
        tool_proxy: Any,
    ) -> dict[str, Any]:
        del func, context_snapshot, tool_proxy
        return {"request": request_payload}

    async def _invoke_with_worker(self, payload: Mapping[str, Any]) -> Any:
        process, parent_conn = self._spawn_worker(payload)
        try:
            result_kind, result_data = await self._await_worker_result(parent_conn, payload, process)
            return self._unwrap_worker_result(result_kind, result_data)
        finally:
            parent_conn.close()
            self._join_process(process)

    def _spawn_worker(self, payload: Mapping[str, Any]) -> tuple[multiprocessing.Process, Connection]:
        parent_conn, child_conn = self._mp.Pipe(duplex=False)
        process = self._mp.Process(
            target=_entrypoint_worker,
            args=(
                payload["entrypoint_name"],
                payload["request_payload"],
                payload["context"],
                payload["tool_config"],
                payload["headers"],
                self._tool_factory,
                payload["preload"],
                child_conn,
            ),
        )
        process.start()
        child_conn.close()
        return process, parent_conn

    def _unwrap_worker_result(self, kind: str, data: Any) -> Any:
        if kind == "ok":
            return data

        detail = data if isinstance(data, Mapping) else {"error": "entrypoint failed"}
        code = detail.get("code") if isinstance(detail, Mapping) else None
        if code == "MissingEntrypoint":
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    async def _await_worker_result(
        self,
        parent_conn: Connection,
        payload: Mapping[str, Any],
        process: multiprocessing.Process,
    ) -> tuple[str, Any]:
        loop = asyncio.get_running_loop()
        recv_future = loop.run_in_executor(None, parent_conn.recv)
        try:
            return await asyncio.wait_for(recv_future, timeout=ENTRYPOINT_TIMEOUT_SECONDS)
        except TimeoutError as exc:  # pragma: no cover - integration timing
            return self._handle_timeout(process, payload, exc)
        except Exception as exc:  # pragma: no cover - unexpected worker failure
            return self._handle_worker_failure(process, exc)

    def _terminate_process(self, process: multiprocessing.Process) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(WORKER_KILL_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - guardrail
            process.kill()

    def _handle_timeout(
        self,
        process: multiprocessing.Process,
        payload: Mapping[str, Any],
        exc: TimeoutError,
    ) -> tuple[str, Any]:
        self._terminate_process(process)
        session_id = payload["headers"].get("x-caster-session-id")
        logger.exception(
            "sandbox entrypoint timed out",
            extra={
                "entrypoint": payload["entrypoint_name"],
                "session_id": session_id,
                "timeout_seconds": ENTRYPOINT_TIMEOUT_SECONDS,
            },
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": f"entrypoint exceeded {ENTRYPOINT_TIMEOUT_SECONDS}s",
                "exception": "TimeoutError",
            },
        ) from exc

    def _handle_worker_failure(
        self,
        process: multiprocessing.Process,
        exc: Exception,
    ) -> tuple[str, Any]:
        self._terminate_process(process)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "entrypoint worker failed",
                "exception": exc.__class__.__name__,
            },
        ) from exc

    def _join_process(self, process: multiprocessing.Process) -> None:
        process.join(WORKER_KILL_GRACE_SECONDS)
        if process.is_alive():  # pragma: no cover - guardrail
            process.kill()


def _entrypoint_worker(
    entrypoint_name: str,
    request_payload: Mapping[str, Any],
    context_data: Mapping[str, Any],
    tool_config: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    tool_factory: ToolFactory | None,
    preload: Callable[[], None] | None,
    conn: Connection,
) -> None:
    tool_proxy = None
    try:
        _block_new_tasks_in_this_process()
        if preload is not None:
            preload()
        try:
            func = get_entrypoint(entrypoint_name)
        except KeyError as exc:
            _send_worker_error(conn, "MissingEntrypoint", exc)
            return
        context_snapshot = ContextSnapshot(context_data or {})
        if tool_factory is not None:
            tool_proxy = tool_factory(tool_config, headers)
        call_kwargs = SandboxHarness._build_call_kwargs(
            func,
            request_payload,
            context_snapshot,
            tool_proxy,
        )
        if tool_proxy is not None:
            with bind_tool_invoker(tool_proxy):
                result = _execute_entrypoint(func, call_kwargs)
        else:
            result = _execute_entrypoint(func, call_kwargs)
        conn.send(("ok", result))
    except BaseException as exc:  # pragma: no cover - propagated to parent
        _send_worker_error(conn, "UnhandledException", exc)
    finally:
        if tool_proxy is not None:
            with contextlib.suppress(Exception):
                asyncio.run(tool_proxy.aclose())
        conn.close()


def _block_new_tasks_in_this_process() -> None:
    """Install a seccomp filter that denies task-creation syscalls."""

    filter_ = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in ("clone", "clone3", "fork", "vfork", "execve", "execveat"):
        filter_.add_rule(seccomp.ERRNO(errno.EPERM), name)
    filter_.load()
    logger.debug("worker seccomp filter installed", extra={"pid": os.getpid()})


def _send_worker_error(conn: Connection, code: str, exc: BaseException) -> None:
    conn.send(
        (
            "error",
            {
                "code": code,
                "error": str(exc),
                "exception": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            },
        ),
    )


def _execute_entrypoint(func: Callable[..., Any], call_kwargs: Mapping[str, Any]) -> Any:
    if not inspect.iscoroutinefunction(func):
        raise RuntimeError("sandbox entrypoints must be async def")
    coroutine = cast(Coroutine[Any, Any, Any], func(**call_kwargs))
    return asyncio.run(coroutine)


__all__ = ["SandboxHarness", "ToolFactory", "ToolHeaders", "ToolConfig", "EntrypointRequest"]
