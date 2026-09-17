from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("harnyx_validator.http")
_LOW_NOISE_PROBE_PATHS = frozenset({"/healthz", "/readyz"})
_REQUEST_ID_SCOPE_KEY = "harnyx_request_id"


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    request_id = request.headers.get("x-request-id", uuid4().hex)
    request.state.request_id = request_id
    request.scope[_REQUEST_ID_SCOPE_KEY] = request_id
    log_level = logging.DEBUG if request.url.path in _LOW_NOISE_PROBE_PATHS else logging.INFO
    request_line = _format_request_line(request)
    query_params = list(request.query_params.multi_items())

    body_str = (
        "<redacted>" if "/validator/endpoint-assignments" in request.url.path else _truncate_body(await request.body())
    )
    logger.log(
        log_level,
        "request_received",
        extra={
            "data": {
                "request_id": request_id,
                "request_line": request_line,
                "method": request.method,
                "path": request.url.path,
                "query_params": query_params,
                "body": body_str,
            },
        },
    )

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "data": {
                    "request_id": request_id,
                    "request_line": request_line,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": query_params,
                },
            },
        )
        raise

    duration = time.perf_counter() - start
    status_code = response.status_code

    logger.log(
        log_level,
        "request_completed",
        extra={
            "data": {
                "request_id": request_id,
                "request_line": request_line,
                "method": request.method,
                "path": request.url.path,
                "query_params": query_params,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        },
    )
    return response


def _format_request_line(request: Request) -> str:
    query = request.url.query
    if query:
        return f"{request.method} {request.url.path}?{query}"
    return f"{request.method} {request.url.path}"


def _truncate_body(body: bytes, limit: int = 1024) -> str:
    try:
        text = body.decode("utf-8")
        if len(text) <= limit:
            return text
        return text[:limit] + "... (truncated)"
    except UnicodeDecodeError:
        return f"<binary data: {len(body)} bytes>"


__all__ = ["request_logging_middleware"]
