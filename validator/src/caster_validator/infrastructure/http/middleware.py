from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("caster_validator.http")


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    request_id = request.headers.get("x-request-id", uuid4().hex)
    request_line = _format_request_line(request)
    query_params = list(request.query_params.multi_items())

    body_bytes = await request.body()
    body_str = _truncate_body(body_bytes)
    logger.info(
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

    logger.info(
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
