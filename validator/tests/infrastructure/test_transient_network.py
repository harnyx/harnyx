from __future__ import annotations

import errno
import socket

import httpx
import pytest

from harnyx_validator.infrastructure.transient_network import classify_transient_network_failure


def _wrap_with_cause(cause: BaseException) -> RuntimeError:
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = cause
    return wrapper


def _wrap_with_context(context: BaseException) -> RuntimeError:
    wrapper = RuntimeError("wrapper")
    wrapper.__context__ = context
    return wrapper


def test_classifies_temporary_dns_gaierror() -> None:
    cause = classify_transient_network_failure(
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )

    assert cause is not None
    assert cause.kind == "temporary_dns"
    assert cause.exception_type == "gaierror"
    assert cause.errno == socket.EAI_AGAIN


def test_classifies_top_level_connection_error_with_eai_again() -> None:
    cause = classify_transient_network_failure(
        ConnectionError(socket.EAI_AGAIN, "Temporary failure in name resolution")
    )

    assert cause is not None
    assert cause.kind == "temporary_dns"
    assert cause.exception_type == "ConnectionError"
    assert cause.errno == socket.EAI_AGAIN


def test_classifies_httpx_connect_timeout() -> None:
    cause = classify_transient_network_failure(httpx.ConnectTimeout("connect timed out"))

    assert cause is not None
    assert cause.kind == "connect_timeout"
    assert cause.exception_type == "ConnectTimeout"
    assert cause.errno is None


@pytest.mark.parametrize(
    "message",
    [
        "timed out while waiting for handshake response",
        "_ssl.c:1108: The handshake operation timed out",
    ],
)
def test_classifies_websocket_handshake_timeouts(message: str) -> None:
    cause = classify_transient_network_failure(TimeoutError(message))

    assert cause is not None
    assert cause.kind == "websocket_handshake_timeout"
    assert cause.exception_type == "TimeoutError"
    assert cause.errno is None


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError(errno.ECONNRESET, "connection reset"),
        TimeoutError(errno.ETIMEDOUT, "connection timed out"),
    ],
)
def test_classifies_connection_interrupted_errno_values(exc: OSError) -> None:
    cause = classify_transient_network_failure(exc)

    assert cause is not None
    assert cause.kind == "connection_interrupted"
    assert cause.exception_type == type(exc).__name__
    assert cause.errno == exc.errno


def test_classifies_nested_cause_or_context() -> None:
    cause_from_cause = classify_transient_network_failure(
        _wrap_with_cause(socket.gaierror(socket.EAI_AGAIN, "temporary dns"))
    )
    cause_from_context = classify_transient_network_failure(
        _wrap_with_context(ConnectionResetError(errno.ECONNRESET, "connection reset"))
    )

    assert cause_from_cause is not None
    assert cause_from_cause.kind == "temporary_dns"
    assert cause_from_context is not None
    assert cause_from_context.kind == "connection_interrupted"


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("connection failed"),
        TimeoutError("_ssl.c:line: The handshake operation timed out"),
        httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
    ],
)
def test_does_not_classify_non_transient_shapes(exc: BaseException) -> None:
    assert classify_transient_network_failure(exc) is None
