from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from harnyx_validator.infrastructure.http.middleware import request_logging_middleware


def test_request_logging_middleware_includes_method_path_query_and_truncated_body(caplog) -> None:
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)

    @app.post("/rpc")
    async def rpc() -> dict[str, bool]:
        return {"ok": True}

    target_logger = logging.getLogger("harnyx_validator.http")
    original_propagate = target_logger.propagate
    target_logger.propagate = False
    target_logger.addHandler(caplog.handler)
    target_logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)

    body = "y" * 2000
    try:
        client = TestClient(app)
        response = client.post(
            "/rpc",
            params=[("q", "1"), ("q", "2")],
            content=body,
        )
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.propagate = original_propagate

    assert response.status_code == 200

    records = [record for record in caplog.records if record.name == "harnyx_validator.http"]
    received = next(record for record in records if record.msg == "request_received")
    completed = next(record for record in records if record.msg == "request_completed")

    assert received.data["method"] == "POST"
    assert received.data["path"] == "/rpc"
    assert received.data["query_params"] == [("q", "1"), ("q", "2")]
    assert received.data["request_line"] == "POST /rpc?q=1&q=2"
    assert received.data["body"].startswith("y" * 1024)
    assert received.data["body"].endswith("... (truncated)")
    assert received.levelno == logging.INFO

    assert completed.data["method"] == "POST"
    assert completed.data["path"] == "/rpc"
    assert completed.data["query_params"] == [("q", "1"), ("q", "2")]
    assert completed.data["request_line"] == "POST /rpc?q=1&q=2"
    assert completed.data["status_code"] == 200
    assert completed.levelno == logging.INFO


def test_request_logging_middleware_downgrades_probe_paths_to_debug(caplog) -> None:
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    target_logger = logging.getLogger("harnyx_validator.http")
    original_propagate = target_logger.propagate
    target_logger.propagate = False
    target_logger.addHandler(caplog.handler)
    target_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    try:
        client = TestClient(app)
        response = client.get("/healthz")
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.propagate = original_propagate

    assert response.status_code == 200

    records = [record for record in caplog.records if record.name == "harnyx_validator.http"]
    received = next(record for record in records if record.msg == "request_received")
    completed = next(record for record in records if record.msg == "request_completed")

    assert received.levelno == logging.DEBUG
    assert received.data["method"] == "GET"
    assert received.data["path"] == "/healthz"
    assert received.data["request_line"] == "GET /healthz"
    assert received.data["body"] == ""

    assert completed.levelno == logging.DEBUG
    assert completed.data["method"] == "GET"
    assert completed.data["path"] == "/healthz"
    assert completed.data["request_line"] == "GET /healthz"
    assert completed.data["status_code"] == 200
