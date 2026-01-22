"""Entrypoint for running the validator RPC service under uvicorn."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from caster_commons.observability.logging import shutdown_logging
from caster_commons.observability.tracing import configure_tracing
from caster_validator.infrastructure.http.middleware import request_logging_middleware
from caster_validator.infrastructure.http.routes import add_control_routes, add_tool_routes
from caster_validator.infrastructure.observability.logging import (
    configure_logging,
    enable_cloud_logging,
    init_logging,
)
from caster_validator.runtime.bootstrap import build_runtime, close_runtime_resources
from caster_validator.runtime.evaluation_worker import create_evaluation_worker_from_context
from caster_validator.runtime.settings import Settings
from caster_validator.runtime.weight_worker import create_weight_worker

init_logging()
configure_tracing(service_name="caster-validator")
_settings = Settings.load()
if _settings.observability.enable_cloud_logging:
    gcp_project = _settings.observability.gcp_project_id
    if gcp_project is None:
        raise RuntimeError("Cloud logging enabled but no GCP project configured")
    enable_cloud_logging(
        gcp_project=gcp_project,
        cloud_log_labels={"service": "validator"},
    )
else:
    configure_logging(
        cloud_logging_enabled=False,
        gcp_project=_settings.observability.gcp_project_id,
        cloud_log_labels=None,
    )

_runtime = build_runtime(_settings)

_evaluation_worker = create_evaluation_worker_from_context(_runtime)
_weight_worker = create_weight_worker(
    submission_service=_runtime.weight_submission_service,
    status_provider=_runtime.status_provider,
)

WORKER_STOP_TIMEOUT_SECONDS = 30 * 60


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _weight_worker.start()
    _evaluation_worker.start()
    yield
    await _evaluation_worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
    _weight_worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
    await close_runtime_resources(_runtime)
    shutdown_logging()


def create_app() -> FastAPI:
    app = FastAPI(title="Caster Validator RPC", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(request_logging_middleware)

    add_tool_routes(app, _runtime.tool_route_deps_provider)
    add_control_routes(app, _runtime.control_deps_provider)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=_runtime.settings.rpc_listen_host,
        port=_runtime.settings.rpc_port,
        timeout_graceful_shutdown=WORKER_STOP_TIMEOUT_SECONDS,
        # logging already setup
        log_config=None,
    )


__all__ = ["app", "main"]
