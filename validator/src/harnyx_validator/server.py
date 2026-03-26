"""Entrypoint for running the validator API service under uvicorn."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from harnyx_commons.observability.logging import shutdown_logging
from harnyx_commons.observability.tracing import configure_tracing
from harnyx_validator.infrastructure.http.middleware import request_logging_middleware
from harnyx_validator.infrastructure.http.routes import add_control_routes, add_system_routes, add_tool_routes
from harnyx_validator.infrastructure.observability.logging import (
    configure_logging,
    enable_cloud_logging,
    init_logging,
)
from harnyx_validator.infrastructure.observability.sentry import configure_sentry_from_env
from harnyx_validator.runtime.bootstrap import build_runtime, close_runtime_resources
from harnyx_validator.runtime.evaluation_worker import create_evaluation_worker_from_context
from harnyx_validator.runtime.settings import Settings
from harnyx_validator.runtime.weight_worker import create_weight_worker
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

init_logging()
configure_sentry_from_env()
_settings = Settings.load()
configure_tracing(service_name="harnyx-validator")

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

WORKER_STOP_TIMEOUT_SECONDS = 60
logger = logging.getLogger("harnyx_validator.server")


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
    app = FastAPI(
        title="Harnyx Validator API",
        version=VALIDATOR_RELEASE_VERSION,
        lifespan=lifespan,
    )
    app.middleware("http")(request_logging_middleware)
    add_system_routes(app, _runtime.status_provider)
    add_tool_routes(app, _runtime.tool_route_deps_provider)
    add_control_routes(app, _runtime.control_deps_provider)

    return app


app = create_app()


def main() -> None:
    import asyncio

    import uvicorn

    config = uvicorn.Config(
        app,
        host=_runtime.settings.rpc_listen_host,
        port=_runtime.settings.rpc_port,
        timeout_graceful_shutdown=WORKER_STOP_TIMEOUT_SECONDS,
        # logging already setup
        log_config=None,
    )
    server = uvicorn.Server(config)

    async def _run() -> None:
        server_task = asyncio.create_task(server.serve(), name="uvicorn-server")
        try:
            while not server.started:
                if server_task.done():
                    await server_task
                    return
                await asyncio.sleep(0.05)

            try:
                await asyncio.to_thread(_runtime.register_with_platform)
                _runtime.status_provider.mark_platform_registration_succeeded()
            except Exception as exc:
                _runtime.status_provider.mark_platform_registration_failed(str(exc))
                logger.exception("validator platform registration failed during startup")
                server.should_exit = True
                await server_task
                raise

            await server_task
        finally:
            if not server_task.done():
                server.should_exit = True
                await server_task

    asyncio.run(_run())


__all__ = ["app", "main"]
