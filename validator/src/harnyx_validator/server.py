"""Entrypoint for running the validator API service under uvicorn."""

from __future__ import annotations

import logging
import os
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
from harnyx_validator.runtime.registration_worker import create_registration_refresh_worker
from harnyx_validator.runtime.settings import Settings
from harnyx_validator.runtime.weight_worker import create_weight_worker
from harnyx_validator.version import VALIDATOR_RELEASE_VERSION

init_logging()
configure_sentry_from_env()
_settings = Settings.load()
configure_tracing(service_name="harnyx-validator")

_COMPOSE_SMOKE_MARKER_ENV = "VALIDATOR_COMPOSE_SMOKE"
_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_ENV = "VALIDATOR_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_SECONDS"


def _smoke_weight_worker_poll_interval_seconds() -> float | None:
    if os.getenv(_COMPOSE_SMOKE_MARKER_ENV) != "1":
        return None
    raw_value = os.getenv(_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_ENV)
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError("smoke weight-worker poll interval must be a number") from exc
    if value <= 0:
        raise RuntimeError("smoke weight-worker poll interval must be positive")
    return value


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
_platform_work_worker = _runtime.platform_work_worker
_weight_worker_poll_interval_seconds = _smoke_weight_worker_poll_interval_seconds()
if _weight_worker_poll_interval_seconds is None:
    _weight_worker = create_weight_worker(
        submission_service=_runtime.weight_submission_service,
        status_provider=_runtime.status_provider,
    )
else:
    _weight_worker = create_weight_worker(
        submission_service=_runtime.weight_submission_service,
        status_provider=_runtime.status_provider,
        poll_interval_seconds=_weight_worker_poll_interval_seconds,
    )
_registration_refresh_worker = create_registration_refresh_worker(
    registration_refresh=_runtime.refresh_platform_registration,
    status_provider=_runtime.status_provider,
)

WORKER_STOP_TIMEOUT_SECONDS = 60
logger = logging.getLogger("harnyx_validator.server")


async def _stop_runtime_components(
    *,
    platform_work_started: bool,
    weight_started: bool,
    registration_refresh_started: bool,
    auth_started: bool,
) -> None:
    if platform_work_started and _platform_work_worker is not None:
        try:
            await _platform_work_worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("failed stopping platform work worker during shutdown")
    if registration_refresh_started:
        try:
            _registration_refresh_worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("failed stopping registration refresh worker during shutdown")
    if weight_started:
        try:
            _weight_worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("failed stopping weight worker during shutdown")
    if auth_started:
        try:
            _runtime.inbound_auth_verifier.stop(timeout_seconds=WORKER_STOP_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("failed stopping inbound auth verifier during shutdown")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    auth_started = False
    weight_started = False
    registration_refresh_started = False
    platform_work_started = False
    try:
        _runtime.inbound_auth_verifier.start()
        auth_started = True
        _weight_worker.start()
        weight_started = True
        _registration_refresh_worker.start()
        registration_refresh_started = True
        if _platform_work_worker is not None:
            _platform_work_worker.start()
            platform_work_started = True
        yield
    finally:
        try:
            await _stop_runtime_components(
                platform_work_started=platform_work_started,
                weight_started=weight_started,
                registration_refresh_started=registration_refresh_started,
                auth_started=auth_started,
            )
        finally:
            try:
                await close_runtime_resources(_runtime)
            finally:
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
