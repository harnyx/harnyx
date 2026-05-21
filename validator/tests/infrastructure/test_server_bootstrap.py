from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import harnyx_commons.observability.tracing as tracing_mod
import harnyx_validator.infrastructure.http.routes as routes_mod
import harnyx_validator.infrastructure.observability.logging as logging_mod
import harnyx_validator.infrastructure.observability.sentry as sentry_mod
import harnyx_validator.runtime.bootstrap as bootstrap_mod
import harnyx_validator.runtime.evaluation_worker as evaluation_worker_mod
import harnyx_validator.runtime.settings as settings_mod
import harnyx_validator.runtime.weight_worker as weight_worker_mod


def test_validator_import_configures_sentry_before_tracing(monkeypatch) -> None:
    calls: list[str] = []
    fake_settings = SimpleNamespace(
        observability=SimpleNamespace(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        restore_worker=object(),
        register_with_platform=lambda: None,
    )
    fake_worker = SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None)

    def _fake_settings_load(cls) -> SimpleNamespace:
        calls.append("settings")
        return fake_settings

    def _fake_configure_sentry() -> None:
        calls.append("sentry")

    def _fake_configure_tracing(*, service_name: str) -> None:
        assert service_name == "harnyx-validator"
        calls.append("tracing")

    def _fake_init_logging() -> None:
        calls.append("logging")

    def _fake_configure_logging(
        *,
        cloud_logging_enabled: bool,
        gcp_project: str | None,
        cloud_log_labels: dict[str, str] | None,
    ) -> None:
        assert cloud_logging_enabled is False
        assert gcp_project is None
        assert cloud_log_labels is None
        calls.append("configure_logging")

    def _fake_build_runtime(settings: object) -> SimpleNamespace:
        assert settings is fake_settings
        calls.append("build_runtime")
        return fake_runtime

    def _fake_create_evaluation_worker_from_context(context: object) -> object:
        assert context is fake_runtime
        calls.append("evaluation_worker")
        return fake_worker

    def _fake_create_weight_worker(*, submission_service: object, status_provider: object) -> object:
        assert submission_service is fake_runtime.weight_submission_service
        assert status_provider is fake_runtime.status_provider
        calls.append("weight_worker")
        return fake_worker

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(_fake_settings_load))
    monkeypatch.setattr(
        sentry_mod,
        "configure_sentry_from_env",
        _fake_configure_sentry,
    )
    monkeypatch.setattr(tracing_mod, "configure_tracing", _fake_configure_tracing)
    monkeypatch.setattr(logging_mod, "init_logging", _fake_init_logging)
    monkeypatch.setattr(logging_mod, "configure_logging", _fake_configure_logging)
    monkeypatch.setattr(bootstrap_mod, "build_runtime", _fake_build_runtime)
    monkeypatch.setattr(
        evaluation_worker_mod,
        "create_evaluation_worker_from_context",
        _fake_create_evaluation_worker_from_context,
    )
    monkeypatch.setattr(weight_worker_mod, "create_weight_worker", _fake_create_weight_worker)
    monkeypatch.setattr(routes_mod, "add_tool_routes", lambda app, dependency_provider: None)
    monkeypatch.setattr(routes_mod, "add_control_routes", lambda app, control_deps_provider: None)

    module_name = "harnyx_validator.server"
    original_module = sys.modules.pop(module_name, None)
    try:
        imported = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    assert imported._settings is fake_settings
    assert calls[:6] == [
        "logging",
        "sentry",
        "settings",
        "tracing",
        "configure_logging",
        "build_runtime",
    ]
    assert calls.index("sentry") < calls.index("tracing")


def test_validator_logging_config_defaults_measurement_logger_to_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALIDATOR_MEASUREMENT_LOG_LEVEL", raising=False)

    config = logging_mod.build_log_config()

    assert config["loggers"]["harnyx_validator.measurement"]["level"] == "WARNING"


def test_validator_logging_config_respects_measurement_logger_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_MEASUREMENT_LOG_LEVEL", "debug")

    config = logging_mod.build_log_config()

    assert config["loggers"]["harnyx_validator.measurement"]["level"] == "DEBUG"


@pytest.mark.anyio
async def test_lifespan_stops_auth_when_later_startup_step_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_settings = SimpleNamespace(
        observability=SimpleNamespace(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        inbound_auth_verifier=object(),
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        restore_worker=object(),
        register_with_platform=lambda: None,
    )

    class _FakeVerifier:
        def start(self) -> None:
            calls.append("auth-start")

        def stop(self, *, timeout_seconds: float) -> None:
            calls.append(f"auth-stop:{timeout_seconds}")

    class _FakeWeightWorker:
        def start(self) -> None:
            calls.append("weight-start")

        def stop(self, *, timeout: float) -> None:
            calls.append(f"weight-stop:{timeout}")

    class _FakeRestoreWorker:
        def start(self) -> None:
            calls.append("restore-start")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"restore-stop:{timeout}")

    class _FailingEvaluationWorker:
        def start(self) -> None:
            calls.append("eval-start")
            raise RuntimeError("evaluation startup failed")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"eval-stop:{timeout}")

    async def _fake_close_runtime_resources(runtime: object) -> None:
        calls.append("close-runtime")

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(lambda cls: fake_settings))
    monkeypatch.setattr(bootstrap_mod, "build_runtime", lambda settings: fake_runtime)
    monkeypatch.setattr(
        evaluation_worker_mod,
        "create_evaluation_worker_from_context",
        lambda context: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        weight_worker_mod,
        "create_weight_worker",
        lambda **kwargs: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(routes_mod, "add_tool_routes", lambda app, dependency_provider: None)
    monkeypatch.setattr(routes_mod, "add_control_routes", lambda app, control_deps_provider: None)

    module_name = "harnyx_validator.server"
    original_module = sys.modules.pop(module_name, None)
    try:
        server = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    monkeypatch.setattr(server, "_runtime", SimpleNamespace(inbound_auth_verifier=_FakeVerifier()))
    monkeypatch.setattr(server, "_weight_worker", _FakeWeightWorker())
    monkeypatch.setattr(server, "_restore_worker", _FakeRestoreWorker())
    monkeypatch.setattr(server, "_evaluation_worker", _FailingEvaluationWorker())
    monkeypatch.setattr(server, "close_runtime_resources", _fake_close_runtime_resources)
    monkeypatch.setattr(server, "shutdown_logging", lambda: calls.append("shutdown-logging"))

    with pytest.raises(RuntimeError, match="evaluation startup failed"):
        async with server.lifespan(FastAPI()):
            raise AssertionError("lifespan should not yield after startup failure")

    assert calls == [
        "auth-start",
        "weight-start",
        "restore-start",
        "eval-start",
        f"restore-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"weight-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"auth-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        "close-runtime",
        "shutdown-logging",
    ]


@pytest.mark.anyio
async def test_lifespan_closes_runtime_resources_when_auth_stop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_settings = SimpleNamespace(
        observability=SimpleNamespace(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        inbound_auth_verifier=object(),
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        restore_worker=object(),
        register_with_platform=lambda: None,
    )

    class _FakeVerifier:
        def start(self) -> None:
            calls.append("auth-start")

        def stop(self, *, timeout_seconds: float) -> bool:
            calls.append(f"auth-stop:{timeout_seconds}")
            raise RuntimeError("auth stop hung")

    class _FakeWeightWorker:
        def start(self) -> None:
            calls.append("weight-start")

        def stop(self, *, timeout: float) -> None:
            calls.append(f"weight-stop:{timeout}")

    class _FakeRestoreWorker:
        def start(self) -> None:
            calls.append("restore-start")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"restore-stop:{timeout}")

    class _FakeEvaluationWorker:
        def start(self) -> None:
            calls.append("eval-start")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"eval-stop:{timeout}")

    async def _fake_close_runtime_resources(runtime: object) -> None:
        calls.append("close-runtime")

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(lambda cls: fake_settings))
    monkeypatch.setattr(bootstrap_mod, "build_runtime", lambda settings: fake_runtime)
    monkeypatch.setattr(
        evaluation_worker_mod,
        "create_evaluation_worker_from_context",
        lambda context: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        weight_worker_mod,
        "create_weight_worker",
        lambda **kwargs: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(routes_mod, "add_tool_routes", lambda app, dependency_provider: None)
    monkeypatch.setattr(routes_mod, "add_control_routes", lambda app, control_deps_provider: None)

    module_name = "harnyx_validator.server"
    original_module = sys.modules.pop(module_name, None)
    try:
        server = importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    monkeypatch.setattr(server, "_runtime", SimpleNamespace(inbound_auth_verifier=_FakeVerifier()))
    monkeypatch.setattr(server, "_weight_worker", _FakeWeightWorker())
    monkeypatch.setattr(server, "_restore_worker", _FakeRestoreWorker())
    monkeypatch.setattr(server, "_evaluation_worker", _FakeEvaluationWorker())
    monkeypatch.setattr(server, "close_runtime_resources", _fake_close_runtime_resources)
    monkeypatch.setattr(server, "shutdown_logging", lambda: calls.append("shutdown-logging"))

    async with server.lifespan(FastAPI()):
        calls.append("yielded")

    assert calls == [
        "auth-start",
        "weight-start",
        "restore-start",
        "eval-start",
        "yielded",
        f"restore-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"eval-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"weight-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"auth-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        "close-runtime",
        "shutdown-logging",
    ]
