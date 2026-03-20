from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

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
