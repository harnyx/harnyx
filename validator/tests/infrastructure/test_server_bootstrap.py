from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import sentry_sdk
from fastapi import FastAPI

import harnyx_commons.observability.tracing as tracing_mod
import harnyx_validator.infrastructure.http.routes as routes_mod
import harnyx_validator.infrastructure.observability.logging as logging_mod
import harnyx_validator.infrastructure.observability.sentry as sentry_mod
import harnyx_validator.runtime.bootstrap as bootstrap_mod
import harnyx_validator.runtime.registration_worker as registration_worker_mod
import harnyx_validator.runtime.settings as settings_mod
import harnyx_validator.runtime.weight_worker as weight_worker_mod
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.runtime import CONTAINER_SECURITY
from harnyx_validator.application.dto.registration import ValidatorRegistrationMetadata

_SANDBOX_CPUSET_LABEL = "harnyx.sandbox.cpuset_cpus"


def _sandbox_settings() -> SimpleNamespace:
    return SimpleNamespace(
        sandbox=SimpleNamespace(
            sandbox_image="sandbox:test",
            sandbox_network="sandbox-network",
            sandbox_pull_policy="missing",
        ),
        rpc_port=8100,
    )


def _capture_sandbox_options_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def _build_sandbox_options(**kwargs: object) -> SandboxOptions:
        labels = kwargs.get("labels")
        assert isinstance(labels, dict)
        return SandboxOptions(
            image=str(kwargs["image"]),
            container_name=str(kwargs["container_name"]),
            extra_args=CONTAINER_SECURITY.extra_args,
            labels=labels,
        )

    monkeypatch.setattr(bootstrap_mod, "build_sandbox_options", _build_sandbox_options)


def _docker_argument_value(options: SandboxOptions, name: str) -> str:
    argument_index = options.extra_args.index(name)
    return options.extra_args[argument_index + 1]


@pytest.mark.parametrize(
    ("allowed_cpu_ids", "expected_cpuset"),
    [
        ({9}, "9"),
        ({7, 3, 11}, "3,7,11"),
        ({10, 8, 6, 4, 2}, "2,4,6,8"),
    ],
)
def test_sandbox_options_factory_uses_all_allowed_cpu_ids_up_to_four(
    monkeypatch: pytest.MonkeyPatch,
    allowed_cpu_ids: set[int],
    expected_cpuset: str,
) -> None:
    monkeypatch.setattr(
        bootstrap_mod,
        "os",
        SimpleNamespace(sched_getaffinity=lambda process_id: allowed_cpu_ids),
        raising=False,
    )
    _capture_sandbox_options_build(monkeypatch)

    options = bootstrap_mod._make_options_factory(_sandbox_settings())()

    assert _docker_argument_value(options, "--cpuset-cpus") == expected_cpuset
    assert _docker_argument_value(options, "--cpus") == "1"
    assert options.labels[_SANDBOX_CPUSET_LABEL] == expected_cpuset


def _import_server_with_captured_weight_worker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_real_sentry_startup: bool = False,
) -> list[dict[str, object]]:
    fake_settings = SimpleNamespace(
        observability=SimpleNamespace(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        platform_api=SimpleNamespace(),
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        platform_work_worker=None,
        register_with_platform=lambda: None,
        refresh_platform_registration=lambda: None,
    )
    fake_worker = SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None)
    captured: list[dict[str, object]] = []

    def _fake_create_weight_worker(**kwargs: object) -> object:
        captured.append(kwargs)
        return fake_worker

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(lambda cls: fake_settings))
    if not use_real_sentry_startup:
        monkeypatch.setattr(sentry_mod, "configure_sentry_from_env", lambda: None)
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda *, service_name: None)
    monkeypatch.setattr(logging_mod, "init_logging", lambda: None)
    monkeypatch.setattr(logging_mod, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(bootstrap_mod, "build_runtime", lambda settings: fake_runtime)
    monkeypatch.setattr(weight_worker_mod, "create_weight_worker", _fake_create_weight_worker)
    monkeypatch.setattr(
        registration_worker_mod,
        "create_registration_refresh_worker",
        lambda **kwargs: fake_worker,
    )
    monkeypatch.setattr(routes_mod, "add_tool_routes", lambda app, dependency_provider: None)
    monkeypatch.setattr(routes_mod, "add_control_routes", lambda app, control_deps_provider: None)

    module_name = "harnyx_validator.server"
    original_module = sys.modules.pop(module_name, None)
    try:
        importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
    return captured


def test_validator_startup_initializes_sentry_from_validator_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: Validator startup must initialize the active Sentry client from SENTRY_DSN."""
    dsn = "https://public@example.invalid/1"
    monkeypatch.setenv("SENTRY_DSN", dsn)
    try:
        _import_server_with_captured_weight_worker_kwargs(monkeypatch, use_real_sentry_startup=True)

        assert sentry_sdk.get_client().dsn == dsn
    finally:
        sentry_sdk.init(dsn=None)


def test_validator_startup_initializes_sentry_before_settings_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: startup faults must be reportable even when settings cannot load."""
    dsn = "https://public@example.invalid/1"
    monkeypatch.setenv("SENTRY_DSN", dsn)
    monkeypatch.setattr(logging_mod, "init_logging", lambda: None)

    def _fail_settings_load(cls: object) -> object:
        del cls
        raise RuntimeError("settings failed")

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(_fail_settings_load))
    module_name = "harnyx_validator.server"
    original_module = sys.modules.pop(module_name, None)
    try:
        with pytest.raises(RuntimeError, match="settings failed"):
            importlib.import_module(module_name)

        assert sentry_sdk.get_client().dsn == dsn
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        sentry_sdk.init(dsn=None)


def test_validator_import_ignores_smoke_weight_worker_interval_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALIDATOR_COMPOSE_SMOKE", raising=False)
    monkeypatch.setenv("VALIDATOR_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_SECONDS", "5")

    captured = _import_server_with_captured_weight_worker_kwargs(monkeypatch)

    assert len(captured) == 1
    assert set(captured[0]) == {"submission_service", "status_provider"}
    assert "poll_interval_seconds" not in captured[0]


def test_validator_import_uses_compose_smoke_weight_worker_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_COMPOSE_SMOKE", "1")
    monkeypatch.setenv("VALIDATOR_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_SECONDS", "5")

    captured = _import_server_with_captured_weight_worker_kwargs(monkeypatch)

    assert captured[0]["poll_interval_seconds"] == 5.0


def test_validator_import_rejects_invalid_compose_smoke_weight_worker_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_COMPOSE_SMOKE", "1")
    monkeypatch.setenv("VALIDATOR_SMOKE_WEIGHT_WORKER_POLL_INTERVAL_SECONDS", "0")

    with pytest.raises(RuntimeError, match="smoke weight-worker poll interval must be positive"):
        _import_server_with_captured_weight_worker_kwargs(monkeypatch)


def test_validator_runtime_separates_startup_registration_from_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = ValidatorRegistrationMetadata(
        validator_version="test-version",
        source_revision=None,
        registry_digest=None,
        local_image_id=None,
    )
    platform_api = SimpleNamespace(validator_public_base_url="https://validator.invalid")
    fake_context = SimpleNamespace(
        settings=SimpleNamespace(platform_api=platform_api),
        platform_hotkey=object(),
        registration_metadata=metadata,
    )
    calls: list[dict[str, object]] = []

    def _fake_register_with_platform(
        settings: object,
        hotkey: object,
        public_url: str | None,
        *,
        metadata: ValidatorRegistrationMetadata,
        attempts: int,
        delay_seconds: float,
    ) -> None:
        calls.append(
            {
                "settings": settings,
                "hotkey": hotkey,
                "public_url": public_url,
                "metadata": metadata,
                "attempts": attempts,
                "delay_seconds": delay_seconds,
            }
        )

    monkeypatch.setattr(bootstrap_mod, "_register_with_platform", _fake_register_with_platform)

    bootstrap_mod.RuntimeContext.register_with_platform(fake_context)  # type: ignore[arg-type]
    bootstrap_mod.RuntimeContext.refresh_platform_registration(fake_context)  # type: ignore[arg-type]

    assert calls == [
        {
            "settings": fake_context.settings,
            "hotkey": fake_context.platform_hotkey,
            "public_url": "https://validator.invalid",
            "metadata": metadata,
            "attempts": 30,
            "delay_seconds": 2.0,
        },
        {
            "settings": fake_context.settings,
            "hotkey": fake_context.platform_hotkey,
            "public_url": "https://validator.invalid",
            "metadata": metadata,
            "attempts": 1,
            "delay_seconds": 0.0,
        },
    ]


def test_platform_work_worker_uses_task_capacity_and_artifact_cap() -> None:
    scoring_services = {entry.model: object() for entry in bootstrap_mod._SCORING_SLOT_CONFIG.entries}
    worker = bootstrap_mod._build_platform_work_worker(
        resolved=SimpleNamespace(),
        platform_client=object(),  # type: ignore[arg-type]
        subtensor_client=object(),  # type: ignore[arg-type]
        sandbox_manager=object(),  # type: ignore[arg-type]
        state=SimpleNamespace(
            session_manager=object(),
            evaluation_records=object(),
            receipt_log=object(),
            progress_tracker=object(),
            batch_activity=object(),
            platform_tool_proxy_scopes=object(),
        ),
        batch_blocking_executor=object(),  # type: ignore[arg-type]
        scoring_services=scoring_services,  # type: ignore[arg-type]
        orchestrator_factory=lambda _client: object(),  # type: ignore[arg-type]
        options_factory=lambda: object(),  # type: ignore[arg-type]
    )

    assert worker is not None
    assert worker._target_concurrency == 20
    assert worker._max_active_artifacts == 4
    assert worker._scoring_slot_config is bootstrap_mod._SCORING_SLOT_CONFIG
    assert worker._scoring_slot_config.total_slot_limit == 20
    assert tuple(entry.slot_limit for entry in worker._scoring_slot_config.entries) == (10, 10)
    assert set(worker._score_execution_by_model) == set(scoring_services)
    assert worker._target_concurrency > worker._max_active_artifacts


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
        platform_api=SimpleNamespace(),
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        inbound_auth_verifier=object(),
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        platform_work_worker=None,
        register_with_platform=lambda: None,
        refresh_platform_registration=lambda: None,
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

    class _FakeRegistrationRefreshWorker:
        def start(self) -> None:
            calls.append("registration-start")

        def stop(self, *, timeout: float) -> None:
            calls.append(f"registration-stop:{timeout}")

    class _FailingPlatformWorkWorker:
        def start(self) -> None:
            calls.append("platform-work-start")
            raise RuntimeError("platform work startup failed")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"platform-work-stop:{timeout}")

    async def _fake_close_runtime_resources(runtime: object) -> None:
        calls.append("close-runtime")

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(lambda cls: fake_settings))
    monkeypatch.setattr(bootstrap_mod, "build_runtime", lambda settings: fake_runtime)
    monkeypatch.setattr(
        weight_worker_mod,
        "create_weight_worker",
        lambda **kwargs: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        registration_worker_mod,
        "create_registration_refresh_worker",
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
    monkeypatch.setattr(server, "_registration_refresh_worker", _FakeRegistrationRefreshWorker())
    monkeypatch.setattr(server, "_platform_work_worker", _FailingPlatformWorkWorker())
    monkeypatch.setattr(server, "close_runtime_resources", _fake_close_runtime_resources)
    monkeypatch.setattr(server, "shutdown_logging", lambda: calls.append("shutdown-logging"))

    with pytest.raises(RuntimeError, match="platform work startup failed"):
        async with server.lifespan(FastAPI()):
            raise AssertionError("lifespan should not yield after startup failure")

    assert calls == [
        "auth-start",
        "weight-start",
        "registration-start",
        "platform-work-start",
        f"registration-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
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
        platform_api=SimpleNamespace(),
    )
    fake_runtime = SimpleNamespace(
        settings=fake_settings,
        inbound_auth_verifier=object(),
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        platform_work_worker=None,
        register_with_platform=lambda: None,
        refresh_platform_registration=lambda: None,
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

    class _FakeRegistrationRefreshWorker:
        def start(self) -> None:
            calls.append("registration-start")

        def stop(self, *, timeout: float) -> None:
            calls.append(f"registration-stop:{timeout}")

    class _FakePlatformWorkWorker:
        def start(self) -> None:
            calls.append("platform-work-start")

        async def stop(self, *, timeout: float) -> None:
            calls.append(f"platform-work-stop:{timeout}")

    async def _fake_close_runtime_resources(runtime: object) -> None:
        calls.append("close-runtime")

    monkeypatch.setattr(settings_mod.Settings, "load", classmethod(lambda cls: fake_settings))
    monkeypatch.setattr(bootstrap_mod, "build_runtime", lambda settings: fake_runtime)
    monkeypatch.setattr(
        weight_worker_mod,
        "create_weight_worker",
        lambda **kwargs: SimpleNamespace(start=lambda: None, stop=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        registration_worker_mod,
        "create_registration_refresh_worker",
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
    monkeypatch.setattr(server, "_registration_refresh_worker", _FakeRegistrationRefreshWorker())
    monkeypatch.setattr(server, "_platform_work_worker", _FakePlatformWorkWorker())
    monkeypatch.setattr(server, "close_runtime_resources", _fake_close_runtime_resources)
    monkeypatch.setattr(server, "shutdown_logging", lambda: calls.append("shutdown-logging"))

    async with server.lifespan(FastAPI()):
        calls.append("yielded")

    assert calls == [
        "auth-start",
        "weight-start",
        "registration-start",
        "platform-work-start",
        "yielded",
        f"platform-work-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"registration-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"weight-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        f"auth-stop:{server.WORKER_STOP_TIMEOUT_SECONDS}",
        "close-runtime",
        "shutdown-logging",
    ]
