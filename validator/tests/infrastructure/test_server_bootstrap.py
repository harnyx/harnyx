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
        ({7, 1, 5, 3}, "1,3,5,7"),
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


def test_sandbox_options_factory_resolves_one_shared_four_cpu_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    affinity_reads = 0

    def _sched_getaffinity(process_id: int) -> set[int]:
        nonlocal affinity_reads
        assert process_id == 0
        affinity_reads += 1
        return {10, 8, 6, 4, 2}

    monkeypatch.setattr(
        bootstrap_mod,
        "os",
        SimpleNamespace(sched_getaffinity=_sched_getaffinity),
        raising=False,
    )
    _capture_sandbox_options_build(monkeypatch)

    factory = bootstrap_mod._make_options_factory(_sandbox_settings())
    first_options = factory()
    second_options = factory()

    assert affinity_reads == 1
    assert _docker_argument_value(first_options, "--cpuset-cpus") == "2,4,6,8"
    assert _docker_argument_value(second_options, "--cpuset-cpus") == "2,4,6,8"
    assert first_options.extra_args[:-2] == CONTAINER_SECURITY.extra_args
    assert first_options.labels[_SANDBOX_CPUSET_LABEL] == "2,4,6,8"
    assert second_options.labels[_SANDBOX_CPUSET_LABEL] == "2,4,6,8"


def test_sandbox_options_factory_fails_when_process_affinity_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_affinity_error(process_id: int) -> set[int]:
        raise OSError(f"cannot read affinity for process {process_id}")

    monkeypatch.setattr(
        bootstrap_mod,
        "os",
        SimpleNamespace(sched_getaffinity=_raise_affinity_error),
        raising=False,
    )

    with pytest.raises(OSError, match="cannot read affinity"):
        bootstrap_mod._make_options_factory(_sandbox_settings())


def test_validator_import_configures_sentry_before_tracing(monkeypatch) -> None:
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
        weight_submission_service=object(),
        status_provider=object(),
        tool_route_deps_provider=lambda: object(),
        control_deps_provider=lambda: object(),
        platform_work_worker=None,
        register_with_platform=lambda: None,
        refresh_platform_registration=lambda: None,
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

    def _fake_create_weight_worker(*, submission_service: object, status_provider: object) -> object:
        assert submission_service is fake_runtime.weight_submission_service
        assert status_provider is fake_runtime.status_provider
        calls.append("weight_worker")
        return fake_worker

    def _fake_create_registration_refresh_worker(
        *,
        registration_refresh: object,
        status_provider: object,
    ) -> object:
        assert registration_refresh is fake_runtime.refresh_platform_registration
        assert status_provider is fake_runtime.status_provider
        calls.append("registration_worker")
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
    monkeypatch.setattr(weight_worker_mod, "create_weight_worker", _fake_create_weight_worker)
    monkeypatch.setattr(
        registration_worker_mod,
        "create_registration_refresh_worker",
        _fake_create_registration_refresh_worker,
    )
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


def _import_server_with_captured_weight_worker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
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
async def test_platform_work_worker_scoreable_execution_keeps_scoring_errors_validator_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_score_platform_execution(
        scoring_service: object,
        execution: object,
        *,
        convert_scoring_error: bool,
    ) -> object:
        observed["scoring_service"] = scoring_service
        observed["execution"] = execution
        observed["convert_scoring_error"] = convert_scoring_error
        return "result"

    monkeypatch.setattr(bootstrap_mod, "score_platform_execution", _fake_score_platform_execution)
    scoring_service = object()
    scoring_services = {
        entry.model: scoring_service if index == 0 else object()
        for index, entry in enumerate(bootstrap_mod._SCORING_SLOT_CONFIG.entries)
    }
    scoring_model = bootstrap_mod._SCORING_SLOT_CONFIG.entries[0].model
    execution = object()
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
    assert await worker._score_execution_by_model[scoring_model](execution) == "result"  # type: ignore[arg-type]
    assert observed == {
        "scoring_service": scoring_service,
        "execution": execution,
        "convert_scoring_error": True,
    }


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
