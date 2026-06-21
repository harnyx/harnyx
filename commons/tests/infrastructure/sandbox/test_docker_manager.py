from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from subprocess import CompletedProcess

import httpx
import pytest

import harnyx_commons.sandbox.docker as docker_module
from harnyx_commons.sandbox.docker import (
    DockerSandboxManager,
    HttpSandboxClient,
    SandboxOptions,
    resolve_sandbox_host_container_url,
)
from harnyx_commons.sandbox.manager import SandboxDeployment

_HOST_CONTAINER_URL = "http://127.0.0.1:1"


@dataclass
class DummyClient:
    base_url: str
    host_container_url: str | None
    closed: bool = False

    def invoke(self, *args, **kwargs):  # pragma: no cover - not used in test
        raise NotImplementedError

    def close(self) -> None:
        self.closed = True


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object):
        self.commands.append((list(args), dict(kwargs)))
        stdout = ""
        if args[:2] == ["docker", "run"] and "-d" in args:
            stdout = "container123\n"
        elif args[:2] == ["docker", "inspect"]:
            stdout = inspect_container_stdout(ip_address="172.18.0.2")
        return subprocess_completed(args, stdout)


def subprocess_completed(args: list[str], stdout: str) -> CompletedProcess[str]:
    return CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def inspect_container_stdout(
    *,
    ip_address: str = "172.19.0.2",
    network: str = "harnyx-net",
    status: str = "running",
    exit_code: int = 0,
    error: str = "",
) -> str:
    return (
        json.dumps(
            {
                "State": {
                    "Status": status,
                    "ExitCode": exit_code,
                    "Error": error,
                },
                "NetworkSettings": {
                    "Networks": {
                        network: {
                            "IPAddress": ip_address,
                        }
                    },
                },
            }
        )
        + "\n"
    )


def test_http_sandbox_client_default_timeout_exceeds_entrypoint_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, *, base_url: str, timeout: float, limits: httpx.Limits) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            captured["limits"] = limits

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(docker_module.httpx, "AsyncClient", FakeAsyncClient)

    client = HttpSandboxClient("http://sandbox")
    try:
        assert captured == {
            "base_url": "http://sandbox",
            "timeout": 310.0,
            "limits": httpx.Limits(max_keepalive_connections=0),
        }
    finally:
        client.close()


def test_docker_sandbox_manager_builds_commands(monkeypatch) -> None:
    runner = RecordingRunner()
    created_clients: list[DummyClient] = []

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        client = DummyClient(base_url, host_container_url)
        created_clients.append(client)
        return client

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=9000,
        container_port=8000,
        env={"EXAMPLE": "value"},
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    assert isinstance(deployment, SandboxDeployment)
    assert deployment.identifier == "container123"
    assert deployment.base_url == "http://127.0.0.1:9000"
    assert isinstance(deployment.client, DummyClient)
    assert deployment.client.host_container_url == options.host_container_url

    run_args, run_kwargs = runner.commands[0]
    assert run_args[:4] == [
        "docker",
        "run",
        "--pull",
        options.pull_policy,
    ]
    assert run_args[4:9] == [
        "-d",
        "--name",
        "sandbox-demo",
        "-p",
        "9000:8000",
    ]
    assert "--network" in run_args
    assert "-e" in run_args
    assert run_args[-1] == options.image
    assert run_kwargs["capture_output"] is True
    assert run_kwargs["text"] is True

    manager.stop(deployment)
    stop_args, stop_kwargs = runner.commands[1]
    assert stop_args == ["docker", "stop", "-t", "5", "container123"]
    rm_args, rm_kwargs = runner.commands[2]
    assert rm_args == ["docker", "rm", "-f", "container123"]
    assert deployment.client.closed is True
    assert created_clients[0].closed is True
    assert stop_kwargs["capture_output"] is True
    assert rm_kwargs["capture_output"] is True


def test_pull_policy_always_retries_docker_pull_before_local_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module.time, "sleep", lambda _seconds: None)
    commands: list[list[str]] = []
    pull_attempts = 0

    def command_runner(args: list[str], **kwargs: object):
        nonlocal pull_attempts
        del kwargs
        commands.append(list(args))
        if args == ["docker", "pull", "harnyx/sandbox:demo"]:
            pull_attempts += 1
            if pull_attempts == 1:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=args,
                    stderr="lookup auth.docker.io: i/o timeout",
                )
            return subprocess_completed(args, "")
        if args[:2] == ["docker", "run"]:
            return subprocess_completed(args, "container123\n")
        raise AssertionError(f"unexpected command: {args}")

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
    )
    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="always",
        host_port=9000,
        container_port=8000,
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)

    assert deployment.identifier == "container123"
    assert commands[:2] == [
        ["docker", "pull", "harnyx/sandbox:demo"],
        ["docker", "pull", "harnyx/sandbox:demo"],
    ]
    run_args = commands[2]
    assert run_args[:4] == ["docker", "run", "--pull", "never"]
    assert "--rm" not in run_args


def test_docker_sandbox_manager_adds_labels_to_docker_run() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=9000,
        container_port=8000,
        labels={"b": "two", "a": "one"},
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]

    name_index = run_args.index("--name")
    assert run_args[name_index : name_index + 8] == [
        "--name",
        "sandbox-demo",
        "--label",
        "a=one",
        "--label",
        "b=two",
        "-p",
        "9000:8000",
    ]
    manager.stop(deployment)


def test_docker_sandbox_manager_removes_labeled_and_non_running_legacy_prefixed_containers() -> None:
    commands: list[list[str]] = []

    def command_runner(args: list[str], **kwargs: object):
        commands.append(list(args))
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        if args == [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=harnyx.sandbox.managed=true",
            "--filter",
            "label=harnyx.sandbox.owner=validator",
        ]:
            return subprocess_completed(args, "new-labeled\n")
        if args == [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "name=^/harnyx-sandbox-",
            "--filter",
            "status=created",
        ]:
            return subprocess_completed(args, "old-created\n")
        if args == [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "name=^/harnyx-sandbox-",
            "--filter",
            "status=exited",
        ]:
            return subprocess_completed(args, "old-exited\n")
        if args == [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "name=^/harnyx-sandbox-",
            "--filter",
            "status=dead",
        ]:
            return subprocess_completed(args, "old-dead\nnew-labeled\n")
        if args == ["docker", "rm", "-f", "new-labeled", "old-created", "old-dead", "old-exited"]:
            return subprocess_completed(args, "")
        raise AssertionError(f"unexpected command: {args}")

    manager = DockerSandboxManager(docker_binary="docker", command_runner=command_runner)

    manager.cleanup_stale_sandbox_containers(
        labels={"harnyx.sandbox.managed": "true", "harnyx.sandbox.owner": "validator"},
        name_prefix="harnyx-sandbox-",
    )

    assert commands == [
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=harnyx.sandbox.managed=true",
            "--filter",
            "label=harnyx.sandbox.owner=validator",
        ],
        ["docker", "ps", "-aq", "--filter", "name=^/harnyx-sandbox-", "--filter", "status=created"],
        ["docker", "ps", "-aq", "--filter", "name=^/harnyx-sandbox-", "--filter", "status=exited"],
        ["docker", "ps", "-aq", "--filter", "name=^/harnyx-sandbox-", "--filter", "status=dead"],
        ["docker", "rm", "-f", "new-labeled", "old-created", "old-dead", "old-exited"],
    ]


def test_docker_sandbox_manager_logs_and_continues_when_stale_list_fails(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_commons.sandbox.docker")

    def command_runner(args: list[str], **kwargs: object):
        raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="docker unavailable")

    manager = DockerSandboxManager(docker_binary="docker", command_runner=command_runner)

    manager.cleanup_stale_sandbox_containers(
        labels={"harnyx.sandbox.managed": "true", "harnyx.sandbox.owner": "validator"},
        name_prefix="harnyx-sandbox-",
    )

    assert "failed to remove stale sandbox containers" in caplog.text


def test_docker_sandbox_manager_logs_and_continues_when_stale_cleanup_fails(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="harnyx_commons.sandbox.docker")

    def command_runner(args: list[str], **kwargs: object):
        if args[1:3] == ["ps", "-aq"]:
            return subprocess_completed(args, "stale-container\n")
        raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="remove failed")

    manager = DockerSandboxManager(docker_binary="docker", command_runner=command_runner)

    manager.cleanup_stale_sandbox_containers(
        labels={"harnyx.sandbox.managed": "true", "harnyx.sandbox.owner": "validator"},
        name_prefix="harnyx-sandbox-",
    )

    assert "failed to remove stale sandbox containers" in caplog.text


def test_docker_sandbox_manager_binds_published_port_when_configured() -> None:
    runner = RecordingRunner()
    created_clients: list[DummyClient] = []

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        client = DummyClient(base_url, host_container_url)
        created_clients.append(client)
        return client

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        published_port_bind_host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=9000,
        container_port=8000,
        env={"EXAMPLE": "value"},
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)

    run_args, _ = runner.commands[0]
    assert run_args[4:9] == [
        "-d",
        "--name",
        "sandbox-demo",
        "-p",
        "127.0.0.1:9000:8000",
    ]
    assert deployment.base_url == "http://127.0.0.1:9000"
    assert created_clients[0].base_url == "http://127.0.0.1:9000"


def test_docker_sandbox_manager_does_not_use_probe_host_as_bind_host() -> None:
    runner = RecordingRunner()
    created_clients: list[DummyClient] = []

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        client = DummyClient(base_url, host_container_url)
        created_clients.append(client)
        return client

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="host.docker.internal",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=9000,
        container_port=8000,
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)

    run_args, _ = runner.commands[0]
    assert run_args[8] == "9000:8000"
    assert "host.docker.internal:9000:8000" not in run_args
    assert deployment.base_url == "http://host.docker.internal:9000"
    assert created_clients[0].base_url == "http://host.docker.internal:9000"


def test_docker_manager_skips_port_mapping_when_host_port_missing() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=None,
        container_port=8000,
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "-p" not in run_args
    assert "--network" in run_args
    assert deployment.base_url == "http://172.18.0.2:8000"
    manager.stop(deployment)


class SequencedInspectRunner:
    def __init__(self, inspect_stdout: list[str]) -> None:
        self._inspect_stdout = list(inspect_stdout)
        self.commands: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object):
        self.commands.append((list(args), dict(kwargs)))
        if args[:2] == ["docker", "run"] and "-d" in args:
            return subprocess_completed(args, "container123\n")
        if args[:2] == ["docker", "inspect"]:
            assert args[-1] == "container123"
            if not self._inspect_stdout:
                raise AssertionError("unexpected docker inspect call")
            return subprocess_completed(args, self._inspect_stdout.pop(0))
        if args[:2] == ["docker", "stop"]:
            return subprocess_completed(args, "")
        if args[:2] == ["docker", "rm"]:
            return subprocess_completed(args, "")
        raise AssertionError(f"unexpected command: {args}")

    @property
    def inspect_commands(self) -> list[tuple[list[str], dict[str, object]]]:
        return [(args, kwargs) for args, kwargs in self.commands if args[:2] == ["docker", "inspect"]]


def _internal_network_options() -> SandboxOptions:
    return SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=None,
        container_port=8000,
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )


def test_docker_manager_waits_for_container_ip_before_internal_network_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module.time, "sleep", lambda _seconds: None)
    runner = SequencedInspectRunner(
        [
            inspect_container_stdout(ip_address=""),
            inspect_container_stdout(ip_address="172.19.0.2"),
        ]
    )

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(command_runner=runner, client_factory=client_factory)

    deployment = manager.start(_internal_network_options())

    assert deployment.base_url == "http://172.19.0.2:8000"
    assert len(runner.inspect_commands) == 2


def test_docker_manager_waits_when_configured_network_is_not_attached_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module.time, "sleep", lambda _seconds: None)
    runner = SequencedInspectRunner(
        [
            inspect_container_stdout(network="other-net", ip_address="172.18.0.2"),
            inspect_container_stdout(ip_address="172.19.0.2"),
        ]
    )

    manager = DockerSandboxManager(
        command_runner=runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
    )

    deployment = manager.start(_internal_network_options())

    assert deployment.base_url == "http://172.19.0.2:8000"
    assert len(runner.inspect_commands) == 2


def test_docker_manager_network_readiness_uses_container_id_and_classifies_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module, "_CONTAINER_IP_READY_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(docker_module, "_CONTAINER_IP_READY_POLL_INTERVAL_SECONDS", 0.001)
    runner = SequencedInspectRunner(
        [inspect_container_stdout(status="exited", exit_code=137, error="startup failed")]
    )
    manager = DockerSandboxManager(
        command_runner=runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
    )

    with pytest.raises(RuntimeError, match="sandbox container exited before readiness"):
        manager.start(_internal_network_options())

    inspect_commands = [args for args, _ in runner.inspect_commands]
    assert inspect_commands == [
        ["docker", "inspect", "--format", "{{json .}}", "container123"],
    ]
    assert any(args == ["docker", "rm", "-f", "container123"] for args, _ in runner.commands)


def test_docker_manager_network_readiness_inspect_command_failure_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module.time, "sleep", lambda _seconds: None)
    commands: list[list[str]] = []

    def command_runner(args: list[str], **kwargs: object):
        del kwargs
        commands.append(list(args))
        if args[:2] == ["docker", "run"]:
            return subprocess_completed(args, "container123\n")
        if args[:2] == ["docker", "inspect"]:
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="No such object")
        if args[:2] in (["docker", "stop"], ["docker", "rm"]):
            return subprocess_completed(args, "")
        raise AssertionError(f"unexpected command: {args}")

    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
    )

    with pytest.raises(RuntimeError, match="docker inspect failed while resolving sandbox network"):
        manager.start(_internal_network_options())

    inspect_commands = [args for args in commands if args[:2] == ["docker", "inspect"]]
    assert inspect_commands == [
        ["docker", "inspect", "--format", "{{json .}}", "container123"],
    ]


def test_docker_manager_bounds_each_container_ip_inspect_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module.time, "sleep", lambda _seconds: None)
    inspect_timeouts: list[float] = []
    calls: list[list[str]] = []

    def command_runner(args: list[str], **kwargs: object):
        calls.append(list(args))
        if args[:2] == ["docker", "run"] and "-d" in args:
            return subprocess_completed(args, "container123\n")
        if args[:2] == ["docker", "inspect"]:
            inspect_timeouts.append(float(kwargs["timeout"]))
            if len(inspect_timeouts) == 1:
                raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])
            return subprocess_completed(args, inspect_container_stdout(ip_address="172.19.0.2"))
        raise AssertionError(f"unexpected command: {args}")

    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
        command_timeout_seconds=120.0,
    )

    deployment = manager.start(_internal_network_options())

    assert deployment.base_url == "http://172.19.0.2:8000"
    assert len(inspect_timeouts) == 2
    assert inspect_timeouts[0] <= docker_module._CONTAINER_IP_READY_TIMEOUT_SECONDS
    assert calls[0][:2] == ["docker", "run"]


def test_docker_manager_cleans_up_when_container_ip_never_becomes_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_module, "_CONTAINER_IP_READY_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(docker_module, "_CONTAINER_IP_READY_POLL_INTERVAL_SECONDS", 0.001)
    runner = SequencedInspectRunner([inspect_container_stdout(ip_address="")] * 100)
    manager = DockerSandboxManager(
        command_runner=runner,
        client_factory=lambda base_url, host_container_url: DummyClient(base_url, host_container_url),
    )

    with pytest.raises(RuntimeError, match="invalid IP address for network: harnyx-net"):
        manager.start(_internal_network_options())

    assert any(args == ["docker", "stop", "-t", "5", "container123"] for args, _ in runner.commands)
    assert any(args == ["docker", "rm", "-f", "container123"] for args, _ in runner.commands)


def test_docker_manager_mounts_volumes() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        volumes=(("/host/agent.py", "/workspace/agent.py", "ro"),),
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "-v" in run_args
    volume_arg_index = run_args.index("-v") + 1
    assert run_args[volume_arg_index] == "/host/agent.py:/workspace/agent.py:ro"
    manager.stop(deployment)


def test_docker_manager_requires_network_when_host_port_missing() -> None:
    manager = DockerSandboxManager()
    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=None,
        host_container_url=_HOST_CONTAINER_URL,
    )
    with pytest.raises(ValueError):
        manager.start(options)


def test_docker_manager_adds_extra_hosts() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        extra_hosts=(("host.docker.internal", "host-gateway"),),
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "--add-host" in run_args
    host_arg_index = run_args.index("--add-host") + 1
    assert run_args[host_arg_index] == "host.docker.internal:host-gateway"
    manager.stop(deployment)


def test_docker_manager_sets_seccomp_profile() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        return DummyClient(base_url, host_container_url)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    seccomp_path = "/workspace/runtime-seccomp.json"
    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        seccomp_profile=seccomp_path,
        host_container_url=_HOST_CONTAINER_URL,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "--security-opt" in run_args
    opt_index = run_args.index("--security-opt") + 1
    assert run_args[opt_index] == f"seccomp={seccomp_path}"
    manager.stop(deployment)


def test_start_cleans_up_container_on_healthz_failure(monkeypatch) -> None:
    runner = RecordingRunner()
    created_clients: list[DummyClient] = []

    def client_factory(base_url: str, host_container_url: str | None) -> DummyClient:
        client = DummyClient(base_url, host_container_url)
        created_clients.append(client)
        return client

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    def fail_healthz(*args, **kwargs) -> None:
        raise RuntimeError("healthz timeout")

    monkeypatch.setattr(manager, "_wait_for_healthz", fail_healthz)

    options = SandboxOptions(
        image="harnyx/sandbox:demo",
        container_name="sandbox-demo",
        pull_policy="missing",
        host_port=9000,
        container_port=8000,
        wait_for_healthz=True,
        network="harnyx-net",
        host_container_url=_HOST_CONTAINER_URL,
    )

    with pytest.raises(RuntimeError, match="healthz timeout"):
        manager.start(options)

    run_args, _ = runner.commands[0]
    assert run_args[4:7] == ["-d", "--name", "sandbox-demo"]
    stop_args, _ = runner.commands[1]
    assert stop_args == ["docker", "stop", "-t", "5", "container123"]
    rm_args, _ = runner.commands[2]
    assert rm_args == ["docker", "rm", "-f", "container123"]
    assert created_clients[0].closed is True


def test_resolve_sandbox_host_container_url_falls_back_to_mountinfo_container_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_hostname = "6aadabeb0b48"
    live_container_id = "7ffe3b2775de"
    calls: list[str] = []

    def fake_exists(self) -> bool:
        return str(self) == "/.dockerenv"

    def fake_read_text(self, *, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        if str(self) != "/proc/self/mountinfo":
            raise AssertionError(f"unexpected read_text path: {self}")
        return (
            "1533 1522 8:1 "
            f"/var/lib/docker/containers/{live_container_id}/hostname "
            "/etc/hostname rw,relatime - ext4 /dev/sda1 rw,commit=30\n"
        )

    def fake_run(args: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(args[-1])
        if args[-1] == stale_hostname:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=args,
                stderr=f"error: no such object: {stale_hostname}",
            )
        if args[-1] == live_container_id:
            return subprocess_completed(args, '{"harnyx-net":{"IPAddress":"172.19.0.2"}}\n')
        raise AssertionError(f"unexpected docker target: {args[-1]}")

    monkeypatch.setenv("HOSTNAME", stale_hostname)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(docker_module.Path, "exists", fake_exists)
    monkeypatch.setattr(docker_module.Path, "read_text", fake_read_text)
    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)

    result = resolve_sandbox_host_container_url(
        docker_binary="docker",
        sandbox_network="harnyx-net",
        rpc_port=8100,
    )

    assert result == "http://172.19.0.2:8100"
    assert calls == [stale_hostname, live_container_id]


def test_resolve_sandbox_host_container_url_raises_when_hostname_and_mountinfo_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_hostname = "6aadabeb0b48"

    def fake_exists(self) -> bool:
        return str(self) == "/.dockerenv"

    def fake_read_text(self, *, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        if str(self) != "/proc/self/mountinfo":
            raise AssertionError(f"unexpected read_text path: {self}")
        return "1522 1498 0:95 / / rw,relatime - overlay overlay rw\n"

    def fake_run(args: list[str], **kwargs: object) -> CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args,
            stderr=f"error: no such object: {stale_hostname}",
        )

    monkeypatch.setenv("HOSTNAME", stale_hostname)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr(docker_module.Path, "exists", fake_exists)
    monkeypatch.setattr(docker_module.Path, "read_text", fake_read_text)
    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=f"container={stale_hostname}"):
        resolve_sandbox_host_container_url(
            docker_binary="docker",
            sandbox_network="harnyx-net",
            rpc_port=8100,
        )
