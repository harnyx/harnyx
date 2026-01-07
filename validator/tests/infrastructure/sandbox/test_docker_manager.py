from __future__ import annotations

from dataclasses import dataclass
from subprocess import CompletedProcess

import pytest

from caster_commons.sandbox.docker import (
    DockerSandboxManager,
    SandboxOptions,
)
from caster_commons.sandbox.manager import SandboxDeployment


@dataclass
class DummyClient:
    base_url: str
    token_header: str
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
        stdout = "container123\n" if "-d" in args else ""
        return subprocess_completed(args, stdout)


def subprocess_completed(args: list[str], stdout: str) -> CompletedProcess[str]:
    return CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def test_docker_sandbox_manager_builds_commands(monkeypatch) -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, token_header: str) -> DummyClient:
        return DummyClient(base_url, token_header)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        host_port=9000,
        container_port=8000,
        env={"EXAMPLE": "value"},
        network="caster-net",
    )

    deployment = manager.start(options)
    assert isinstance(deployment, SandboxDeployment)
    assert deployment.identifier == "container123"
    assert deployment.base_url == "http://127.0.0.1:9000"
    assert isinstance(deployment.client, DummyClient)

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
    assert deployment.client.closed is True


def test_docker_manager_skips_port_mapping_when_host_port_missing() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, token_header: str) -> DummyClient:
        return DummyClient(base_url, token_header)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        host_port=None,
        container_port=8000,
        network="caster-net",
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "-p" not in run_args
    assert "--network" in run_args
    assert deployment.base_url == "http://sandbox-demo:8000"
    manager.stop(deployment)


def test_docker_manager_mounts_volumes() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, token_header: str) -> DummyClient:
        return DummyClient(base_url, token_header)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        volumes=(("/host/agent.py", "/workspace/agent.py", "ro"),),
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
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        host_port=None,
    )
    with pytest.raises(ValueError):
        manager.start(options)


def test_docker_manager_adds_extra_hosts() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, token_header: str) -> DummyClient:
        return DummyClient(base_url, token_header)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    options = SandboxOptions(
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        extra_hosts=(("host.docker.internal", "host-gateway"),),
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "--add-host" in run_args
    host_arg_index = run_args.index("--add-host") + 1
    assert run_args[host_arg_index] == "host.docker.internal:host-gateway"
    manager.stop(deployment)


def test_docker_manager_sets_seccomp_profile() -> None:
    runner = RecordingRunner()

    def client_factory(base_url: str, token_header: str) -> DummyClient:
        return DummyClient(base_url, token_header)

    manager = DockerSandboxManager(
        docker_binary="docker",
        host="127.0.0.1",
        command_runner=runner,
        client_factory=client_factory,
    )

    seccomp_path = "/workspace/runtime-seccomp.json"
    options = SandboxOptions(
        image="caster/sandbox:demo",
        container_name="sandbox-demo",
        seccomp_profile=seccomp_path,
    )

    deployment = manager.start(options)
    run_args, _ = runner.commands[0]
    assert "--security-opt" in run_args
    opt_index = run_args.index("--security-opt") + 1
    assert run_args[opt_index] == f"seccomp={seccomp_path}"
    manager.stop(deployment)
