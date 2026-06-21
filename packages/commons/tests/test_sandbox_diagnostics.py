from __future__ import annotations

import json
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from harnyx_commons.json_types import JsonValue
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.docker import DockerSandboxManager
from harnyx_commons.sandbox.options import SandboxOptions


class _FakeSandboxClient(SandboxClient):
    async def invoke(
        self,
        entrypoint: str,
        *,
        payload: Mapping[str, JsonValue],
        context: Mapping[str, JsonValue],
        token: str,
        session_id: UUID,
    ) -> Mapping[str, JsonValue]:
        del entrypoint, payload, context, token, session_id
        raise AssertionError("invoke should not be reached")

    def close(self) -> None:
        return None


def test_docker_sandbox_manager_writes_diagnostics_when_docker_run_fails(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        raise subprocess.CalledProcessError(
            returncode=125,
            cmd=args,
            output="container stdout",
            stderr="docker error includes super-secret and /state/agent.py while binding 0.0.0.0:8000",
        )

    options = _sandbox_options(tmp_path)
    _precreate_public_file(tmp_path / "docker-run-result.json")
    manager = DockerSandboxManager(command_runner=command_runner)

    with pytest.raises(RuntimeError, match="docker run failed") as excinfo:
        manager.start(options)

    raised_message = str(excinfo.value)
    assert "SECRET_TOKEN=<redacted>" in raised_message
    assert "AGENT_PATH=<redacted>" in raised_message
    assert "0.0.0.0:8000" in raised_message
    assert "super-secret" not in raised_message
    assert "/state/agent.py" not in raised_message
    assert commands == [_expected_docker_run(options)]
    docker_run_result = json.loads((tmp_path / "docker-run-result.json").read_text(encoding="utf-8"))
    assert docker_run_result["stderr"] == (
        "docker error includes <redacted> and <redacted> while binding 0.0.0.0:8000"
    )
    error_text = (tmp_path / "error.txt").read_text(encoding="utf-8")
    assert error_text.startswith("CalledProcessError:")
    assert "0.0.0.0" in error_text  # noqa: S104 - verifying diagnostic text preserves bind address
    assert "8000" in error_text
    assert "super-secret" not in error_text
    assert "/state/agent.py" not in error_text
    docker_run = (tmp_path / "docker-run.txt").read_text(encoding="utf-8")
    assert "AGENT_PATH=<redacted>" in docker_run
    assert "SECRET_TOKEN=<redacted>" in docker_run
    assert "/state/agent.py" not in docker_run
    assert "super-secret" not in docker_run
    sandbox_options = json.loads((tmp_path / "sandbox-options.json").read_text(encoding="utf-8"))
    assert sandbox_options["env"] == {
        "AGENT_PATH": "<redacted>",
        "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - verifying diagnostic snapshot preserves sandbox host
        "SANDBOX_PORT": "8000",
        "SECRET_TOKEN": "<redacted>",
    }
    _assert_private_mode(tmp_path, 0o700)
    _assert_private_mode(tmp_path / "sandbox-options.json", 0o600)
    _assert_private_mode(tmp_path / "docker-run.txt", 0o600)
    _assert_private_mode(tmp_path / "error.txt", 0o600)
    _assert_private_mode(tmp_path / "docker-run-result.json", 0o600)
    assert not (tmp_path / "docker-inspect.json").exists()
    assert not (tmp_path / "docker-logs.txt").exists()


def test_docker_sandbox_manager_times_out_docker_run(tmp_path: Path) -> None:
    observed_timeout: float | None = None

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal observed_timeout
        observed_timeout = kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd=args, timeout=observed_timeout)

    options = _sandbox_options(tmp_path)
    manager = DockerSandboxManager(command_runner=command_runner, command_timeout_seconds=3.5)

    with pytest.raises(RuntimeError, match="docker run timed out after 3.5s"):
        manager.start(options)

    assert observed_timeout == 3.5
    error_text = (tmp_path / "error.txt").read_text(encoding="utf-8")
    assert error_text.startswith("TimeoutExpired:")


def test_docker_sandbox_manager_writes_diagnostics_when_docker_pull_retries_exhaust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("harnyx_commons.sandbox.docker.time.sleep", lambda _seconds: None)
    commands: list[list[str]] = []

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(list(args))
        if args[:2] == ["docker", "pull"]:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=args,
                output="pull stdout mentions super-secret",
                stderr="pull stderr mentions /state/agent.py",
            )
        raise AssertionError(f"unexpected command: {args}")

    options = replace(_sandbox_options(tmp_path), pull_policy="always")
    _precreate_public_file(tmp_path / "docker-pull-result.json")
    manager = DockerSandboxManager(command_runner=command_runner)

    with pytest.raises(RuntimeError, match="docker pull failed after 3 attempts"):
        manager.start(options)

    assert commands == [["docker", "pull", options.image]] * 3
    pull_result = json.loads((tmp_path / "docker-pull-result.json").read_text(encoding="utf-8"))
    assert pull_result == {
        "returncode": 1,
        "stdout": "pull stdout mentions <redacted>",
        "stderr": "pull stderr mentions <redacted>",
    }
    error_text = (tmp_path / "error.txt").read_text(encoding="utf-8")
    assert "docker pull failed after 3 attempts" in error_text
    assert "super-secret" not in error_text
    assert "/state/agent.py" not in error_text
    assert (tmp_path / "docker-pull.txt").read_text(encoding="utf-8") == f"docker pull {options.image}"
    assert not (tmp_path / "docker-inspect.json").exists()
    assert not (tmp_path / "docker-logs.txt").exists()
    assert not (tmp_path / "docker-run-result.json").exists()
    _assert_private_mode(tmp_path / "docker-pull-result.json", 0o600)


def test_docker_sandbox_manager_writes_inspect_and_logs_before_cleanup(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    container_id = "sandbox-container-1"

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{container_id}\n", stderr="")
        if args == ["docker", "inspect", container_id]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='[{"Id":"sandbox-container-1","Config":{"Env":["SECRET_TOKEN=super-secret"]}}]\n',
                stderr="",
            )
        if args == ["docker", "logs", container_id]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="sandbox listening on 0.0.0.0:8000\nsandbox log line\n",
                stderr="",
            )
        if args[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    options = _sandbox_options(tmp_path, wait_for_healthz=True, healthz_timeout=0.0)
    _precreate_public_file(tmp_path / "docker-inspect.json")
    _precreate_public_file(tmp_path / "docker-logs.txt")
    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: _FakeSandboxClient(),
    )

    with pytest.raises(RuntimeError, match="sandbox healthz did not succeed"):
        manager.start(options)

    assert ["docker", "inspect", container_id] in commands
    assert ["docker", "logs", container_id] in commands
    assert any(command[:2] == ["docker", "stop"] for command in commands)
    assert any(command == ["docker", "rm", "-f", container_id] for command in commands)
    inspect_index = commands.index(["docker", "inspect", container_id])
    logs_index = commands.index(["docker", "logs", container_id])
    remove_index = commands.index(["docker", "rm", "-f", container_id])
    assert inspect_index < remove_index
    assert logs_index < remove_index
    inspect_output = (tmp_path / "docker-inspect.json").read_text(encoding="utf-8")
    assert "SECRET_TOKEN=<redacted>" in inspect_output
    assert "super-secret" not in inspect_output
    assert (tmp_path / "docker-logs.txt").read_text(encoding="utf-8") == (
        "sandbox listening on 0.0.0.0:8000\nsandbox log line\n"
    )
    _assert_private_mode(tmp_path / "docker-inspect.json", 0o600)
    _assert_private_mode(tmp_path / "docker-logs.txt", 0o600)


def test_docker_sandbox_manager_writes_private_diagnostic_command_error_files(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    container_id = "sandbox-container-1"
    error_path = tmp_path / "docker-inspect.json.error.txt"
    _precreate_public_file(error_path)

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{container_id}\n", stderr="")
        if args == ["docker", "inspect", container_id]:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=args,
                output="",
                stderr="inspect failed",
            )
        if args == ["docker", "logs", container_id]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="sandbox log line\n", stderr="")
        if args[:2] == ["docker", "stop"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    options = _sandbox_options(tmp_path, wait_for_healthz=True, healthz_timeout=0.0)
    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: _FakeSandboxClient(),
    )

    with pytest.raises(RuntimeError, match="sandbox healthz did not succeed"):
        manager.start(options)

    assert ["docker", "inspect", container_id] in commands
    assert error_path.read_text(encoding="utf-8").startswith("command=docker inspect")
    _assert_private_mode(error_path, 0o600)


def test_docker_sandbox_manager_removes_container_when_stop_fails(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    container_id = "sandbox-container-1"

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{container_id}\n", stderr="")
        if args == ["docker", "inspect", container_id]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="[]\n", stderr="")
        if args == ["docker", "logs", container_id]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="sandbox log line\n", stderr="")
        if args[:2] == ["docker", "stop"]:
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="stop failed")
        if args == ["docker", "rm", "-f", container_id]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    options = _sandbox_options(tmp_path, wait_for_healthz=True, healthz_timeout=0.0)
    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: _FakeSandboxClient(),
    )

    with pytest.raises(RuntimeError, match="sandbox healthz did not succeed"):
        manager.start(options)

    assert ["docker", "rm", "-f", container_id] in commands


def test_docker_sandbox_manager_publishes_allocated_port_on_client_host(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    container_id = "sandbox-container-1"

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{container_id}\n", stderr="")
        if args == ["docker", "port", container_id, "8000"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="127.0.0.1:45678\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {args}")

    options = replace(_sandbox_options(tmp_path), host_port=0)
    manager = DockerSandboxManager(
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: _FakeSandboxClient(),
    )

    deployment = manager.start(options)

    assert deployment.base_url == "http://127.0.0.1:45678"
    assert commands[0] == _expected_docker_run(options)


def test_docker_sandbox_manager_binds_allocated_port_when_configured(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    container_id = "sandbox-container-1"

    def command_runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(args)
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{container_id}\n", stderr="")
        if args == ["docker", "port", container_id, "8000"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="127.0.0.1:45678\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {args}")

    options = replace(_sandbox_options(tmp_path), host_port=0)
    manager = DockerSandboxManager(
        published_port_bind_host="127.0.0.1",
        command_runner=command_runner,
        client_factory=lambda base_url, host_container_url: _FakeSandboxClient(),
    )

    deployment = manager.start(options)

    assert deployment.base_url == "http://127.0.0.1:45678"
    port_index = commands[0].index("-p")
    assert commands[0][port_index : port_index + 2] == ["-p", "127.0.0.1::8000"]


def _sandbox_options(
    diagnostics_dir: Path,
    *,
    wait_for_healthz: bool = False,
    healthz_timeout: float = 15.0,
) -> SandboxOptions:
    return SandboxOptions(
        image="local/test-sandbox",
        container_name="test-sandbox",
        pull_policy="missing",
        host_port=12345,
        container_port=8000,
        env={
            "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - inside container
            "SANDBOX_PORT": "8000",
            "AGENT_PATH": "/state/agent.py",
            "SECRET_TOKEN": "super-secret",
        },
        host_container_url="http://host.docker.internal:39100",
        wait_for_healthz=wait_for_healthz,
        healthz_timeout=healthz_timeout,
        failure_diagnostics_dir=str(diagnostics_dir),
    )


def _expected_docker_run(options: SandboxOptions) -> list[str]:
    args = [
        "docker",
        "run",
        "--pull",
        options.pull_policy,
        "-d",
        "--name",
        options.container_name,
    ]
    for key, value in sorted(options.labels.items()):
        args.extend(["--label", f"{key}={value}"])
    args.extend(
        [
            "-p",
            _expected_port_publish(options),
            "-e",
            "SANDBOX_HOST=0.0.0.0",
            "-e",
            "SANDBOX_PORT=8000",
            "-e",
            "AGENT_PATH=/state/agent.py",
            "-e",
            "SECRET_TOKEN=super-secret",
            options.image,
        ]
    )
    return args


def _expected_port_publish(options: SandboxOptions) -> str:
    if options.host_port == 0:
        return str(options.container_port)
    return f"{options.host_port}:{options.container_port}"


def _precreate_public_file(path: Path) -> None:
    path.write_text("old public content", encoding="utf-8")
    path.chmod(0o644)


def _assert_private_mode(path: Path, expected_mode: int) -> None:
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode
