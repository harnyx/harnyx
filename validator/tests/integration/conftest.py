"""Integration-level fixtures for validator tests."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from caster_commons.sandbox.agent_staging import stage_agent_source
from caster_commons.sandbox.docker import (
    DockerSandboxManager,
    SandboxOptions,
    resolve_sandbox_host_container_url,
)
from caster_commons.sandbox.manager import SandboxDeployment
from caster_commons.sandbox.state import DEFAULT_STATE_DIR

_DOCKER_CLI = os.getenv("DOCKER_CLI", "docker")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE = os.getenv("CASTER_SANDBOX_IMAGE", "local/caster-sandbox:0.1.0-dev")


def _docker_binary() -> str:
    return shutil.which(_DOCKER_CLI) or _DOCKER_CLI


def _ensure_docker_available(docker_bin: str) -> None:
    try:
        subprocess.run(  # noqa: S603 - docker binary provided by test harness
            [docker_bin, "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - depends on host tooling
        pytest.skip(f"Docker CLI is required for this test suite: {exc}")


def _ensure_image_present(docker_bin: str, image: str) -> None:
    result = subprocess.run(  # noqa: S603 - docker image inspected from trusted config
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            (
                f"Sandbox image {image!r} not found. "
                "Build it with scripts/build/build_sandbox_image.sh before running integration tests."
            ),
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sandbox_launcher() -> Callable[[str], SandboxDeployment]:
    """Start a sandbox container for the provided agent module and clean it up afterward."""

    docker_bin = _docker_binary()
    _ensure_docker_available(docker_bin)
    image = _DEFAULT_IMAGE
    _ensure_image_present(docker_bin, image)

    manager = DockerSandboxManager(docker_binary=docker_bin, host="127.0.0.1")
    deployments = []
    state_dir = Path(tempfile.mkdtemp(prefix="caster-validator-int-state-"))

    def _start(agent_module: str):
        module_rel_path = Path(*agent_module.split(".")).with_suffix(".py")
        module_path = _REPO_ROOT / module_rel_path
        if not module_path.exists():
            raise RuntimeError(f"agent module file not found: module={agent_module} path={module_path}")
        artifact = stage_agent_source(
            state_dir=state_dir,
            container_root=DEFAULT_STATE_DIR,
            namespace="integration_agents",
            key=agent_module.replace(".", "_"),
            data=module_path.read_bytes(),
        )
        port = _find_free_port()
        host_container_url = resolve_sandbox_host_container_url(
            docker_binary=docker_bin,
            sandbox_network="bridge",
            rpc_port=1,
        )
        options = SandboxOptions(
            image=image,
            container_name=f"validator-int-{uuid.uuid4().hex[:8]}",
            pull_policy="missing",
            host_port=port,
            container_port=8000,
            env={
                "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - container binding
                "SANDBOX_PORT": "8000",
                "CASTER_AGENT_PATH": artifact.container_path,
            },
            volumes=((str(state_dir), DEFAULT_STATE_DIR, "ro"),),
            wait_for_healthz=True,
            healthz_timeout=30.0,
            host_container_url=host_container_url,
        )
        deployment = manager.start(options)
        deployments.append(deployment)
        return deployment

    yield _start

    for deployment in deployments:
        manager.stop(deployment)

    shutil.rmtree(state_dir, ignore_errors=True)
