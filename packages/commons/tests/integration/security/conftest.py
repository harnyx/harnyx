from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from caster_commons.sandbox.docker import (
    DockerSandboxManager,
    SandboxOptions,
    resolve_sandbox_host_container_url,
)
from caster_commons.sandbox.runtime import CONTAINER_SECURITY
from caster_commons.sandbox.seccomp.paths import default_profile_path

DOCKER_CLI = os.getenv("DOCKER_CLI", "docker")
DOCKER_BINARY = shutil.which(DOCKER_CLI) or DOCKER_CLI
DOCKER_IMAGE_PATTERN = re.compile(r"^[\w./:-]+$")
DEFAULT_TOKEN_HEADER = os.getenv("CASTER_TOKEN_HEADER", "x-caster-token")


@pytest.fixture(scope="session")
def attacker_agent_path() -> Path:
    return Path(__file__).with_name("attacker_agent.py")


def _require_docker_cli() -> None:
    try:
        subprocess.run(  # noqa: S603 - static docker command
            [DOCKER_BINARY, "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - depends on host tooling
        pytest.fail(f"Docker CLI is required to run security tests: {exc}")


def _require_image(image: str) -> None:
    if not DOCKER_IMAGE_PATTERN.fullmatch(image):
        raise ValueError("Invalid docker image reference")
    result = subprocess.run(  # noqa: S603 - static docker command
        [DOCKER_BINARY, "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            (
                "Sandbox image not found. Build it with scripts/build/build_sandbox_image.sh before "
                "running security tests."
            ),
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sandbox(attacker_agent_path: Path):
    _require_docker_cli()
    image = os.getenv("CASTER_SANDBOX_IMAGE", "local/caster-sandbox:0.1.0-dev")
    _require_image(image)

    sandbox_network = "bridge"
    host_container_url = resolve_sandbox_host_container_url(
        docker_binary=DOCKER_BINARY,
        sandbox_network=sandbox_network,
        rpc_port=1,
    )

    port = _find_free_port()
    manager = DockerSandboxManager(docker_binary=DOCKER_BINARY, host="127.0.0.1")
    options = SandboxOptions(
        image=image,
        container_name=f"security-{uuid.uuid4().hex[:8]}",
        pull_policy="missing",
        host_port=port,
        container_port=8000,
        env={
            "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - container needs to bind all interfaces
            "SANDBOX_PORT": "8000",
            "CASTER_TOKEN_HEADER": DEFAULT_TOKEN_HEADER,
            "CASTER_AGENT_PATH": "/sandbox/agent.py",
        },
        volumes=((str(attacker_agent_path), "/sandbox/agent.py", "ro"),),
        wait_for_healthz=True,
        token_header=DEFAULT_TOKEN_HEADER,
        host_container_url=host_container_url,
        network=sandbox_network,
        user=CONTAINER_SECURITY.user,
        seccomp_profile=default_profile_path(),
        ulimits=CONTAINER_SECURITY.ulimits,
        extra_args=CONTAINER_SECURITY.extra_args,
    )
    deployment = manager.start(options)
    try:
        yield deployment.client
    finally:
        manager.stop(deployment)
