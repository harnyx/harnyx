from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from harnyx_commons.sandbox.agent_staging import stage_agent_source
from harnyx_commons.sandbox.docker import (
    DockerSandboxManager,
    SandboxOptions,
    resolve_sandbox_host_container_url,
)
from harnyx_commons.sandbox.runtime import CONTAINER_SECURITY
from harnyx_commons.sandbox.seccomp.paths import default_profile_path

DOCKER_CLI = os.getenv("DOCKER_CLI", "docker")
DOCKER_BINARY = shutil.which(DOCKER_CLI) or DOCKER_CLI
DOCKER_IMAGE_PATTERN = re.compile(r"^[\w./:-]+$")
SECURITY_SANDBOX_IMAGE = os.getenv(
    "SECURITY_SANDBOX_IMAGE",
    "local/harnyx-sandbox:0.1.0-dev",
)


@dataclass(frozen=True)
class ResultIpcBarrier:
    url: str
    handler: type[_ResultIpcBarrierHandler]

    def observed(self, barrier_id: str, role: str) -> bool:
        with self.handler.condition:
            return role in self.handler.arrivals.get(barrier_id, set())


class _ResultIpcBarrierHandler(BaseHTTPRequestHandler):
    condition = threading.Condition()
    arrivals: dict[str, set[str]] = {}

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        barrier_id = query.get("barrier_id", [""])[0]
        role = query.get("role", [""])[0]
        with self.condition:
            roles = self.arrivals.setdefault(barrier_id, set())
            roles.add(role)
            self.condition.notify_all()
            ready = role == "reduce" or self.condition.wait_for(
                lambda: {"malicious", "healthy"} <= self.arrivals.get(barrier_id, set()),
                timeout=3,
            )
        self.send_response(204 if ready else 504)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


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
def result_ipc_barrier() -> ResultIpcBarrier:
    port = _find_free_port()
    server = ThreadingHTTPServer(("0.0.0.0", port), _ResultIpcBarrierHandler)  # noqa: S104 - Docker bridge access
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = resolve_sandbox_host_container_url(
        docker_binary=DOCKER_BINARY,
        sandbox_network="bridge",
        rpc_port=port,
    )
    try:
        yield ResultIpcBarrier(url=url, handler=_ResultIpcBarrierHandler)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def sandbox(attacker_agent_path: Path):
    _require_docker_cli()
    image = SECURITY_SANDBOX_IMAGE
    _require_image(image)

    sandbox_network = "bridge"
    host_container_url = resolve_sandbox_host_container_url(
        docker_binary=DOCKER_BINARY,
        sandbox_network=sandbox_network,
        rpc_port=1,
    )

    port = _find_free_port()
    manager = DockerSandboxManager(docker_binary=DOCKER_BINARY, host="127.0.0.1")
    workspace_root = Path(os.getenv("HOST_WORKSPACE", str(Path(__file__).resolve().parents[6])))
    state_dir = Path(tempfile.mkdtemp(prefix=".harnyx-security-int-state-", dir=workspace_root))
    artifact = stage_agent_source(
        state_dir=state_dir,
        container_root="/sandbox",
        namespace="security_agents",
        key="attacker_agent",
        data=attacker_agent_path.read_bytes(),
    )
    options = SandboxOptions(
        image=image,
        container_name=f"security-{uuid.uuid4().hex[:8]}",
        pull_policy="missing",
        host_port=port,
        container_port=8000,
        env={
            "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - container needs to bind all interfaces
            "SANDBOX_PORT": "8000",
            "AGENT_PATH": "/sandbox/agent.py",
            "ENTRYPOINT_TIMEOUT_SECONDS": "5",
            "SANDBOX_PIDS_PROBE_LIMIT": str(CONTAINER_SECURITY.pids_limit + 100),
        },
        volumes=((str(artifact.host_path), "/sandbox/agent.py", "ro"),),
        wait_for_healthz=True,
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
        shutil.rmtree(state_dir, ignore_errors=True)
