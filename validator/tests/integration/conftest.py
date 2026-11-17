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

from harnyx_commons.sandbox.agent_staging import stage_agent_source
from harnyx_commons.sandbox.docker import (
    DockerSandboxManager,
    SandboxOptions,
    resolve_sandbox_host_container_url,
)
from harnyx_commons.sandbox.manager import SandboxDeployment
from harnyx_commons.sandbox.runtime import ContainerSecurity
from harnyx_commons.sandbox.seccomp.paths import default_profile_path
from harnyx_commons.sandbox.state import DEFAULT_STATE_DIR
from harnyx_miner_sdk.sandbox_protocol import EVALUATION_RESOURCE_POLICY_ENV

_DOCKER_CLI = os.getenv("DOCKER_CLI", "docker")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE = os.getenv("SANDBOX_IMAGE", "local/harnyx-sandbox:0.1.0-dev")


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
        raise RuntimeError(f"Docker CLI is required for this test suite: {exc}") from exc


def _ensure_image_present(docker_bin: str, image: str) -> None:
    result = subprocess.run(  # noqa: S603 - docker image inspected from trusted config
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
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


def pytest_addoption(parser):
    parser.addoption(
        "--artifact-storage", action="store_true", default=False,
        help="Run manual Docker/loop artifact-storage acceptance tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--artifact-storage"):
        return
    manual = [item for item in items if item.get_closest_marker("artifact_storage")]
    if manual:
        items[:] = [item for item in items if not item.get_closest_marker("artifact_storage")]
        config.hook.pytest_deselected(items=manual)


@pytest.fixture
def sandbox_launcher(request) -> Callable[[str], SandboxDeployment]:
    """Start a sandbox container for the provided agent module and clean it up afterward."""

    docker_bin = _docker_binary()
    _ensure_docker_available(docker_bin)
    image = _DEFAULT_IMAGE
    _ensure_image_present(docker_bin, image)

    manager = DockerSandboxManager(docker_binary=docker_bin, host="127.0.0.1")
    deployments = []
    storage = None
    if request.config.getoption("--artifact-storage"):
        storage = request.getfixturevalue("artifact_storage_manager")
    workspace_root = Path(os.getenv("HOST_WORKSPACE", str(_REPO_ROOT)))
    state_dir = Path(tempfile.mkdtemp(prefix=".harnyx-validator-int-state-", dir=workspace_root))

    def _start(
        agent_module: str, *, source: bytes | None = None, evaluation_policy: bool = False, workspace: bool = True,
    ):
        module_rel_path = Path(*agent_module.split(".")).with_suffix(".py")
        module_path = _REPO_ROOT / module_rel_path
        if not module_path.exists():
            raise RuntimeError(f"agent module file not found: module={agent_module} path={module_path}")
        artifact = stage_agent_source(
            state_dir=state_dir,
            container_root=DEFAULT_STATE_DIR,
            namespace="integration_agents",
            key=agent_module.replace(".", "_"),
            data=module_path.read_bytes() if source is None else source,
        )
        port = _find_free_port()
        host_container_url = resolve_sandbox_host_container_url(
            docker_binary=docker_bin,
            sandbox_network="bridge",
            rpc_port=1,
        )
        options = SandboxOptions(
            image=image,
            container_name=f"{storage._pool if storage else 'validator-int'}-sandbox-{uuid.uuid4().hex[:8]}",
            labels={"storage-test": storage._pool} if storage else {},
            pull_policy="missing",
            host_port=port,
            container_port=8000,
            env={
                "SANDBOX_HOST": "0.0.0.0",  # noqa: S104 - container binding
                "SANDBOX_PORT": "8000",
                "AGENT_PATH": artifact.container_path,
                **({EVALUATION_RESOURCE_POLICY_ENV: "1"} if evaluation_policy else {}),
            },
            volumes=((str(state_dir), DEFAULT_STATE_DIR, "ro"),),
            wait_for_healthz=True,
            healthz_timeout=30.0,
            host_container_url=host_container_url,
            user="harnyx" if evaluation_policy else None,
            seccomp_profile=default_profile_path() if evaluation_policy else None,
            ulimits=ContainerSecurity().ulimits if evaluation_policy else (),
            extra_args=(
                (*ContainerSecurity(memory="1536m").extra_args, "--memory-swap", "1536m",
                 *(("--mount", "type=volume,destination=/sandbox/queries") if storage is None and workspace else ()),
                 ) if evaluation_policy else ()
            ),
        )
        deployment = (storage if storage is not None and evaluation_policy else manager).start(options)
        deployments.append(deployment)
        return deployment

    yield _start

    for deployment in deployments:
        (storage or manager).stop(deployment)

    shutil.rmtree(state_dir, ignore_errors=True)


@pytest.fixture
def artifact_storage_manager(request):
    """Real same-image helper, explicitly opted in on a suitable development host."""
    if not request.config.getoption("--artifact-storage"):
        raise RuntimeError("manual loop-storage trial requires --artifact-storage")
    import json

    from harnyx_validator.infrastructure.artifact_storage import (
        ArtifactStorageSandboxManager,
        ValidatorRuntimeIdentity,
    )

    docker = _docker_binary()
    image = "local/harnyx-validator:0.1.0-dev"
    _ensure_image_present(docker, image)
    prefix = f"harnyx-storage-test-{uuid.uuid4().hex}"

    def run(*args):
        return subprocess.run(  # noqa: S603
            [docker, *args], check=True, text=True, capture_output=True, timeout=180,
        ).stdout.strip()

    owner = run("run", "-d", "--name", f"{prefix}-owner", "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--entrypoint", "/bin/sleep", image, "infinity")
    inspected = json.loads(run("inspect", owner))[0]
    iops, mib_per_second = getattr(request, "param", (32, 4))
    manager = ArtifactStorageSandboxManager(
        DockerSandboxManager(docker_binary=docker, host="127.0.0.1"),
        ValidatorRuntimeIdentity(container_id=owner, started_at=inspected["State"]["StartedAt"],
                                 image_id=inspected["Image"]),
        iops=iops, bytes_per_second=mib_per_second * 1024**2,
        docker_binary=docker, pool_name=prefix, image_bytes=128 * 1024**2,
    )
    try:
        manager.recover(sandbox_labels={"storage-test": prefix}, sandbox_name_prefix=f"{prefix}-sandbox-")
        yield manager
    finally:
        try:
            manager.close()
            run("volume", "rm", f"{prefix}-backing")
        finally:
            run("rm", "-f", owner)
