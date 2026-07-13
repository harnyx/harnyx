from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

import harnyx_validator.runtime.bootstrap as bootstrap_mod
from harnyx_commons.sandbox.docker import DockerSandboxManager

pytestmark = pytest.mark.integration

_CPUSET_LABEL = "harnyx.sandbox.cpuset_cpus"
_DEFAULT_SANDBOX_IMAGE = "local/harnyx-sandbox:0.1.0-dev"


def _docker_binary() -> str:
    configured_binary = os.getenv("DOCKER_CLI", "docker")
    return shutil.which(configured_binary) or configured_binary


def _run_docker(docker_binary: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - Docker binary is supplied by the integration harness
        [docker_binary, *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_validator_sandbox_cpuset_is_enforced_by_docker() -> None:
    docker_binary = _docker_binary()
    _run_docker(docker_binary, "version")

    image = os.getenv("SANDBOX_IMAGE", _DEFAULT_SANDBOX_IMAGE)
    _run_docker(docker_binary, "image", "inspect", image)

    settings = SimpleNamespace(
        sandbox=SimpleNamespace(
            sandbox_image=image,
            sandbox_network="bridge",
            sandbox_pull_policy="missing",
        ),
        rpc_port=1,
    )
    options = bootstrap_mod._make_options_factory(settings)()
    cpuset_index = options.extra_args.index("--cpuset-cpus")
    expected_cpuset = options.extra_args[cpuset_index + 1]
    expected_cpu_ids = [int(cpu_id) for cpu_id in expected_cpuset.split(",")]
    options = replace(
        options,
        container_name=f"validator-cpuset-{uuid.uuid4().hex[:8]}",
        startup_delay_seconds=0.0,
    )

    manager = DockerSandboxManager(docker_binary=docker_binary, host="127.0.0.1")
    deployment = manager.start(options)
    try:
        inspected = json.loads(
            _run_docker(docker_binary, "inspect", deployment.identifier).stdout,
        )[0]
        assert inspected["HostConfig"]["CpusetCpus"] == expected_cpuset
        assert inspected["HostConfig"]["NanoCpus"] == 1_000_000_000
        assert inspected["Config"]["Labels"][_CPUSET_LABEL] == expected_cpuset

        process_affinity = _run_docker(
            docker_binary,
            "exec",
            deployment.identifier,
            "python",
            "-c",
            "import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
        )
        assert json.loads(process_affinity.stdout) == expected_cpu_ids
    finally:
        manager.stop(deployment)
