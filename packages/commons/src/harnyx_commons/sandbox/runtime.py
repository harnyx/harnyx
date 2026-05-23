"""Shared sandbox manager + hardened default options for services."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from harnyx_commons.sandbox.docker import DockerSandboxManager, resolve_sandbox_host_container_url
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_commons.sandbox.seccomp.paths import default_profile_path

DOCKER_BINARY: Final[str] = "/usr/bin/docker"
HOST_PROBE_ADDRESS: Final[str] = "host.docker.internal"


@dataclass(frozen=True, slots=True)
class ContainerSecurity:
    """Security constraints for sandbox containers."""

    user: str = "harnyx"
    ulimits: tuple[str, ...] = ("nproc=512:512", "nofile=512:512")
    pids_limit: int = 512
    memory: str = "1g"
    cpus: str = "1"

    @property
    def extra_args(self) -> tuple[str, ...]:
        """Docker run arguments for security hardening."""
        return (
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
        )


CONTAINER_SECURITY: Final[ContainerSecurity] = ContainerSecurity()


def create_sandbox_manager(
    *,
    logger_name: str,
    host: str = HOST_PROBE_ADDRESS,
    published_port_bind_host: str | None = None,
) -> DockerSandboxManager:
    """Create a Docker sandbox manager with standard configuration."""

    sandbox_log = logging.getLogger(logger_name)
    return DockerSandboxManager(
        docker_binary=DOCKER_BINARY,
        # Published-port health probes must use a service-reachable host address for this runtime.
        host=host,
        published_port_bind_host=published_port_bind_host,
        log_consumer=lambda line: sandbox_log.info("%s", line),
    )


def build_sandbox_options(
    *,
    image: str,
    network: str | None,
    pull_policy: str,
    rpc_port: int,
    container_name: str,
    container_port: int = 8000,
    volumes: tuple[tuple[str, str, str | None], ...] = (),
    extra_env: dict[str, str] | None = None,
    host_container_url: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> SandboxOptions:
    """Build hardened sandbox options shared by platform and validator."""

    resolved_host_container_url = host_container_url or resolve_sandbox_host_container_url(
        docker_binary=DOCKER_BINARY,
        sandbox_network=network,
        rpc_port=rpc_port,
    )

    env: dict[str, str] = {
        "SANDBOX_HOST": "0.0.0.0",  # noqa: S104
        "SANDBOX_PORT": str(container_port),
    }
    if extra_env:
        env.update(extra_env)

    return SandboxOptions(
        image=image,
        container_name=container_name,
        pull_policy=pull_policy,
        host_port=None if network else 0,
        container_port=container_port,
        env=env,
        entrypoint=None,
        command=None,
        network=network,
        host_container_url=resolved_host_container_url,
        volumes=volumes,
        extra_hosts=(("host.docker.internal", "host-gateway"),),
        startup_delay_seconds=2.0,
        wait_for_healthz=True,
        healthz_path="/healthz",
        healthz_timeout=30.0,
        stop_timeout_seconds=5,
        user=CONTAINER_SECURITY.user,
        seccomp_profile=default_profile_path(),
        ulimits=CONTAINER_SECURITY.ulimits,
        extra_args=CONTAINER_SECURITY.extra_args,
        labels=labels or {},
    )


__all__ = [
    "CONTAINER_SECURITY",
    "ContainerSecurity",
    "DOCKER_BINARY",
    "HOST_PROBE_ADDRESS",
    "build_sandbox_options",
    "create_sandbox_manager",
]
