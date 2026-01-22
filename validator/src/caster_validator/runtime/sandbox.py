"""Sandbox configuration and factory for validator runtime."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from caster_commons.sandbox.docker import DockerSandboxManager
from caster_commons.sandbox.manager import default_token_header
from caster_commons.sandbox.options import SandboxOptions
from caster_validator.runtime.seccomp.paths import default_profile_path


@dataclass(frozen=True, slots=True)
class ContainerSecurity:
    """Security constraints for sandbox containers."""

    user: str = "caster"
    ulimits: tuple[str, ...] = ("nproc=128:128", "nofile=512:512")
    pids_limit: int = 128
    memory: str = "1g"
    cpus: str = "1"

    @property
    def extra_args(self) -> tuple[str, ...]:
        """Docker run arguments for security hardening."""
        tmpfs = f"{Path(tempfile.gettempdir())}:rw,noexec,nosuid,nodev,size=64m"
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
            "--tmpfs",
            tmpfs,
        )


# Default security profile for validator sandboxes
CONTAINER_SECURITY = ContainerSecurity()


def create_sandbox_manager() -> DockerSandboxManager:
    """Create the Docker sandbox manager with standard configuration."""
    sandbox_log = logging.getLogger("caster_validator.sandbox")
    return DockerSandboxManager(
        docker_binary="/usr/bin/docker",
        host="host.docker.internal",
        log_consumer=lambda line: sandbox_log.info("%s", line),
    )


def build_sandbox_options(
    *,
    image: str,
    network: str | None,
    pull_policy: str,
    validator_url: str,
    container_name: str = "caster-sandbox-smoke",
) -> SandboxOptions:
    """Build sandbox options for a validator evaluation run.

    Args:
        image: Docker image to use for the sandbox.
        network: Docker network to attach the container to.
        pull_policy: Image pull policy ("always", "missing", "never").
        validator_url: URL for the validator RPC endpoint.
        container_name: Name for the container.

    Returns:
        Configured SandboxOptions instance.
    """
    container_port = 8000
    token_header = default_token_header()

    return SandboxOptions(
        image=image,
        container_name=container_name,
        pull_policy=pull_policy,
        host_port=None if network else 0,
        container_port=container_port,
        env={
            "SANDBOX_HOST": "0.0.0.0",  # noqa: S104
            "SANDBOX_PORT": str(container_port),
            "CASTER_VALIDATOR_URL": validator_url,
            "CASTER_TOKEN_HEADER": token_header,
        },
        entrypoint=None,
        command=None,
        network=network,
        token_header=token_header,
        extra_hosts=(("host.docker.internal", "host-gateway"),),
        startup_delay_seconds=2.0,
        wait_for_healthz=True,
        stop_timeout_seconds=5,
        user=CONTAINER_SECURITY.user,
        seccomp_profile=default_profile_path(),
        ulimits=CONTAINER_SECURITY.ulimits,
        extra_args=CONTAINER_SECURITY.extra_args,
    )


__all__ = [
    "CONTAINER_SECURITY",
    "ContainerSecurity",
    "build_sandbox_options",
    "create_sandbox_manager",
]
