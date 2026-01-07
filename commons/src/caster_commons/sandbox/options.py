"""Sandbox option types shared across managers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_TOKEN_HEADER = "x-caster-token"  # noqa: S105


def default_token_header() -> str:
    return DEFAULT_TOKEN_HEADER


@dataclass(frozen=True)
class SandboxOptions:
    """Configuration for launching a sandbox container."""

    image: str
    container_name: str
    pull_policy: str = "always"
    host_port: int | None = 8000
    container_port: int = 8000
    env: Mapping[str, str] = field(default_factory=dict)
    entrypoint: str | None = None
    command: Sequence[str] | None = None
    network: str | None = None
    token_header: str = field(default_factory=default_token_header)
    volumes: Sequence[tuple[str, str, str | None]] = field(default_factory=tuple)
    working_dir: str | None = None
    extra_hosts: Sequence[tuple[str, str]] = field(default_factory=tuple)
    startup_delay_seconds: float = 0.0
    wait_for_healthz: bool = False
    healthz_path: str = "/healthz"
    healthz_timeout: float = 15.0
    stop_timeout_seconds: int | None = 5
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    user: str | None = None
    seccomp_profile: str | None = None
    ulimits: Sequence[str] = field(default_factory=tuple)


__all__ = ["SandboxOptions", "DEFAULT_TOKEN_HEADER", "default_token_header"]
