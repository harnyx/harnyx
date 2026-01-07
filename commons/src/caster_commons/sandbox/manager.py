"""Sandbox manager interfaces shared across platform and validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from caster_commons.sandbox.client import SandboxClient
from caster_commons.sandbox.options import SandboxOptions, default_token_header


@dataclass(frozen=True)
class SandboxDeployment:
    """Metadata describing a running sandbox instance."""

    client: SandboxClient
    identifier: str | None = None
    base_url: str | None = None
    log_stream_id: str | None = None
    stop_timeout_seconds: int | None = None


class SandboxManager(Protocol):
    """Lifecycle manager responsible for provisioning sandbox entrypoints."""

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        """Start the sandbox and return a deployment descriptor."""

    def stop(self, deployment: SandboxDeployment) -> None:
        """Stop the sandbox instance described by the deployment."""


__all__ = [
    "SandboxDeployment",
    "SandboxManager",
    "default_token_header",
]
