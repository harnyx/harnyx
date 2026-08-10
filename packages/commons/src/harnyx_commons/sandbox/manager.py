"""Sandbox manager interfaces shared across platform and validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.options import SandboxOptions


@dataclass(frozen=True)
class SandboxDeployment:
    """Metadata describing a running sandbox instance."""

    client: SandboxClient
    identifier: str | None = None
    base_url: str | None = None
    log_stream_id: str | None = None
    stop_timeout_seconds: int | None = None


class SandboxStartError(RuntimeError):
    """Startup failed after launch and the physical sandbox may still exist."""

    def __init__(self, message: str, *, unremoved_deployment: SandboxDeployment) -> None:
        super().__init__(message)
        self.unremoved_deployment = unremoved_deployment


class SandboxManager(Protocol):
    """Lifecycle manager responsible for provisioning sandbox entrypoints."""

    def start(self, options: SandboxOptions) -> SandboxDeployment:
        """Start the sandbox and return a deployment descriptor."""

    def stop(self, deployment: SandboxDeployment) -> bool:
        """Stop the sandbox and report whether physical removal was confirmed."""


__all__ = [
    "SandboxDeployment",
    "SandboxManager",
    "SandboxStartError",
]
