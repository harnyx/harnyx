"""Domain-specific exceptions shared across sandbox components."""

from __future__ import annotations

from enum import StrEnum


class SandboxError(Exception):
    """Base class for sandbox-specific failures."""


class MissingEntrypointError(SandboxError):
    """Raised when an entrypoint name has not been registered."""


class BudgetExceededError(RuntimeError):
    """Raised when a tool call exceeds the configured session budget."""


class SessionBudgetExhaustedError(RuntimeError):
    """Raised when execution exhausts the session hard limit."""


class ConcurrencyLimitError(RuntimeError):
    """Raised when a token exceeds its permitted parallel call allowance."""


class ToolInvocationTimeoutError(RuntimeError):
    """Raised when a caller-selected tool invocation deadline expires."""


class ToolProviderFailureCode(StrEnum):
    CREDENTIAL_UNAVAILABLE = "credential_unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    PROVIDER_FAILED = "provider_failed"


class ProviderCredentialUnavailableError(RuntimeError):
    """Raised when a configured provider has no usable credential."""

    def __init__(self, provider: str) -> None:
        super().__init__("configured provider credential unavailable")
        self.provider = provider


class ToolProviderError(RuntimeError):
    """Raised when a tool's backing provider fails after retry exhaustion."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: ToolProviderFailureCode = ToolProviderFailureCode.PROVIDER_FAILED,
        provider: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.provider = provider
        self.http_status = http_status


__all__ = [
    "SandboxError",
    "MissingEntrypointError",
    "BudgetExceededError",
    "SessionBudgetExhaustedError",
    "ConcurrencyLimitError",
    "ToolInvocationTimeoutError",
    "ProviderCredentialUnavailableError",
    "ToolProviderFailureCode",
    "ToolProviderError",
]
