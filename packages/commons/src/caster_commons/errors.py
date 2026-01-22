"""Domain-specific exceptions shared across sandbox components."""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for sandbox-specific failures."""


class MissingEntrypointError(SandboxError):
    """Raised when an entrypoint name has not been registered."""


class BudgetExceededError(RuntimeError):
    """Raised when a tool call exceeds the configured session budget."""


class ConcurrencyLimitError(RuntimeError):
    """Raised when a token exceeds its permitted parallel call allowance."""


__all__ = [
    "SandboxError",
    "MissingEntrypointError",
    "BudgetExceededError",
    "ConcurrencyLimitError",
]
