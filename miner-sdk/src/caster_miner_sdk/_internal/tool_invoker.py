from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Protocol


class ToolInvoker(Protocol):
    """Protocol for host-provided tool invokers."""

    async def invoke(
        self,
        method: str,
        *,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


_ACTIVE_INVOKER: ToolInvoker | None = None


@contextmanager
def bind_tool_invoker(invoker: ToolInvoker) -> Iterator[None]:
    """Bind the provided tool invoker for the duration of the context."""
    global _ACTIVE_INVOKER
    if _ACTIVE_INVOKER is not None:
        raise RuntimeError("a tool invoker is already bound")
    _ACTIVE_INVOKER = invoker
    try:
        yield
    finally:
        _ACTIVE_INVOKER = None


def reset_tool_invoker() -> None:
    """Clear any bound tool invoker."""
    global _ACTIVE_INVOKER
    _ACTIVE_INVOKER = None


def _current_tool_invoker() -> ToolInvoker:
    invoker = _ACTIVE_INVOKER
    if invoker is None:
        raise RuntimeError("no tool invoker bound in this context")
    return invoker


__all__ = ["ToolInvoker", "bind_tool_invoker", "reset_tool_invoker"]
