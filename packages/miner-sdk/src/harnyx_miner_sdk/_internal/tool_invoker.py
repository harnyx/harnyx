from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol


class ToolInvoker(Protocol):
    """Protocol for host-provided tool invokers."""

    async def invoke(
        self,
        method: str,
        *,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any: ...


_ACTIVE_INVOKER: ContextVar[ToolInvoker | None] = ContextVar("tool_invoker", default=None)


@contextmanager
def bind_tool_invoker(invoker: ToolInvoker) -> Iterator[None]:
    """Bind the provided tool invoker for the duration of the context."""
    if _ACTIVE_INVOKER.get() is not None:
        raise RuntimeError("a tool invoker is already bound")
    token = _ACTIVE_INVOKER.set(invoker)
    try:
        yield
    finally:
        _ACTIVE_INVOKER.reset(token)


def reset_tool_invoker() -> None:
    """Clear any bound tool invoker."""
    _ACTIVE_INVOKER.set(None)


def _current_tool_invoker() -> ToolInvoker:
    invoker = _ACTIVE_INVOKER.get()
    if invoker is None:
        raise RuntimeError("no tool invoker bound in this context")
    return invoker


__all__ = ["ToolInvoker", "bind_tool_invoker", "reset_tool_invoker"]
