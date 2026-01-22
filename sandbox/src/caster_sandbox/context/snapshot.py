"""Read-only context snapshot primitives used by sandboxed agents."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from caster_sandbox.json_types import JsonValue


class ContextSnapshot(Mapping[str, JsonValue]):
    """Lightweight read-only mapping exposed to sandboxed miners."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, JsonValue] | None = None) -> None:
        self._data: dict[str, JsonValue] = dict(data or {})

    def __getitem__(self, key: str) -> JsonValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return a shallow copy for serialization."""
        return dict(self._data)

    def __repr__(self) -> str:  # pragma: no cover - trivial representation
        return f"ContextSnapshot({self._data!r})"


__all__ = ["ContextSnapshot"]

