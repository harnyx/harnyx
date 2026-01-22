"""Verdict scales and helpers."""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_VERDICT_VALUE_SETS = (
    frozenset({-1, 1}),
    frozenset({-1, 0, 1}),
    frozenset({1, 2, 3, 4, 5}),
)


@dataclass(frozen=True, slots=True)
class VerdictOption:
    value: int
    description: str

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("verdict option value must be an integer")
        description = self.description.strip()
        if not description:
            raise ValueError("verdict option description must be non-empty")
        object.__setattr__(self, "description", description)


@dataclass(frozen=True, slots=True)
class VerdictOptions:
    options: tuple[VerdictOption, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("verdict_options must not be empty")
        if len(self.options) < 2:
            raise ValueError("verdict_options must contain at least 2 options")

        normalized: list[VerdictOption] = []
        seen: set[int] = set()
        for entry in self.options:
            if entry.value in seen:
                raise ValueError(f"verdict_options contains duplicate value {entry.value!r}")
            seen.add(entry.value)
            normalized.append(entry)

        values = frozenset(entry.value for entry in normalized)
        if values not in _ALLOWED_VERDICT_VALUE_SETS:
            raise ValueError(
                "verdict_options values must be one of (-1, 1), (-1, 0, 1), "
                "(1, 2, 3, 4, 5)"
            )

        object.__setattr__(self, "options", tuple(normalized))

    def __repr__(self) -> str:
        entries = ", ".join(f"{entry.value}={entry.description}" for entry in self.options)
        return f"VerdictOptions({entries})"

    def validate(self, value: int) -> int:
        allowed = {entry.value for entry in self.options}
        if value not in allowed:
            raise ValueError(f"verdict must be one of {sorted(allowed)}, got {value!r}")
        return value

    def normalize(self, value: int) -> float:
        self.validate(value)
        values = [entry.value for entry in self.options]
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            raise ValueError("invalid verdict option bounds")
        return (value - min_value) / (max_value - min_value)

    def description_for(self, value: int) -> str:
        self.validate(value)
        for entry in self.options:
            if entry.value == value:
                return entry.description
        raise ValueError(f"verdict option missing for value {value!r}")


BINARY_VERDICT_OPTIONS = VerdictOptions(
    options=(
        VerdictOption(value=-1, description="Fail"),
        VerdictOption(value=1, description="Pass"),
    )
)


__all__ = [
    "VerdictOption",
    "VerdictOptions",
    "BINARY_VERDICT_OPTIONS",
]
