"""Private provider evidence with reversible references to already stored text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict

from harnyx_commons.json_types import JsonValue

Path = tuple[str | int, ...]
_MIN_REFERENCE_LENGTH = 128
_JSONB_UNSAFE = re.compile("[\\x00\\ud800-\\udfff]")
_EXCHANGES: ContextVar[list[JsonValue] | None] = ContextVar("provider_exchanges", default=None)


@contextmanager
def capture_provider_exchanges() -> Iterator[list[JsonValue]]:
    exchanges: list[JsonValue] = []
    token = _EXCHANGES.set(exchanges)
    try:
        yield exchanges
    finally:
        _EXCHANGES.reset(token)


def record_provider_exchange(exchange: JsonValue) -> None:
    if (exchanges := _EXCHANGES.get()) is not None:
        exchanges.append(exchange)


class EvidenceTextReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    source: Literal["base", "evidence"]
    source_path: Path
    start: int
    end: int
    prefix: str = ""
    suffix: str = ""


class ProviderEvidence(BaseModel):
    """A private snapshot; references never point outside its owning record."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    payload: JsonValue
    text_references: tuple[EvidenceTextReference, ...] = ()
    escaped_json: str | None = None

    @classmethod
    def capture(cls, payload: JsonValue, *, base: JsonValue = None) -> ProviderEvidence:
        # PostgreSQL cannot represent NUL or isolated surrogates in JSONB.
        # Preserve the exceptional snapshot as escaped JSON, without offsets
        # into a base that storage sanitization would change.
        serialized = json.dumps(payload, ensure_ascii=False)
        serialized_base = json.dumps(base, ensure_ascii=False)
        if any(_JSONB_UNSAFE.search(value) or "\\u0000" in value for value in (serialized, serialized_base)):
            return cls(payload=None, escaped_json=json.dumps(payload, ensure_ascii=True))
        payload = json.loads(serialized)
        base = json.loads(serialized_base)
        stored = deepcopy(payload)
        sources: list[tuple[Literal["base", "evidence"], Path, str]] = [
            ("base", path, text) for path, text in _strings(base)
        ]
        references: list[EvidenceTextReference] = []
        for path, text in sorted(_strings(payload), key=lambda item: len(item[1]), reverse=True):
            if len(text) < _MIN_REFERENCE_LENGTH:
                continue
            # Parallel strips excerpt whitespace before joining. Preserve it while
            # referring to the matching slice of the stored snippet/page content.
            core = text.strip()
            prefix_length = len(text) - len(text.lstrip())
            if len(core) < _MIN_REFERENCE_LENGTH:
                continue
            for source, source_path, source_text in sources:
                start = source_text.find(core)
                if start < 0:
                    continue
                references.append(EvidenceTextReference(
                    path=path, source=source, source_path=source_path,
                    start=start, end=start + len(core),
                    prefix=text[:prefix_length], suffix=text[prefix_length + len(core):],
                ))
                stored = _replace(stored, path, None)
                break
            else:
                sources.append(("evidence", path, text))
        return cls(payload=stored, text_references=tuple(references))

    def restore(self, *, base: JsonValue = None) -> JsonValue:
        if self.escaped_json is not None:
            return json.loads(self.escaped_json)
        restored = deepcopy(self.payload)
        for reference in self.text_references:
            source = base if reference.source == "base" else restored
            text = _at(source, reference.source_path)
            if not isinstance(text, str) or not 0 <= reference.start <= reference.end <= len(text):
                raise ValueError("provider evidence text reference is invalid")
            restored = _replace(
                restored, reference.path,
                reference.prefix + text[reference.start:reference.end] + reference.suffix,
            )
        return restored


def _strings(value: JsonValue, path: Path = ()) -> Iterator[tuple[Path, str]]:
    if isinstance(value, str):
        if len(value) >= _MIN_REFERENCE_LENGTH:
            yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))


def _at(value: JsonValue, path: Path) -> JsonValue:
    for key in path:
        if isinstance(value, dict) and isinstance(key, str):
            value = value[key]
        elif isinstance(value, list) and isinstance(key, int):
            value = value[key]
        else:
            raise ValueError("provider evidence path is invalid")
    return value


def _replace(value: JsonValue, path: Path, replacement: JsonValue) -> JsonValue:
    if not path:
        return replacement
    parent = _at(value, path[:-1])
    key = path[-1]
    if isinstance(parent, dict) and isinstance(key, str):
        parent[key] = replacement
    elif isinstance(parent, list) and isinstance(key, int):
        parent[key] = replacement
    else:
        raise ValueError("provider evidence destination is invalid")
    return value
