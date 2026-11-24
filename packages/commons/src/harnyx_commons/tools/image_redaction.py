"""Keep request-local image capabilities out of receipts and output streams."""

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import TypeVar, cast

from pydantic import BaseModel

from harnyx_miner_sdk.tools.llm_chat_stream_models import LlmChatChunk, LlmChatOutputSink, TextUpdate, ToolUpdate

T = TypeVar("T")


class ImageUrlRedactor:
    def __init__(self, urls: Sequence[str]) -> None:
        self._urls = tuple(urls)
        self._max_length = max(map(len, self._urls), default=0)
        self._pending: dict[tuple[int, str, int], str] = {}
        self._sink: LlmChatOutputSink | None = None
        self._details: dict[int, list[dict[str, object]]] = {}

    def text(self, value: str) -> str:
        for url in self._urls:
            value = value.replace(url, "[image]")
        return value

    def value(self, value: T) -> T:
        """Preserve typed accounting facts while removing signed URLs from their strings."""
        if isinstance(value, str):
            return cast(T, self.text(value))
        if isinstance(value, Mapping):
            return cast(T, {
                self.value(key): self._reasoning_details(item)
                if key == "reasoning_details" and isinstance(item, (list, tuple)) else self.value(item)
                for key, item in value.items()
            })
        if isinstance(value, list):
            return cast(T, [self.value(item) for item in value])
        if isinstance(value, tuple):
            return cast(T, tuple(self.value(item) for item in value))
        if isinstance(value, BaseModel):
            return cast(
                T,
                value.model_copy(update={name: self.value(getattr(value, name)) for name in type(value).model_fields}),
            )
        if is_dataclass(value) and not isinstance(value, type):
            return cast(
                T,
                replace(
                    value,
                    **{field.name: self.value(getattr(value, field.name)) for field in fields(value) if field.init},
                ),
            )
        return value

    def _reasoning_details(self, details: Sequence[object]) -> list[object]:
        """Redact URLs spanning structured reasoning deltas without changing their order.

        Unlike TextUpdate fields these are append-only dictionaries with no replace
        operation. Delay them until the choice ends so a URL split across fragments
        can be removed while retaining every dictionary and its replay metadata.
        """
        cleaned = [self.value(item) for item in details]
        groups: dict[tuple[str, str, tuple[str, ...]], list[tuple[dict[str, object], str, str]]] = {}

        def collect(node: object, kind: str, index: str, path: tuple[str, ...] = ()) -> None:
            if isinstance(node, dict):
                mapping = cast(dict[str, object], node)
                for key, item in mapping.items():
                    if isinstance(item, str):
                        groups.setdefault((kind, index, (*path, key)), []).append((mapping, key, item))
                    else:
                        collect(item, kind, index, (*path, key))
            elif isinstance(node, list):
                for offset, item in enumerate(node):
                    collect(item, kind, index, (*path, str(offset)))

        for detail in cleaned:
            if isinstance(detail, dict):
                mapping = cast(dict[str, object], detail)
                collect(mapping, str(mapping.get("type", "")), str(mapping.get("index", "")))
        for fragments in groups.values():
            joined = "".join(fragment for _, _, fragment in fragments)
            matches: list[tuple[int, int]] = []
            for url in self._urls:
                start = joined.find(url)
                while start >= 0:
                    matches.append((start, start + len(url)))
                    start = joined.find(url, start + len(url))
            if not matches:
                continue
            matches.sort()
            offset = 0
            for node, key, fragment in fragments:
                end = offset + len(fragment)
                cursor = offset
                pieces: list[str] = []
                for start, stop in matches:
                    if stop <= cursor or start >= end:
                        continue
                    pieces.append(joined[cursor:max(cursor, start)])
                    if start >= offset:
                        pieces.append("[image]")
                    cursor = min(end, stop)
                pieces.append(joined[cursor:end])
                node[key] = "".join(pieces)
                offset = end
        return cleaned

    def _update(self, key: tuple[int, str, int], update: TextUpdate, *, final: bool) -> TextUpdate:
        previous = "" if update.operation == "replace" else self._pending.pop(key, "")
        text = self.text(previous + update.value)
        cut = len(text)
        if not final:
            # Retain only a suffix that could still become a complete signed URL.
            for start in range(max(0, len(text) - self._max_length + 1), len(text)):
                if any(url.startswith(text[start:]) for url in self._urls):
                    cut = start
                    break
        self._pending[key] = text[cut:]
        return TextUpdate(operation=update.operation, value=text[:cut])

    def wrap(self, sink: LlmChatOutputSink) -> LlmChatOutputSink:
        self._sink = sink

        async def emit(chunk: LlmChatChunk) -> None:
            choice = chunk.choice_index
            final = chunk.finish_reason is not None
            changes: dict[str, object] = {}
            for name in ("text", "reasoning"):
                key = (choice, name, -1)
                update = getattr(chunk, name)
                if update is not None or (final and self._pending.get(key)):
                    changes[name] = self._update(key, update or TextUpdate(operation="append", value=""), final=final)
            calls = {call.index: call for call in chunk.tool_calls or ()}
            if final:
                for owner, name, index in self._pending:
                    if owner == choice and index >= 0 and self._pending[(owner, name, index)]:
                        calls.setdefault(index, ToolUpdate(index=index))
            cleaned_calls: list[ToolUpdate] = []
            for index, call in calls.items():
                updates: dict[str, object] = {"type": self.value(call.type)}
                for name in ("id", "name", "arguments"):
                    key = (choice, name, index)
                    update = getattr(call, name)
                    if update is not None or (final and self._pending.get(key)):
                        updates[name] = self._update(
                            key, update or TextUpdate(operation="append", value=""), final=final
                        )
                cleaned_calls.append(call.model_copy(update=updates))
            if chunk.tool_calls is not None or cleaned_calls:
                changes["tool_calls"] = cleaned_calls
            if chunk.reasoning_details:
                self._details.setdefault(choice, []).extend(cast(list[dict[str, object]], chunk.reasoning_details))
            changes["reasoning_details"] = (
                self._reasoning_details(self._details.pop(choice, [])) or None if final else None
            )
            changes["finish_reason"] = self.value(chunk.finish_reason)
            await sink(chunk.model_copy(update=changes))

        return emit

    async def finish(self) -> None:
        """Flush harmless suffixes when a provider ends without a finish-reason chunk."""
        if self._sink is None:
            return
        for choice in {key[0] for key, value in self._pending.items() if value} | set(self._details):
            changes: dict[str, object] = {
                "reasoning_details": self._reasoning_details(self._details.pop(choice, [])) or None
            }
            tools: dict[int, dict[str, object]] = {}
            for (owner, field, index), value in self._pending.items():
                if owner != choice or not value:
                    continue
                update = TextUpdate(operation="append", value=value)
                if index < 0:
                    changes[field] = update
                else:
                    tools.setdefault(index, {})[field] = update
            if tools:
                changes["tool_calls"] = [
                    ToolUpdate.model_validate({"index": index, **updates}) for index, updates in tools.items()
                ]
            await self._sink(LlmChatChunk.model_validate({"choice_index": choice, **changes}))
        self._pending.clear()
