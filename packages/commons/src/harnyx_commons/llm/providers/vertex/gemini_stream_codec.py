"""Gemini streaming accumulation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.llm.providers.vertex.codec import (
    attach_search_metadata,
    build_choices,
    collect_search_queries,
)
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmUsage,
)


class _AccumulatedChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_text_parts: list[str] = Field(default_factory=list)
    tool_calls: dict[int, LlmMessageToolCall] = Field(default_factory=dict)
    reasoning_parts: list[str] = Field(default_factory=list)
    finish_reason: str | None = None

    def merge(self, choice: LlmChoice, *, reasoning_text: str | None = None) -> bool:
        saw_output = False
        for part in choice.message.content:
            if part.text:
                self.content_text_parts.append(part.text)
                saw_output = True
        final_reasoning = reasoning_text if reasoning_text is not None else choice.message.reasoning
        if final_reasoning:
            self.reasoning_parts.append(final_reasoning)
            saw_output = True
        if choice.message.tool_calls:
            self._merge_tool_calls(choice.message.tool_calls)
            saw_output = True
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        return saw_output

    def to_choice(self, index: int) -> LlmChoice:
        content = (
            (LlmMessageContentPart(type="text", text="".join(self.content_text_parts)),)
            if self.content_text_parts
            else ()
        )
        return LlmChoice(
            index=index,
            message=LlmChoiceMessage(
                role="assistant",
                content=content,
                tool_calls=tuple(self.tool_calls[index] for index in sorted(self.tool_calls)) or None,
                reasoning="".join(self.reasoning_parts) or None,
            ),
            finish_reason=self.finish_reason or "stop",
        )

    def _merge_tool_calls(self, tool_calls: Sequence[LlmMessageToolCall]) -> None:
        for index, tool_call in enumerate(tool_calls):
            self.tool_calls[index] = tool_call


class _GeminiRawPartPayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    text: str | None = None
    thought: bool | None = None
    thought_signature: str | None = None


class _GeminiRawContentPayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    parts: list[_GeminiRawPartPayload] | None = None


class _GeminiRawCandidatePayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    content: _GeminiRawContentPayload | None = None
    grounding_metadata: dict[str, Any] | None = None

    def reasoning_text(self) -> str | None:
        content = self.content
        if content is None or not content.parts:
            return None
        fragments = [part.text for part in content.parts if part.thought and part.text]
        return "".join(fragments) or None


class _GeminiRawResponsePayload(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    text: str | None = None
    candidates: list[_GeminiRawCandidatePayload] | None = None


class _GeminiRawCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any] = Field(default_factory=dict)
    content_parts: list[dict[str, object]] = Field(default_factory=list)
    grounding_metadata: dict[str, Any] | None = None

    def merge(self, candidate_payload: _GeminiRawCandidatePayload) -> None:
        merged_payload = candidate_payload.model_dump(mode="python", exclude_none=True)
        self._merge_payload_fields(merged_payload)
        content_payload = candidate_payload.content
        if content_payload is not None:
            self.payload["content"] = content_payload.model_dump(mode="python", exclude_none=True)
            if content_payload.parts:
                self.content_parts.extend(
                    part.model_dump(mode="python", exclude_none=True) for part in content_payload.parts
                )
            if self.content_parts:
                content = dict(self.payload["content"])
                content["parts"] = list(self.content_parts)
                self.payload["content"] = content
        if candidate_payload.grounding_metadata is not None:
            self.grounding_metadata = dict(candidate_payload.grounding_metadata)
            self.payload["grounding_metadata"] = dict(candidate_payload.grounding_metadata)

    def to_payload(self) -> dict[str, Any]:
        return dict(self.payload)

    def _merge_payload_fields(self, candidate_payload: dict[str, Any]) -> None:
        for key, value in candidate_payload.items():
            if key in ("content", "grounding_metadata"):
                continue
            self.payload[key] = value


class GeminiAccumulatedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choices: dict[int, _AccumulatedChoice] = Field(default_factory=dict)
    raw_candidates: dict[int, _GeminiRawCandidate] = Field(default_factory=dict)
    web_search_queries: list[str] = Field(default_factory=list)
    seen_web_search_queries: set[str] = Field(default_factory=set)

    def merge_chunk(self, chunk: Any) -> bool:
        saw_output = False
        raw_payload = _vertex_response_payload(chunk)
        raw_candidates = raw_payload.candidates or []
        for choice in build_choices(chunk):
            state = self.choices.setdefault(choice.index, _AccumulatedChoice())
            raw_reasoning = None
            if choice.index < len(raw_candidates):
                raw_reasoning = raw_candidates[choice.index].reasoning_text()
            if state.merge(choice, reasoning_text=raw_reasoning):
                saw_output = True
        self._merge_raw_payload(raw_payload)
        self._merge_web_search_queries(collect_search_queries(chunk))
        return saw_output

    def to_choices(self) -> tuple[LlmChoice, ...]:
        return tuple(self.choices[index].to_choice(index) for index in sorted(self.choices))

    def metadata(self, usage: LlmUsage) -> tuple[dict[str, Any] | None, LlmUsage]:
        return attach_search_metadata(self.web_search_queries, usage)

    def raw_response_payload(self, latest_response: Any) -> dict[str, Any]:
        payload = _vertex_response_payload(latest_response).model_dump(mode="python", exclude_none=True)
        if self.raw_candidates:
            payload["candidates"] = [
                self.raw_candidates[index].to_payload() for index in sorted(self.raw_candidates)
            ]
        if self.choices and 0 in self.choices and self.choices[0].content_text_parts:
            payload["text"] = "".join(self.choices[0].content_text_parts)
        return payload

    def _merge_raw_payload(self, payload: _GeminiRawResponsePayload) -> None:
        if not payload.candidates:
            return
        for index, candidate in enumerate(payload.candidates):
            state = self.raw_candidates.setdefault(index, _GeminiRawCandidate())
            state.merge(candidate)

    def _merge_web_search_queries(self, queries: list[str]) -> None:
        for query in queries:
            if query in self.seen_web_search_queries:
                continue
            self.seen_web_search_queries.add(query)
            self.web_search_queries.append(query)


def _vertex_response_payload(response: Any) -> _GeminiRawResponsePayload:
    return _GeminiRawResponsePayload.model_validate(response.model_dump(mode="json"))

__all__ = ["GeminiAccumulatedResponse"]
