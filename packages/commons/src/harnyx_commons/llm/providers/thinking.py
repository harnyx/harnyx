"""Provider-side thinking controls for OpenAI-compatible template routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from harnyx_commons.llm.schema import LlmThinkingConfig, ReasoningEffort
from harnyx_commons.llm.tool_models import ToolModelThinkingCapability, tool_model_thinking_capability

_TYPED_REASONING_EFFORTS = frozenset(("low", "medium", "high"))


@dataclass(frozen=True, slots=True)
class ResolvedTemplateThinking:
    capability: ToolModelThinkingCapability
    thinking: LlmThinkingConfig

    def chat_template_kwargs(self) -> dict[str, bool]:
        return self.capability.chat_template_kwargs(enabled=self.thinking.enabled)


def resolve_template_thinking(
    *,
    canonical_model: str | None,
    provider_name: str,
    request_thinking: LlmThinkingConfig | None,
    reasoning_effort: str | None,
) -> ResolvedTemplateThinking | None:
    """Resolve request thinking to a provider template capability when one exists."""
    capability = tool_model_thinking_capability(canonical_model, provider_name=provider_name)
    if capability is None:
        return None
    if request_thinking is not None:
        return ResolvedTemplateThinking(capability=capability, thinking=request_thinking)
    named_effort = _named_reasoning_effort(reasoning_effort)
    if named_effort is None:
        return None
    return ResolvedTemplateThinking(
        capability=capability,
        thinking=LlmThinkingConfig(enabled=True, effort=named_effort),
    )


def _named_reasoning_effort(reasoning_effort: str | None) -> ReasoningEffort | None:
    if reasoning_effort is None:
        return None
    normalized = reasoning_effort.strip().lower()
    if normalized not in _TYPED_REASONING_EFFORTS:
        return None
    return cast(ReasoningEffort, normalized)


__all__ = [
    "ResolvedTemplateThinking",
    "resolve_template_thinking",
]
