"""Anthropic-on-Vertex helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from anthropic import APIError as AnthropicAPIError
from anthropic import APIResponseValidationError as AnthropicAPIResponseValidationError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic.types.server_tool_use_block import ServerToolUseBlock
from anthropic.types.text_block import TextBlock

from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmResponse,
    LlmUsage,
)

CLAUDE_WEB_SEARCH_BETA = "web-search-2025-03-05"

# Claude models that support web_search via Anthropic tooling on Vertex
_CLAUDE_WEB_SEARCH_PREFIXES: tuple[str, ...] = (
    "claude-opus-4-5@",
    "claude-opus-4-1@",
    "claude-opus-4@",
    "claude-sonnet-4-5@",
    "claude-sonnet-4@",
    "claude-haiku-4-5@",
)


_ANTHROPIC_PUBLISHER_MODELS_PREFIX = "publishers/anthropic/models/"
_ANTHROPIC_MODELS_PREFIX = "anthropic/models/"


def _extract_claude_model_id(model: str) -> str | None:
    trimmed = model.strip()
    if not trimmed:
        return None

    normalized = trimmed.lower()
    if normalized.startswith("claude-"):
        return trimmed

    for prefix in (_ANTHROPIC_PUBLISHER_MODELS_PREFIX, _ANTHROPIC_MODELS_PREFIX):
        idx = normalized.find(prefix)
        if idx == -1:
            continue
        extracted = trimmed[idx + len(prefix) :].strip()
        return extracted or None

    idx = normalized.find("claude-")
    if idx != -1 and (idx == 0 or normalized[idx - 1] == "/"):
        extracted = trimmed[idx:].strip()
        return extracted or None

    return None


def is_claude_model(model: str) -> bool:
    extracted = _extract_claude_model_id(model)
    return bool(extracted and extracted.lower().startswith("claude-"))


def normalize_claude_model(model: str) -> str:
    extracted = _extract_claude_model_id(model)
    if not extracted or not extracted.lower().startswith("claude-"):
        raise ValueError("expected a Claude model id (e.g. claude-sonnet-4-5@YYYYMMDD)")
    return extracted


def is_claude_web_search_model(model: str) -> bool:
    extracted = _extract_claude_model_id(model)
    if not extracted:
        return False
    normalized = extracted.lower()
    return any(normalized.startswith(prefix) for prefix in _CLAUDE_WEB_SEARCH_PREFIXES)


def classify_anthropic_exception(
    exc: Exception,
    classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
) -> tuple[bool, str]:
    if isinstance(exc, AnthropicAPIError):
        status_code = (
            exc.status_code
            if isinstance(exc, (AnthropicAPIStatusError, AnthropicAPIResponseValidationError))
            else None
        )
        message = str(exc)
        retryable = status_code in {429, 503, 529}
        return retryable, f"anthropic_api_error:{status_code or 'unknown'}:{message}"
    if classify_exception is not None:
        return classify_exception(exc)
    return False, str(exc)


def resolve_anthropic_thinking_budget(
    *, reasoning_effort: str | None, max_tokens: int | None,
) -> int | None:
    if reasoning_effort is None:
        return None

    effort_raw = reasoning_effort.strip()
    if not effort_raw:
        return None

    try:
        budget = int(effort_raw)
    except ValueError:
        raise ValueError("Anthropic thinking budget must be an integer token count") from None

    if budget < 1024:
        raise ValueError("Anthropic thinking budget must be at least 1024 tokens")
    if max_tokens is not None and budget >= max_tokens:
        raise ValueError("Anthropic thinking budget must be less than max_tokens")

    return budget


def build_claude_web_search_tool(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    options = (extra or {}).get("web_search_options", {}) if isinstance(extra, Mapping) else {}
    max_uses = int(options.get("max_uses", 100))
    allowed = options.get("allowed_domains")
    blocked = options.get("blocked_domains")
    user_location = options.get("user_location")

    if allowed and blocked:
        raise ValueError("Provide only one of allowed_domains or blocked_domains for web search")

    config: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }

    optional_fields = {
        "allowed_domains": list(allowed) if allowed else None,
        "blocked_domains": list(blocked) if blocked else None,
        "user_location": user_location,
    }
    config |= {k: v for k, v in optional_fields.items() if v is not None}

    return config


def build_anthropic_response(response: Any) -> LlmResponse:
    """Convert Anthropic Message response to LlmResponse."""
    parts: list[LlmMessageContentPart] = []
    web_search_queries: list[str] = []

    for block in response.content:
        if isinstance(block, TextBlock):
            parts.append(LlmMessageContentPart(type="text", text=block.text))
            continue
        if isinstance(block, ServerToolUseBlock) and block.name == "web_search":
            tool_input = block.input
            if isinstance(tool_input, dict):
                query = tool_input.get("query")
                if isinstance(query, str) and query.strip():
                    web_search_queries.append(query.strip())
            continue

    choice = LlmChoice(
        index=0,
        message=LlmChoiceMessage(
            role="assistant",
            content=tuple(parts),
            tool_calls=None,
        ),
        finish_reason=response.stop_reason or "stop",
    )

    usage_data = response.usage
    usage = LlmUsage(
        prompt_tokens=usage_data.input_tokens,
        completion_tokens=usage_data.output_tokens,
        total_tokens=(
            (usage_data.input_tokens or 0)
            + (usage_data.output_tokens or 0)
        ) or None,
    )

    server_tool_use = usage_data.server_tool_use
    web_search_calls = int(server_tool_use.web_search_requests or 0) if server_tool_use else 0

    usage = usage + LlmUsage(web_search_calls=web_search_calls)
    metadata: dict[str, Any] | None = None
    if web_search_calls:
        metadata = {
            "web_search_calls": web_search_calls,
            "web_search_queries": tuple(web_search_queries),
        }

    return LlmResponse(
        id=response.id,
        choices=(choice,),
        usage=usage,
        metadata=metadata,
        finish_reason=response.stop_reason or "stop",
    )


__all__ = [
    "CLAUDE_WEB_SEARCH_BETA",
    "is_claude_model",
    "normalize_claude_model",
    "is_claude_web_search_model",
    "classify_anthropic_exception",
    "resolve_anthropic_thinking_budget",
    "build_claude_web_search_tool",
    "build_anthropic_response",
]
