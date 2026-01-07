"""LLM provider backed by Vertex AI's Generative AI SDK."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

from anthropic import AnthropicVertex
from google import genai
from google.genai import errors, types

from caster_commons.llm.provider import BaseLlmProvider
from caster_commons.llm.schema import (
    AbstractLlmRequest,
    LlmInputImagePart,
    LlmInputTextPart,
    LlmResponse,
    LlmUsage,
)

from .anthropic import (
    CLAUDE_WEB_SEARCH_BETA,
    build_anthropic_response,
    build_claude_web_search_tool,
    classify_anthropic_exception,
    is_claude_web_search_model,
    resolve_anthropic_thinking_budget,
)
from .codec import (
    attach_search_metadata,
    build_choices,
    extract_usage,
    json_schema_from_model,
    normalize_messages,
    resolve_thinking_config,
    resolve_tool_config,
    serialize_tools,
)
from .credentials import cleanup_credentials_file, prepare_credentials

_API_VERSION = "v1"


class VertexLlmProvider(BaseLlmProvider):
    """Bridges the generic LLM port to Vertex Generative AI models."""

    def __init__(
        self,
        *,
        project: str | None,
        location: str | None,
        timeout: float = 30.0,
        credentials_path: str | None = None,
        service_account_b64: str | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        if not project or not location:
            raise ValueError("Vertex project and location must be configured")
        super().__init__(provider_label="vertex", max_concurrent=max_concurrent)
        self._credentials, self._credentials_file = prepare_credentials(credentials_path, service_account_b64)
        http_timeout = math.ceil(timeout * 1000) if timeout and timeout > 0 else None
        http_options = types.HttpOptions(
            api_version=_API_VERSION,
            timeout=int(http_timeout) if http_timeout is not None else None,
        )
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=self._credentials,
            http_options=http_options,
        )
        self._anthropic_client = AnthropicVertex(
            project_id=project,
            region=location,
            credentials=self._credentials,
        )
        self._logger = logging.getLogger("caster_commons.llm.calls")

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        if request.grounded and is_claude_web_search_model(request.model):
            return await self._call_with_retry(
                request,
                call_coro=lambda: asyncio.to_thread(self._call_claude_anthropic, request),
                verifier=self._verify_response,
                classify_exception=self._classify_anthropic_exception,
            )

        return await self._call_with_retry(
            request,
            call_coro=lambda: asyncio.to_thread(self._call_vertex_with_request, request),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    async def aclose(self) -> None:
        self._client.close()
        self._anthropic_client.close()
        cleanup_credentials_file(self._credentials_file, self._logger)

    def _tools_for(self, request: AbstractLlmRequest) -> tuple[list[Any] | None, types.ToolConfig | None]:
        if request.grounded:
            if is_claude_web_search_model(request.model):
                return None, None
            tools = [types.Tool(google_search=types.GoogleSearch())]
            return tools, None

        serialized_tools = serialize_tools(request.tools)
        tool_config = resolve_tool_config(request.tool_choice, serialized_tools)
        return serialized_tools, tool_config

    def _build_generation_config(
        self,
        request: AbstractLlmRequest,
        system_instruction: str | None,
        tools: list[types.Tool] | None,
        tool_config: types.ToolConfig | None,
    ) -> types.GenerateContentConfig | None:
        if request.grounded and request.output_mode != "text":
            raise ValueError("grounded Vertex requests must use text output")

        config_kwargs: dict[str, Any] = {}

        timeout = request.timeout_seconds
        if timeout is not None:
            http_timeout = math.ceil(timeout * 1000) if timeout > 0 else None
            config_kwargs["http_options"] = types.HttpOptions(
                api_version=_API_VERSION,
                timeout=int(http_timeout) if http_timeout is not None else None,
            )

        if not request.grounded:
            if request.output_mode in {"json_object", "structured"}:
                config_kwargs["response_mime_type"] = "application/json"
            if request.output_mode == "structured" and request.output_schema is not None:
                config_kwargs["response_schema"] = json_schema_from_model(request.output_schema)

        config_kwargs |= {
            k: v
            for k, v in {
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": tool_config,
                "thinking_config": resolve_thinking_config(
                    model=request.model,
                    reasoning_effort=request.reasoning_effort,
                ),
            }.items()
            if v is not None
        }

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    def _call_vertex(
        self,
        request: AbstractLlmRequest,
        contents: list[Any],
        generation_config: types.GenerateContentConfig | None,
    ) -> LlmResponse:
        response = self._client.models.generate_content(
            model=request.model,
            contents=contents,
            config=generation_config,
        )
        choices = build_choices(response)
        primary_finish_reason = choices[0].finish_reason if choices else None
        usage = extract_usage(response.usage_metadata) or LlmUsage()
        response_id = _as_str(response.response_id)

        metadata, usage = attach_search_metadata(response, usage)
        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("raw_response", response)

        return LlmResponse(
            id=response_id,
            choices=choices,
            usage=usage,
            metadata=combined_metadata,
            finish_reason=primary_finish_reason,
        )

    def _call_vertex_with_request(self, request: AbstractLlmRequest) -> LlmResponse:
        system_instruction, contents = normalize_messages(request.messages)
        tools, tool_config = self._tools_for(request)
        generation_config = self._build_generation_config(
            request,
            system_instruction,
            tools,
            tool_config,
        )
        return self._call_vertex(request, contents, generation_config)

    def _call_claude_anthropic(self, request: AbstractLlmRequest) -> LlmResponse:
        system_content, messages = _anthropic_messages_from_request(request)
        tools = [build_claude_web_search_tool(request.extra)]
        max_tokens = request.max_output_tokens or 1024

        base_kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": tools,
            "extra_headers": {"anthropic-beta": CLAUDE_WEB_SEARCH_BETA},
        }

        thinking_budget = resolve_anthropic_thinking_budget(
            reasoning_effort=request.reasoning_effort,
            max_tokens=max_tokens,
        )
        optional_kwargs: dict[str, Any] = {
            "system": system_content,
            "temperature": request.temperature,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            if thinking_budget is not None
            else None,
        }

        kwargs = base_kwargs | {k: v for k, v in optional_kwargs.items() if v is not None}

        response = self._anthropic_client.messages.create(timeout=request.timeout_seconds or 900, **kwargs)
        llm_response = build_anthropic_response(response)

        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", response)

        usage_with_calls = llm_response.usage
        if llm_response.usage.web_search_calls in (None, 0):
            raw_queries = metadata.get("web_search_queries", ())
            web_search_calls = len(raw_queries) if isinstance(raw_queries, (list, tuple)) else 0
            usage_with_calls = llm_response.usage + LlmUsage(web_search_calls=web_search_calls)

        return LlmResponse(
            id=llm_response.id,
            choices=llm_response.choices,
            usage=usage_with_calls,
            metadata=metadata,
            finish_reason=llm_response.finish_reason,
        )

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text:
            return False, True, "empty_output"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        if isinstance(exc, errors.APIError):
            code = exc.code
            message = str(exc)
            retryable = code in { 429, 503, 529 }
            return retryable, f"api_error:{code}:{message}"
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    @staticmethod
    def _classify_anthropic_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        return classify_anthropic_exception(exc, classify_exception)

    def _as_mapping(self, response: LlmResponse) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], response.to_payload())


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _anthropic_messages_from_request(
    request: AbstractLlmRequest,
) -> tuple[str | None, list[dict[str, Any]]]:
    system_content: str | None = None
    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        if any(isinstance(part, LlmInputImagePart) for part in msg.content):
            raise ValueError("Claude-on-Vertex requests do not support input_image content parts")
        text = "\n".join(part.text for part in msg.content if isinstance(part, LlmInputTextPart))
        if msg.role == "system":
            system_content = text
        else:
            messages.append({"role": msg.role, "content": text})
    return system_content, messages


__all__ = ["VertexLlmProvider"]
