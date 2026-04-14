"""LLM provider backed by Vertex AI's Generative AI SDK."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from typing import Any

import google.auth
import httpx
from anthropic import AsyncAnthropicVertex
from google import genai
from google.auth.credentials import Credentials as GoogleCredentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import errors, types

from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiStreamError,
    OpenAiStreamState,
    iter_openai_sse_events,
    normalize_openai_text_fragments,
)
from harnyx_commons.llm.schema import (
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
    is_claude_model,
    is_claude_web_search_model,
    normalize_claude_model,
    resolve_anthropic_thinking_budget,
)
from .codec import (
    _VertexMaasChatRequest,
    _VertexMaasChatResponse,
    extract_usage,
    json_schema_from_model,
    normalize_messages,
    resolve_thinking_config,
    resolve_tool_config,
    serialize_provider_native_tools,
    serialize_tools,
    supports_thinking_config,
)
from .credentials import cleanup_credentials_file, prepare_credentials
from .gemini_stream_codec import GeminiAccumulatedResponse

# v1beta1 exposes grounding metadata (e.g. retrievalQueries) that we use for
# richer tool attribution in observability.
_API_VERSION = "v1beta1"
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_VERTEX_MAAS_OPENAI_CHAT_MODELS = frozenset(
    {
        "publishers/openai/models/gpt-oss-20b-maas",
        "publishers/openai/models/gpt-oss-120b-maas",
        "openai/gpt-oss-20b-tee",
        "openai/gpt-oss-120b-tee",
    }
)


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
        self._project = project
        self._location = location
        self._credentials, self._credentials_file = prepare_credentials(credentials_path, service_account_b64)
        self._http_credentials: GoogleCredentials | None = self._credentials
        http_timeout = math.ceil(timeout * 1000) if timeout and timeout > 0 else None
        http_options = types.HttpOptions(
            api_version=_API_VERSION,
            timeout=int(http_timeout) if http_timeout is not None else None,
        )
        self._genai_client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=self._credentials,
            http_options=http_options,
        )
        self._genai_async_client = self._genai_client.aio
        self._anthropic_client = AsyncAnthropicVertex(
            project_id=project,
            region=location,
            credentials=self._credentials,
        )
        self._http_client = httpx.AsyncClient(timeout=timeout)
        self._logger = logging.getLogger("harnyx_commons.llm.calls")

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        if is_claude_model(request.model):
            return await self._call_with_retry(
                request,
                call_coro=lambda current_request: self._call_claude_anthropic(current_request),
                verifier=self._verify_response,
                classify_exception=self._classify_anthropic_exception,
            )

        return await self._call_with_retry(
            request,
            call_coro=lambda current_request: self._call_vertex_with_request(current_request),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()
        await self._genai_async_client.aclose()
        self._genai_client.close()
        await self._anthropic_client.close()
        cleanup_credentials_file(self._credentials_file, self._logger)

    def _tools_for(self, request: AbstractLlmRequest) -> tuple[list[Any] | None, types.ToolConfig | None]:
        if request.grounded:
            if is_claude_web_search_model(request.model):
                return None, None
            tools = [types.Tool(google_search=types.GoogleSearch())]
            tools.extend(serialize_provider_native_tools(request.tools))
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

        thinking_config = (
            resolve_thinking_config(
                model=request.model,
                reasoning_effort=request.reasoning_effort,
            )
            if supports_thinking_config(model=request.model)
            else None
        )

        config_kwargs |= {
            k: v
            for k, v in {
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                "system_instruction": system_instruction,
                "tools": tools,
                "tool_config": tool_config,
                "thinking_config": thinking_config,
            }.items()
            if v is not None
        }

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    async def _call_vertex(
        self,
        request: AbstractLlmRequest,
        contents: list[Any],
        generation_config: types.GenerateContentConfig | None,
    ) -> LlmResponse:
        started_at = time.perf_counter()
        accumulated = GeminiAccumulatedResponse()
        latest_response: Any | None = None
        ttft_ms: float | None = None
        stream = await self._genai_async_client.models.generate_content_stream(
            model=request.model,
            contents=contents,
            config=generation_config,
        )
        async for chunk in stream:
            latest_response = chunk
            if _merge_gemini_chunk(accumulated, chunk) and ttft_ms is None:
                ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
        if latest_response is None:
            raise RuntimeError("vertex streaming generation returned no response chunks")

        choices = accumulated.to_choices()
        primary_finish_reason = choices[0].finish_reason if choices else None
        usage = extract_usage(latest_response.usage_metadata) or LlmUsage()
        response_id = _as_str(latest_response.response_id)

        metadata, usage = accumulated.metadata(usage)
        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("raw_response", accumulated.raw_response_payload(latest_response))
        self._log_stream_ttft(branch="gemini", model=request.model, response_id=response_id, ttft_ms=ttft_ms)

        return LlmResponse(
            id=response_id,
            choices=choices,
            usage=usage,
            metadata=combined_metadata,
            finish_reason=primary_finish_reason,
        )

    async def _call_vertex_with_request(self, request: AbstractLlmRequest) -> LlmResponse:
        if _should_use_vertex_maas_openai_chat(request):
            return await self._call_vertex_maas_chat_completions(request)
        system_instruction, contents = normalize_messages(request.messages)
        tools, tool_config = self._tools_for(request)
        generation_config = self._build_generation_config(
            request,
            system_instruction,
            tools,
            tool_config,
        )
        return await self._call_vertex(request, contents, generation_config)

    async def _call_vertex_maas_chat_completions(self, request: AbstractLlmRequest) -> LlmResponse:
        payload = _VertexMaasChatRequest.from_request(request)
        access_token = await self._vertex_maas_access_token()
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            "json": payload.model_dump(mode="json", exclude_none=True),
        }
        if request.timeout_seconds is not None:
            request_kwargs["timeout"] = request.timeout_seconds

        started_at = time.perf_counter()
        state = OpenAiStreamState()
        ttft_ms: float | None = None
        async with self._http_client.stream(
            "POST",
            _vertex_maas_chat_completions_url(project=self._project, location=self._location),
            **request_kwargs,
        ) as response:
            response.raise_for_status()
            async for event in iter_openai_sse_events(
                response,
                invalid_data_message="vertex maas streaming returned non-JSON SSE data",
                invalid_event_message="vertex maas streaming SSE event must be a JSON object",
            ):
                if state.merge_event(
                    event,
                    reasoning_keys=("reasoning_content", "reasoning"),
                    normalize_content_fragment=_vertex_stream_text_fragments,
                    normalize_reasoning_fragment=_vertex_stream_text_fragments,
                ):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response_body = _VertexMaasChatResponse.from_stream_state(state)
        llm_response = response_body.to_llm_response()
        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", response_body.raw_payload())
        self._log_stream_ttft(
            branch="vertex_maas_openai",
            model=request.model,
            response_id=llm_response.id,
            ttft_ms=ttft_ms,
        )
        return LlmResponse(
            id=llm_response.id,
            choices=llm_response.choices,
            usage=llm_response.usage,
            metadata=metadata,
            finish_reason=llm_response.finish_reason,
        )

    async def _vertex_maas_access_token(self) -> str:
        credentials = await self._vertex_maas_credentials()
        request = GoogleAuthRequest()
        await asyncio.to_thread(credentials.refresh, request)
        token = credentials.token
        if not token:
            raise RuntimeError("vertex maas credentials refresh returned no access token")
        return token

    async def _vertex_maas_credentials(self) -> GoogleCredentials:
        credentials = self._http_credentials
        if credentials is not None:
            return credentials
        resolved_credentials, _project_id = await asyncio.to_thread(
            google.auth.default,
            scopes=(_CLOUD_PLATFORM_SCOPE,),
        )
        self._http_credentials = resolved_credentials
        return resolved_credentials

    async def _call_claude_anthropic(self, request: AbstractLlmRequest) -> LlmResponse:
        system_content, messages = _anthropic_messages_from_request(request)
        model = normalize_claude_model(request.model)

        if request.output_mode != "text":
            raise ValueError("Claude-on-Vertex requests must use text output")
        if request.tools:
            raise ValueError("Claude-on-Vertex requests do not support tool calls")

        max_tokens = request.max_output_tokens or 1024
        tools: list[dict[str, Any]] | None = None
        extra_headers: dict[str, str] | None = None

        if request.grounded:
            if not is_claude_web_search_model(model):
                raise ValueError("grounded Claude requests require a Claude web_search model")
            tools = [build_claude_web_search_tool(request.extra)]
            extra_headers = {"anthropic-beta": CLAUDE_WEB_SEARCH_BETA}

        base_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        thinking_budget = resolve_anthropic_thinking_budget(
            reasoning_effort=request.reasoning_effort,
            max_tokens=max_tokens,
        )
        optional_kwargs: dict[str, Any] = {
            "system": system_content,
            "temperature": request.temperature,
            "tools": tools,
            "extra_headers": extra_headers,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            if thinking_budget is not None
            else None,
        }

        kwargs = base_kwargs | {k: v for k, v in optional_kwargs.items() if v is not None}

        started_at = time.perf_counter()
        ttft_ms: float | None = None
        async with self._anthropic_client.messages.stream(timeout=request.timeout_seconds or 900, **kwargs) as stream:
            async for text in stream.text_stream:
                if text and ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
            response = await stream.get_final_message()
        llm_response = build_anthropic_response(response)

        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", _raw_response_payload(response))
        self._log_stream_ttft(branch="claude", model=request.model, response_id=llm_response.id, ttft_ms=ttft_ms)

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
        if not resp.raw_text and not resp.tool_calls:
            return False, True, "empty_output"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        match exc:
            case httpx.HTTPStatusError():
                status = exc.response.status_code if exc.response is not None else None
                retryable = status is not None and (status == 429 or status >= 500)
                return retryable, f"http_{status}"
            case httpx.HTTPError():
                return True, exc.__class__.__name__
            case errors.APIError():
                code = exc.code
                message = str(exc)
                retryable = code in {429, 503, 529}
                return retryable, f"api_error:{code}:{message}"
            case OpenAiStreamError():
                return exc.retryable, exc.reason
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    @staticmethod
    def _classify_anthropic_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        return classify_anthropic_exception(exc, classify_exception)

    def _log_stream_ttft(self, *, branch: str, model: str, response_id: str, ttft_ms: float | None) -> None:
        if ttft_ms is None:
            return
        self._logger.debug(
            "llm.vertex.stream.ttft",
            extra={
                "data": {
                    "provider": self._provider_label,
                    "branch": branch,
                    "model": model,
                    "response_id": response_id,
                    "ttft_ms": ttft_ms,
                }
            },
        )


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def _raw_response_payload(response: Any) -> dict[str, Any]:
    return response.model_dump(mode="json")


def _merge_gemini_chunk(accumulated: GeminiAccumulatedResponse, chunk: Any) -> bool:
    return accumulated.merge_chunk(chunk)


def _vertex_stream_text_fragments(value: object) -> tuple[str, ...]:
    return normalize_openai_text_fragments(value, multipart_joiner="\n\n")


def _should_use_vertex_maas_openai_chat(request: AbstractLlmRequest) -> bool:
    provider = (request.provider or "").strip().lower()
    model = request.model.strip().lower()
    return provider == "vertex-maas" and model in _VERTEX_MAAS_OPENAI_CHAT_MODELS


def _vertex_maas_chat_completions_url(*, project: str, location: str) -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
    )


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
