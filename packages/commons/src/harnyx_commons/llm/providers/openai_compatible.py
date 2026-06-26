"""Generic OpenAI-compatible chat completions LLM provider."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token, service_account
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.config.llm import (
    OpenAiCompatibleBearerTokenEnvAuthConfig,
    OpenAiCompatibleEndpointConfig,
    OpenAiCompatibleGoogleIdTokenAuthConfig,
    OpenAiCompatibleNoAuthConfig,
)
from harnyx_commons.llm.adapter import canonical_model_for_provider_model
from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.provider_types import custom_openai_compatible_target
from harnyx_commons.llm.providers.openai_chat_codec import OpenAiChatRequestParts
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiChoiceState,
    OpenAiStreamError,
    OpenAiStreamState,
    OpenAiToolCall,
    iter_openai_sse_events,
    normalize_openai_reasoning_fragments,
)
from harnyx_commons.llm.providers.thinking import resolve_template_thinking
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmMessageToolCall,
    LlmResponse,
    LlmUsage,
)


class OpenAiCompatibleLlmProvider(BaseLlmProvider):
    def __init__(
        self,
        *,
        endpoint: OpenAiCompatibleEndpointConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        provider_label = custom_openai_compatible_target(endpoint.id)
        super().__init__(provider_label=provider_label, max_concurrent=endpoint.max_concurrent)
        self._endpoint = endpoint
        self._authenticator = _OpenAiCompatibleAuthenticator(endpoint.auth)
        self._chat_completions_url = f"{str(endpoint.base_url).rstrip('/')}/chat/completions"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=str(endpoint.base_url).rstrip("/"),
            timeout=endpoint.timeout_seconds or 300.0,
        )

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        return await self._call_with_retry(
            request,
            call_coro=lambda current_request: self._request_chat(
                self._build_request(current_request),
                timeout_seconds=current_request.timeout_seconds,
            ),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
            policy=request.retry_policy,
        )

    def _build_request(self, request: AbstractLlmRequest) -> _OpenAiCompatibleChatRequest:
        return _OpenAiCompatibleChatRequest.from_request(
            request,
            provider_name=custom_openai_compatible_target(self._endpoint.id),
        )

    async def _request_chat(
        self,
        payload: _OpenAiCompatibleChatRequest,
        *,
        timeout_seconds: float | None,
    ) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "json": payload.model_dump(mode="json", exclude_none=True),
            "headers": await self._authenticator.headers(),
        }
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        body, ttft_ms = await self._stream_chat_completions(**request_kwargs)
        response = body.to_llm_response()
        metadata = dict(response.metadata or {})
        metadata.setdefault("raw_response", body.model_dump(mode="python", exclude_none=True))
        if ttft_ms is not None:
            metadata.setdefault("ttft_ms", ttft_ms)
        return LlmResponse(
            id=response.id,
            choices=response.choices,
            usage=response.usage,
            metadata=metadata,
            finish_reason=response.finish_reason,
        )

    async def _stream_chat_completions(
        self,
        **request_kwargs: Any,
    ) -> tuple[_OpenAiCompatibleChatResponse, float | None]:
        started_at = time.perf_counter()
        state = OpenAiStreamState()
        ttft_ms: float | None = None
        async with self._client.stream("POST", self._chat_completions_url, **request_kwargs) as response:
            if response.is_error:
                await response.aread()
            response.raise_for_status()
            async for event in iter_openai_sse_events(
                response,
                invalid_data_message="OpenAI-compatible chat completions returned non-JSON SSE data",
                invalid_event_message="OpenAI-compatible chat completions SSE event must be a JSON object",
            ):
                if state.merge_event(
                    event,
                    reasoning_keys=("reasoning", "reasoning_content", "reasoning_details"),
                    normalize_reasoning_fragment=normalize_openai_reasoning_fragments,
                ):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return _OpenAiCompatibleChatResponse.from_stream_state(state), ttft_ms

    @staticmethod
    def _verify_response(response: LlmResponse) -> tuple[bool, bool, str | None]:
        if not response.choices:
            return False, True, "empty_choices"
        if not response.raw_text and not response.tool_calls:
            return False, True, "empty_output"
        for call in _iter_tool_calls(response):
            if not _is_valid_json(call.arguments):
                return False, True, "tool_call_args_invalid_json"
        return True, False, None

    @staticmethod
    def _classify_exception(
        exc: Exception,
        classify_exception: Callable[[Exception], tuple[bool, str]] | None = None,
    ) -> tuple[bool, str]:
        match exc:
            case httpx.HTTPStatusError():
                status = exc.response.status_code if exc.response else None
                retryable = status is not None and (status == 429 or status >= 500)
                detail = _summarize_response(exc.response) if exc.response is not None else ""
                if detail:
                    return retryable, f"http_{status}: {detail}"
                return retryable, f"http_{status}"
            case httpx.HTTPError():
                return True, exc.__class__.__name__
            case OpenAiStreamError():
                return exc.retryable, exc.reason
        if classify_exception is not None:
            return classify_exception(exc)
        return False, str(exc)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class _OpenAiCompatibleChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str
    messages: list[dict[str, Any]]
    stream: bool = True
    stream_options: dict[str, bool] = Field(default_factory=lambda: {"include_usage": True})
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    include: list[str] | None = None
    response_format: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_request(cls, request: AbstractLlmRequest, *, provider_name: str) -> _OpenAiCompatibleChatRequest:
        if request.grounded:
            raise ValueError("grounded mode is not supported for OpenAI-compatible provider")
        request_parts = OpenAiChatRequestParts.from_request(
            request,
            provider_name=provider_name,
            image_error_message="OpenAI-compatible provider does not support image content parts",
            tool_mix_error_message="OpenAI-compatible input_tool_result messages cannot mix text parts",
            tool_count_error_message="OpenAI-compatible input_tool_result messages must include exactly one part",
        )
        payload = cls(
            model=request.model,
            messages=[message.model_dump(mode="python", exclude_none=True) for message in request_parts.messages],
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            tools=(
                [tool.model_dump(mode="python", exclude_none=True) for tool in request_parts.tools]
                if request_parts.tools
                else None
            ),
            tool_choice=request_parts.tool_choice,
            include=request_parts.include,
            response_format=(
                request_parts.response_format.model_dump(mode="python", exclude_none=True)
                if request_parts.response_format is not None
                else None
            ),
        )
        payload = _apply_openai_compatible_thinking(payload, request, provider_name=provider_name)
        if request.extra:
            payload = payload.model_copy(update=dict(request.extra))
        return payload.model_copy(update={"stream": True})


def _apply_openai_compatible_thinking(
    payload: _OpenAiCompatibleChatRequest,
    request: AbstractLlmRequest,
    *,
    provider_name: str,
) -> _OpenAiCompatibleChatRequest:
    canonical_model = canonical_model_for_provider_model(
        provider_name=provider_name,
        model=request.model,
    )
    resolved = resolve_template_thinking(
        canonical_model=canonical_model,
        provider_name=provider_name,
        request_thinking=request.thinking,
        reasoning_effort=request.reasoning_effort,
    )
    if resolved is None:
        return payload
    return _with_chat_template_kwargs(payload, resolved.chat_template_kwargs())


def _with_chat_template_kwargs(
    payload: _OpenAiCompatibleChatRequest,
    updates: dict[str, Any],
) -> _OpenAiCompatibleChatRequest:
    merged = dict(payload.chat_template_kwargs or {})
    merged.update(updates)
    return payload.model_copy(update={"chat_template_kwargs": merged})


class _OpenAiCompatibleUsageDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    reasoning_tokens: int | None = None


class _OpenAiCompatibleUsagePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = Field(default=None, ge=0.0)
    completion_tokens_details: _OpenAiCompatibleUsageDetails | None = None

    def to_usage(self) -> LlmUsage:
        reasoning_tokens = self._reasoning_tokens()
        return LlmUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=_completion_tokens_excluding_reasoning(
                completion_tokens=self.completion_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
            total_tokens=self.total_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _reasoning_tokens(self) -> int | None:
        if self.reasoning_tokens is not None:
            return self.reasoning_tokens
        if self.completion_tokens_details is None:
            return None
        return self.completion_tokens_details.reasoning_tokens


def _completion_tokens_excluding_reasoning(
    *,
    completion_tokens: int | None,
    reasoning_tokens: int | None,
) -> int | None:
    # OpenAI-style usage includes reasoning inside completion tokens; internal
    # pricing tracks reasoning separately, so normalize completion first.
    if completion_tokens is None or reasoning_tokens is None:
        return completion_tokens
    return max(0, completion_tokens - reasoning_tokens)


class _OpenAiCompatibleChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str
    choices: list[_OpenAiCompatibleChoicePayload] = Field(default_factory=list)
    usage: _OpenAiCompatibleUsagePayload | None = None

    @classmethod
    def from_stream_state(cls, state: OpenAiStreamState) -> _OpenAiCompatibleChatResponse:
        choices = [
            _OpenAiCompatibleChoicePayload.from_choice_state(index=index, state=choice_state)
            for index, choice_state in sorted(state.choices.items())
        ]
        usage = _OpenAiCompatibleUsagePayload.model_validate(state.usage) if state.usage is not None else None
        return cls(id=state.response_id, choices=choices, usage=usage)

    def to_llm_response(self) -> LlmResponse:
        choices = tuple(choice.to_choice() for choice in self.choices)
        return LlmResponse(
            id=self.id,
            choices=choices,
            usage=self.usage.to_usage() if self.usage is not None else LlmUsage(),
            finish_reason=choices[0].finish_reason if choices else None,
        )


class _OpenAiCompatibleChoicePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    content: str
    reasoning: str | None = None
    tool_calls: tuple[LlmMessageToolCall, ...] | None = None
    finish_reason: str | None = None

    @classmethod
    def from_choice_state(cls, *, index: int, state: OpenAiChoiceState) -> _OpenAiCompatibleChoicePayload:
        return cls(
            index=index,
            content=state.content_text,
            reasoning=state.reasoning_text or None,
            tool_calls=_to_llm_tool_calls(state),
            finish_reason=state.finish_reason,
        )

    def to_choice(self) -> LlmChoice:
        return LlmChoice(
            index=self.index,
            message=LlmChoiceMessage(
                role="assistant",
                content=(LlmMessageContentPart(type="text", text=self.content),),
                reasoning=self.reasoning,
                tool_calls=self.tool_calls,
            ),
            finish_reason=self.finish_reason or "stop",
        )


class _OpenAiCompatibleAuthenticator:
    def __init__(
        self,
        auth: OpenAiCompatibleNoAuthConfig
        | OpenAiCompatibleBearerTokenEnvAuthConfig
        | OpenAiCompatibleGoogleIdTokenAuthConfig,
    ) -> None:
        self._auth = auth

    async def headers(self) -> Mapping[str, str]:
        match self._auth:
            case OpenAiCompatibleNoAuthConfig():
                return {}
            case OpenAiCompatibleBearerTokenEnvAuthConfig(token_env=token_env):
                token = os.environ.get(token_env, "").strip()
                if not token:
                    raise RuntimeError(f"{token_env} must be configured for OpenAI-compatible bearer auth")
                return {"Authorization": f"Bearer {token}"}
            case OpenAiCompatibleGoogleIdTokenAuthConfig() as auth:
                token = await asyncio.to_thread(_refresh_google_id_token, auth)
                return {"Authorization": f"Bearer {token}"}
        raise RuntimeError(f"unsupported OpenAI-compatible auth type: {self._auth.type}")


def _refresh_google_id_token(auth: OpenAiCompatibleGoogleIdTokenAuthConfig) -> str:
    request = GoogleAuthRequest()
    if auth.credential_source == "adc":
        credentials = id_token.fetch_id_token_credentials(auth.audience, request=request)
    else:
        if auth.credential_env is None:
            raise RuntimeError("credential_env must be configured for service_account_json_b64_env")
        credentials = service_account.IDTokenCredentials.from_service_account_info(
            _service_account_info_from_env(auth.credential_env),
            target_audience=auth.audience,
        )
    credentials.refresh(request)
    token = credentials.token
    if not token:
        raise RuntimeError("Google ID token refresh returned an empty token")
    return str(token)


def _service_account_info_from_env(env_name: str) -> dict[str, object]:
    encoded = os.environ.get(env_name, "").strip()
    if not encoded:
        raise RuntimeError(f"{env_name} must be configured for OpenAI-compatible Google ID token auth")
    decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{env_name} must decode to a service-account JSON object")
    return dict(payload)


def _to_llm_tool_calls(state: OpenAiChoiceState) -> tuple[LlmMessageToolCall, ...] | None:
    tool_calls = state.tool_call_values()
    if not tool_calls:
        return None
    return tuple(_to_llm_tool_call(tool_call, index=index) for index, tool_call in enumerate(tool_calls))


def _to_llm_tool_call(tool_call: OpenAiToolCall, *, index: int) -> LlmMessageToolCall:
    return LlmMessageToolCall(
        id=tool_call.id or f"toolcall-{index}",
        type=tool_call.type or "function",
        name=tool_call.name,
        arguments=tool_call.arguments,
    )


def _summarize_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (ValueError, RuntimeError):
        try:
            data = response.text
        except RuntimeError:
            data = ""
    text = str(data.get("detail", data)) if isinstance(data, dict) else str(data)
    return text if len(text) <= 500 else text[:500] + "..."


def _iter_tool_calls(response: LlmResponse) -> tuple[LlmMessageToolCall, ...]:
    calls: list[LlmMessageToolCall] = []
    for choice in response.choices:
        calls.extend(choice.message.tool_calls or ())
    return tuple(calls)


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


__all__ = ["OpenAiCompatibleLlmProvider"]
