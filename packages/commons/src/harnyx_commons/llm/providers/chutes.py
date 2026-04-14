"""LLM adapter backed by the Chutes HTTP API."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.providers.chutes_codec import (
    _ChutesChatRequest,
    _ChutesChatResponse,
    _ChutesReasoningStreamState,
    _parse_chutes_response_payload,
)
from harnyx_commons.llm.providers.openai_stream import (
    OpenAiStreamError,
    OpenAiStreamState,
    iter_openai_sse_events,
)
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmMessageToolCall,
    LlmResponse,
)

logger = logging.getLogger(__name__)

_CHUTES_EMBEDDING_BASE_URL_BY_MODEL = {
    "Qwen/Qwen3-Embedding-0.6B": "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai",
}


class ChutesLlmProvider(BaseLlmProvider):
    """Wraps the Chutes chat completions endpoint as an LLM provider."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        auth_header: str = "Authorization",
        max_concurrent: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Chutes API key must be provided for LLM usage")
        super().__init__(provider_label="chutes", max_concurrent=max_concurrent)
        normalized_base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            base_url=normalized_base,
            timeout=timeout,
        )
        self._api_key = api_key
        self._auth_header = auth_header

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        headers = self._auth_headers()

        return await self._call_with_retry(
            request,
            call_coro=lambda current_request: self._request_chutes(
                self._build_request(current_request),
                headers,
                timeout_seconds=current_request.timeout_seconds,
            ),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    def _build_request(self, request: AbstractLlmRequest) -> _ChutesChatRequest:
        return _ChutesChatRequest.from_request(request)

    def _auth_headers(self) -> dict[str, str]:
        return {self._auth_header: f"Bearer {self._api_key}"}

    async def _request_chutes(
        self,
        payload: _ChutesChatRequest,
        headers: Mapping[str, str],
        *,
        timeout_seconds: float | None,
    ) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "json": payload.model_dump(mode="json", exclude_none=True),
            "headers": headers,
        }
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        body, ttft_ms = await self._stream_chat_completions(**request_kwargs)
        llm_response = body.to_llm_response()
        metadata = dict(llm_response.metadata or {})
        metadata.setdefault("raw_response", body.model_dump(mode="python", exclude_none=True))
        self._log_stream_ttft(
            model=payload.model,
            response_id=body.id or "",
            ttft_ms=ttft_ms,
        )
        return LlmResponse(
            id=llm_response.id,
            choices=llm_response.choices,
            usage=llm_response.usage,
            metadata=metadata,
            finish_reason=llm_response.finish_reason,
        )

    async def _stream_chat_completions(self, **request_kwargs: Any) -> tuple[_ChutesChatResponse, float | None]:
        started_at = time.perf_counter()
        state = OpenAiStreamState()
        reasoning_state = _ChutesReasoningStreamState()
        ttft_ms: float | None = None
        async with self._client.stream("POST", "v1/chat/completions", **request_kwargs) as response:
            if response.is_error:
                await response.aread()
            response.raise_for_status()
            async for event in iter_openai_sse_events(
                response,
                invalid_data_message="streamed chat completions returned non-JSON SSE data",
                invalid_event_message="streamed chat completions SSE event must be a JSON object",
            ):
                reasoning_state.merge_event(event)
                if state.merge_event(event, reasoning_keys=()):
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return _ChutesChatResponse.from_stream_state(state, reasoning_state=reasoning_state), ttft_ms

    def _log_stream_ttft(self, *, model: str, response_id: str, ttft_ms: float | None) -> None:
        if ttft_ms is None:
            return
        logger.debug(
            "chutes.stream.ttft",
            extra={
                "data": {
                    "provider": self._provider_label,
                    "model": model,
                    "response_id": response_id,
                    "ttft_ms": ttft_ms,
                }
            },
        )

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text and not resp.tool_calls:
            return False, True, "empty_output"
        for call in _iter_tool_calls(resp):
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
        """Close the underlying HTTP client when this provider owns it."""
        if self._owns_client:
            await self._client.aclose()


class _EmbeddingDatum(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    embedding: list[float] = Field(min_length=1)


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    data: list[_EmbeddingDatum] = Field(min_length=1)


@dataclass(slots=True)
class ChutesTextEmbeddingClient:
    model: str
    api_key: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    dimensions: int | None = None
    client: httpx.AsyncClient | None = None
    _owns_client: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("chutes embedding model must be configured")
        if not self.api_key:
            raise ValueError("Chutes API key must be provided for embeddings")
        resolved_base_url = _normalize_embedding_base_url(self.base_url, normalized_model)
        self.model = normalized_model
        self.base_url = resolved_base_url
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=resolved_base_url,
                timeout=self.timeout_seconds,
            )
            self._owns_client = True
        else:
            self._owns_client = False

    async def embed(self, text: str) -> tuple[float, ...]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("embedding input text must not be empty")
        client = self._require_client()
        response = await client.post(
            "v1/embeddings",
            json=self._request_body(normalized),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        payload = _EmbeddingResponse.model_validate(response.json())
        vector = tuple(float(value) for value in payload.data[0].embedding)
        if self.dimensions is not None and len(vector) != self.dimensions:
            raise RuntimeError(
                f"embedding dimensions mismatch: expected={self.dimensions} actual={len(vector)}"
            )
        return vector

    async def aclose(self) -> None:
        if self._owns_client:
            await self._require_client().aclose()

    def _request_body(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("chutes embedding client is not initialized")
        return self.client


def resolve_chutes_embedding_base_url(model: str) -> str:
    normalized_model = model.strip()
    if not normalized_model:
        raise RuntimeError("chutes embedding model must be configured")
    try:
        return _CHUTES_EMBEDDING_BASE_URL_BY_MODEL[normalized_model]
    except KeyError as exc:
        raise RuntimeError(f"no chutes embedding base_url configured for model: {normalized_model}") from exc


def _normalize_embedding_base_url(base_url: str | None, model: str) -> str:
    if base_url is None:
        return resolve_chutes_embedding_base_url(model)
    normalized_base_url = base_url.rstrip("/")
    if not normalized_base_url:
        raise ValueError("chutes embedding base_url must not be empty")
    return normalized_base_url


def _summarize_response(response: httpx.Response) -> str:
    try:
        data = response.json()
    except (ValueError, RuntimeError):
        try:
            data = response.text
        except RuntimeError:
            data = ""
    summary_payload = _parse_response_summary_payload(data)
    summary_value = summary_payload.detail if summary_payload.detail is not None else summary_payload.raw
    text = str(summary_value)
    return text if len(text) <= 500 else text[:500] + "…"


@dataclass(frozen=True, slots=True)
class _ResponseSummaryPayload:
    raw: object
    detail: object | None


def _parse_response_summary_payload(value: object) -> _ResponseSummaryPayload:
    match value:
        case {"detail": detail, **_rest}:
            return _ResponseSummaryPayload(raw=value, detail=detail)
        case _:
            return _ResponseSummaryPayload(raw=value, detail=None)


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


__all__ = [
    "ChutesLlmProvider",
    "ChutesTextEmbeddingClient",
    "resolve_chutes_embedding_base_url",
    "_parse_chutes_response_payload",
]
