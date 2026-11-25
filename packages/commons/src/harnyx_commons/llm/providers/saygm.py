"""SayGM Chat Completions using the shared OpenAI-compatible transport."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig
from harnyx_commons.llm.provider import LlmProviderConfigurationError
from harnyx_commons.llm.provider_types import SAYGM_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import (
    OpenAiCompatibleLlmProvider,
    _OpenAiCompatibleChatRequest,
)
from harnyx_commons.llm.providers.thinking import resolve_request_thinking
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_miner_sdk.tools.llm_provider_extra import SayGmExtra

SAYGM_BASE_URL = "https://api.saygm.com/v1"
_stream_output_started: ContextVar[bool] = ContextVar("saygm_stream_output_started", default=False)


class SayGmLlmProvider(OpenAiCompatibleLlmProvider):
    def __init__(self, *, api_key: SecretStr, client: httpx.AsyncClient | None = None) -> None:
        key = api_key.get_secret_value().strip()
        if not key:
            raise LlmProviderConfigurationError("SAYGM_API_KEY must be configured to build SayGM provider")
        endpoint = OpenAiCompatibleEndpointConfig.model_validate(
            {"id": SAYGM_PROVIDER, "base_url": SAYGM_BASE_URL, "auth": {"type": "none"}}
        )
        transport = client or httpx.AsyncClient(
            base_url=SAYGM_BASE_URL,
            timeout=300.0,
        )
        transport.headers["Authorization"] = f"Bearer {key}"
        super().__init__(endpoint=endpoint, client=transport, provider_label=SAYGM_PROVIDER)
        self._owns_client = client is None

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        token = _stream_output_started.set(False)
        try:
            return await super()._invoke(replace(request, provider=SAYGM_PROVIDER))
        finally:
            _stream_output_started.reset(token)

    async def _request_chat(
        self,
        request: AbstractLlmRequest,
    ) -> LlmResponse:
        _stream_output_started.set(False)
        body_extra = dict(request.extra or {})
        routing = {key: body_extra.pop(key) for key in ("X-GM-Routing", "X-GM-Providers") if key in body_extra}
        headers = SayGmExtra.model_validate(routing).to_request_extra() if routing else None
        response = await self._send_chat(replace(request, extra=body_extra), headers=headers)
        # A completed response is no longer an interrupted stream when a later backoff expires.
        _stream_output_started.set(False)
        return response

    def _record_stream_output(self) -> None:
        _stream_output_started.set(True)
        super()._record_stream_output()

    def _classify_retry_exception(
        self, exc: Exception, classify_exception: Callable[[Exception], tuple[bool, str]] | None
    ) -> tuple[bool, str]:
        retryable, reason = super()._classify_retry_exception(exc, classify_exception)
        if _stream_output_started.get():
            return False, f"stream_interrupted_after_output: {reason}"
        return retryable, reason

    def _retry_delay_seconds(self, attempt: int, policy: RetryPolicy, exc: Exception | None) -> float:
        delay = super()._retry_delay_seconds(attempt, policy, exc)
        if not isinstance(exc, httpx.HTTPStatusError):
            return delay
        value = exc.response.headers.get("Retry-After", "").strip()
        if value.isdecimal():
            return max(delay, int(value))
        if value:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return delay
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(delay, (retry_at - datetime.now(UTC)).total_seconds())
        return delay

    def _build_request(self, request: AbstractLlmRequest) -> _OpenAiCompatibleChatRequest:
        payload = super()._build_request(request)
        thinking = resolve_request_thinking(
            request_thinking=request.thinking, reasoning_effort=request.reasoning_effort
        )
        if thinking is not None and thinking.budget is not None:
            raise ValueError("SayGM does not support thinking.budget; use thinking.effort instead")
        updates: dict[str, object] = {"stream_options": {"include_usage": True}}
        if thinking is not None:
            updates["reasoning_effort"] = (thinking.effort or "medium") if thinking.enabled else "none"
        return payload.model_copy(update=updates)
