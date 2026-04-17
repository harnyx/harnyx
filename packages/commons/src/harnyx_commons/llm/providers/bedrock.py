"""Async Amazon Bedrock provider backed by ConverseStream."""

from __future__ import annotations

import logging
import time

from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError

from harnyx_commons.llm.provider import BaseLlmProvider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmRequest, LlmResponse

from .bedrock_codec import BEDROCK_STREAM_EVENT_ADAPTER, BedrockConverseStreamRequest, BedrockStreamAccumulator

_ALLOWED_MODELS = frozenset(
    {
        "openai.gpt-oss-20b-1:0",
        "openai.gpt-oss-120b-1:0",
        "moonshotai.kimi-k2.5",
    }
)
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "InternalFailure",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelStreamErrorException",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)


class BedrockLlmProvider(BaseLlmProvider):
    """Normalize Bedrock ConverseStream responses into the shared LLM contract."""

    def __init__(
        self,
        *,
        region: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_concurrent: int | None = None,
    ) -> None:
        normalized_region = region.strip()
        if not normalized_region:
            raise ValueError("Bedrock region must be configured")
        super().__init__(provider_label="bedrock", max_concurrent=max_concurrent)
        self._region = normalized_region
        self._connect_timeout_seconds = connect_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds
        self._session = get_session()
        self._logger = logging.getLogger("harnyx_commons.llm.calls")

    async def _invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        validated_request = _validate_request(request)
        return await self._call_with_retry(
            validated_request,
            call_coro=lambda current_request: self._call_bedrock(_validate_request(current_request)),
            verifier=self._verify_response,
            classify_exception=self._classify_exception,
        )

    async def _call_bedrock(self, request: LlmRequest) -> LlmResponse:
        bedrock_request = BedrockConverseStreamRequest.from_llm_request(request)
        client_config = _build_client_config(
            connect_timeout_seconds=self._connect_timeout_seconds,
            default_read_timeout_seconds=self._read_timeout_seconds,
            request_timeout_seconds=request.timeout_seconds,
        )
        started_at = time.perf_counter()
        ttft_ms: float | None = None
        accumulator = BedrockStreamAccumulator()

        async with self._session.create_client(
            "bedrock-runtime",
            region_name=self._region,
            config=client_config,
        ) as client:
            response = await client.converse_stream(**bedrock_request.to_payload())
            accumulator.set_response_metadata(response.get("ResponseMetadata"))
            async for raw_event in response["stream"]:
                event = BEDROCK_STREAM_EVENT_ADAPTER.validate_python(raw_event)
                produced_output = accumulator.apply(event, raw_event=raw_event)
                if produced_output and ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started_at) * 1000, 2)

        response_id = accumulator.response_id()
        self._log_stream_ttft(model=request.model, response_id=response_id, ttft_ms=ttft_ms)
        return accumulator.to_llm_response()

    async def aclose(self) -> None:
        return None

    @staticmethod
    def _verify_response(resp: LlmResponse) -> tuple[bool, bool, str | None]:
        if not resp.choices:
            return False, True, "empty_choices"
        if not resp.raw_text and not resp.tool_calls:
            return False, True, "empty_output"
        return True, False, None

    @staticmethod
    def _classify_exception(exc: Exception) -> tuple[bool, str]:
        if isinstance(exc, ClientError):
            metadata = exc.response.get("ResponseMetadata", {})
            status = metadata.get("HTTPStatusCode")
            error = exc.response.get("Error", {})
            code = str(error.get("Code") or "unknown")
            message = str(error.get("Message") or exc)
            retryable = code in _RETRYABLE_ERROR_CODES or status == 429 or (isinstance(status, int) and status >= 500)
            return retryable, f"client_error:{code}:{status}:{message}"
        if isinstance(exc, ParamValidationError):
            return False, str(exc)
        if isinstance(exc, BotoCoreError):
            return True, exc.__class__.__name__
        if isinstance(exc, TimeoutError):
            return True, exc.__class__.__name__
        return False, str(exc)

    def _log_stream_ttft(self, *, model: str, response_id: str, ttft_ms: float | None) -> None:
        if ttft_ms is None:
            return
        self._logger.debug(
            "llm.bedrock.stream.ttft",
            extra={
                "data": {
                    "provider": self._provider_label,
                    "model": model,
                    "response_id": response_id,
                    "ttft_ms": ttft_ms,
                }
            },
        )


def _validate_request(request: AbstractLlmRequest) -> LlmRequest:
    if not isinstance(request, LlmRequest):
        raise ValueError("Bedrock first cut supports only ungrounded LlmRequest flows")
    if request.model not in _ALLOWED_MODELS:
        raise ValueError(f"unsupported Bedrock model: {request.model}")
    if request.output_mode == "json_object":
        raise ValueError("Bedrock does not support json_object output mode")
    if request.tools:
        raise ValueError("Bedrock first cut does not support tool definitions")
    if request.tool_choice is not None:
        raise ValueError("Bedrock first cut does not support tool choice")
    return request


def _build_client_config(
    *,
    connect_timeout_seconds: float,
    default_read_timeout_seconds: float,
    request_timeout_seconds: float | None,
) -> Config:
    read_timeout_seconds = (
        request_timeout_seconds if request_timeout_seconds is not None else default_read_timeout_seconds
    )
    return Config(
        connect_timeout=connect_timeout_seconds,
        read_timeout=read_timeout_seconds,
        retries={
            "total_max_attempts": 1,
            "mode": "standard",
        },
    )


__all__ = ["BedrockLlmProvider"]
