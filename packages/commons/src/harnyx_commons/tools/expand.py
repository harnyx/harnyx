"""Expand Fetch JSON adapter: Markdown and JSON are separate referenceable items."""

from __future__ import annotations

import json
import logging
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult
from harnyx_commons.tools.search_http import HttpErrorDecision, JsonSearchProviderClient
from harnyx_commons.tools.search_models import FetchPageRequest, FetchPageResponse, FetchPageResult
from harnyx_miner_sdk.tools.search_provider_extra import ExpandFetchExtra

_LOGGER = logging.getLogger("harnyx.tools.expand")


class _ConversionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    document_pages: int = Field(alias="documentPages", ge=0)
    ocr_pages: int = Field(alias="ocrPages", ge=0)
    audio_seconds: float = Field(alias="audioSeconds", ge=0)
    videos: int = Field(default=0, ge=0)


class _Meta(BaseModel):
    # Additive provider metadata is opaque evidence and must survive normalization.
    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    version: Literal[1]
    url: str = Field(min_length=1)
    captured_at: str = Field(alias="capturedAt")
    snapshot_id: str = Field(alias="snapshotId", min_length=1)
    lang: str | None = None
    page_type: str | None = Field(default=None, alias="pageType")
    playground: str | None = None
    final_url: str | None = Field(default=None, alias="finalUrl", min_length=1)
    title: str | None = None


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    id: float
    kind: Literal["inline", "network"]
    role: Literal[
        "ld+json",
        "next-data",
        "nuxt",
        "apollo-state",
        "initial-state",
        "application-json",
        "script-tag",
        "data-attribute",
        "network-json",
        "network-response",
        "other",
    ]
    score: float = Field(ge=0)
    url: str | None = None
    method: str | None = None
    body: JsonValue = None
    more_tokens: float | None = Field(default=None, alias="moreTokens")
    keys: list[str] | None = None
    url_redacted: Literal[True] | None = Field(default=None, alias="urlRedacted")
    url_omitted: Literal["invalid", "unsafe", "too-long"] | None = Field(default=None, alias="urlOmitted")


class _FetchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    meta: _Meta
    markdown: str = Field(min_length=1)
    json_evidence: list[_Evidence] = Field(alias="json")
    data: dict[str, JsonValue] | None = None
    origin_error: dict[str, JsonValue] | None = Field(default=None, alias="originError")


class _TaggedError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    tag: str = Field(alias="_tag")
    retryable: bool | None = None


class _RetryableError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    retryable: Literal[True]
    retry_after_seconds: float | None = Field(default=None, alias="retryAfterSeconds", ge=0)


class _TransientError(_RetryableError):
    tag: Literal["TooManyRequests", "InternalError", "ServiceUnavailable"] = Field(alias="_tag")


class _ConcurrencyError(_RetryableError):
    tag: Literal["FetchConcurrencyLimitExceeded"] = Field(alias="_tag")
    retry_after_seconds: float = Field(alias="retryAfterSeconds", ge=0)
    max_concurrency: float = Field(alias="maxConcurrency")


class _CapacityError(_RetryableError):
    tag: Literal["FetchCapacityTimeout"] = Field(alias="_tag")
    url: str


class _DocumentProcessingError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tag: Literal["FetchDocumentProcessingUnavailable"] = Field(alias="_tag")
    url: str
    operation: Literal["ocr", "transcription", "video"]
    message: str


class _OriginHttpError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    tag: Literal["FetchOriginHttpError"] = Field(alias="_tag")
    url: str
    status_code: float = Field(alias="statusCode")
    content_type: str | None = Field(default=None, alias="contentType")


class _VideoUnavailableError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tag: Literal["FetchVideoUnavailable"] = Field(alias="_tag")
    url: str
    availability: Literal[
        "private", "removed", "geoRestricted", "ageRestricted", "membersOnly", "live", "upcoming", "unknown"
    ]
    message: str = ""


_RETRYABLE_ERROR = TypeAdapter(_TransientError | _ConcurrencyError | _CapacityError)


_RETRYABLE_ERRORS = {
    429: {"TooManyRequests", "FetchConcurrencyLimitExceeded"},
    500: {"InternalError"},
    503: {"ServiceUnavailable", "FetchDocumentProcessingUnavailable"},
    529: {"FetchCapacityTimeout"},
}
_ORIGIN_ERRORS = ("FetchOriginError", "FetchOriginHttpError", "FetchOriginNotFound", "FetchOriginGone")


def _classify_http_error(response: httpx.Response, body: bytes) -> HttpErrorDecision:
    status = response.status_code
    trace_id = response.headers.get("x-expand-trace-id")
    if trace_id:
        _LOGGER.warning("expand.request.failed", extra={"data": {"status_code": status, "trace_id": trace_id}})
    failure = (
        ToolProviderFailureCode.AUTHENTICATION_FAILED if status == 401 else ToolProviderFailureCode.PROVIDER_FAILED
    )
    error = ToolProviderError("tool provider failed", provider="expand", http_status=status, failure_code=failure)
    try:
        tagged = _TaggedError.model_validate_json(body)
    except ValidationError:
        return HttpErrorDecision(False, error)
    if status == 422 and tagged.tag == "FetchVideoUnavailable":
        try:
            _VideoUnavailableError.model_validate_json(body)
        except ValidationError:
            return HttpErrorDecision(False, error)
        # Live/upcoming videos need a later fetch after the stream, not an immediate retry.
        return HttpErrorDecision(False, ToolProviderError("fetch source failed", provider="expand"))
    if tagged.tag in _ORIGIN_ERRORS:
        # A target site's 401/403 must never invalidate the Expand credential.
        source_error = ToolProviderError("fetch source failed", provider="expand")
        if status == 502 and tagged.tag == "FetchOriginHttpError":
            try:
                _OriginHttpError.model_validate_json(body)
            except ValidationError:
                return HttpErrorDecision(False, source_error)
            return HttpErrorDecision(True, source_error)
        return HttpErrorDecision(False, source_error)
    if tagged.tag not in _RETRYABLE_ERRORS.get(status, set()):
        return HttpErrorDecision(False, error)
    try:
        if tagged.tag == "FetchDocumentProcessingUnavailable":
            # This documented transient error has no retryable field.
            _DocumentProcessingError.model_validate_json(body)
            return HttpErrorDecision(True, error)
        parsed = _RETRYABLE_ERROR.validate_json(body)
    except ValidationError:
        return HttpErrorDecision(False, error)
    return HttpErrorDecision(True, error, retry_after_seconds=parsed.retry_after_seconds)


class ExpandClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        retry_policy: RetryPolicy | None = None,
        max_concurrent: int | None = None,
        max_response_bytes: int | None = None,
        include_payloads_in_logs: bool = True,
    ) -> None:
        self._http = JsonSearchProviderClient(
            provider="expand",
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            auth_header="x-expand-api-key",
            auth_value_prefix="",
            client=client,
            retry_policy=retry_policy,
            max_concurrent=max_concurrent,
            max_response_bytes=max_response_bytes,
            include_payloads_in_logs=include_payloads_in_logs,
            error_classifier=_classify_http_error,
        )

    async def fetch_page(self, request: FetchPageRequest) -> SearchProviderResult[FetchPageResponse]:
        if request.provider != "expand" or (
            request.provider_extra is not None and not isinstance(request.provider_extra, ExpandFetchExtra)
        ):
            raise ValueError("Expand fetch requires provider='expand' and empty options")
        raw = await self._http.post_json(
            "/v1/fetch/json?include=appendix,statejson",
            {"url": request.url, "browserConfig": {"scrollFullPage": True}},
            requested_timeout=request.timeout,
        )
        if raw.get("_tag") in _ORIGIN_ERRORS or raw.get("originError") is not None:
            raise ToolProviderError("fetch source failed", provider="expand")
        try:
            parsed = _FetchResponse.model_validate(raw)
            usage = (
                _ConversionUsage.model_validate(parsed.data["usage"])
                if parsed.data is not None and "usage" in parsed.data
                else None
            )
            if not parsed.markdown.strip():
                raise ValueError("blank Markdown")
            # Keep provider JSON values exactly, including null, integer and omission distinctions.
            structured = {"meta": raw["meta"], "json": raw["json"]}
            if parsed.data is not None and "degraded" in parsed.data:
                structured["data"] = {"degraded": parsed.data["degraded"]}
            json_content = json.dumps(structured, ensure_ascii=False, allow_nan=False, indent=2)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ToolProviderError("tool provider response invalid", provider="expand") from exc
        url = parsed.meta.final_url or parsed.meta.url
        return SearchProviderResult(
            response=FetchPageResponse(
                data=[
                    FetchPageResult(url=url, title=parsed.meta.title, content=parsed.markdown),
                    FetchPageResult(url=url, title=parsed.meta.title, content=json_content),
                ]
            ),
            billing=ProviderBillingMetadata(
                actual_cost_provider="expand",
                source="response_body",
                service="fetch/json",
                provider_request_id=parsed.meta.snapshot_id,
                provider_usage=usage.model_dump(mode="json", by_alias=True, exclude_unset=True) if usage else None,
            ),
        )

    async def aclose(self) -> None:
        await self._http.aclose()
