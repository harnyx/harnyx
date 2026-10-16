"""Protect purchased evidence, stable representation order and Expand retry semantics."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.tools.expand import ExpandClient
from harnyx_commons.tools.provider_billing import billing_evidence_payload
from harnyx_commons.tools.search_models import FetchPageRequest

pytestmark = pytest.mark.anyio("asyncio")
URL = "https://taostats.io/subnets/25"


def _response():
    # Reduced from the retained 2026-10-16 Taostats probe, preserving its wire shape.
    return {
        "meta": {
            "version": 1,
            "url": URL,
            "capturedAt": "2026-10-16T07:26:51.312Z",
            "snapshotId": "01a1439b-21cd-734f-a6db-04cabf518c09",
            "lang": "en",
            "pageType": "article",
            "playground": "https://expand.land/s/yjzVlxI-2G0m",
            "title": "SN25 · taostats",
            "citationEvidenceIds": [4530],
            "appendix": {"omitted": {"blocks": 7, "approxTokens": 566}},
        },
        "markdown": "  # SN25\n\n| Metric | Value |\n| --- | --- |\n| Price | 0.0054 |\n",
        "json": [
            {
                "id": 4979,
                "kind": "network",
                "role": "network-json",
                "score": 7.29,
                "url": "https://taostats.io/api/dtao/udf/symbol_info?netuid=25",
                "method": "GET",
                "body": {"price": 0.0054, "count": 3, "valid": True, "missing": None},
            }
        ],
        "data": {"assets": []},
    }


@pytest.mark.parametrize("variant", ["original", "empty", "pointer", "degraded"])
async def test_full_fetch_preserves_two_ordered_representations_and_provenance(variant):
    raw = _response()
    raw["meta"]["finalUrl"] = URL + "/resolved"
    if variant == "empty":
        raw["json"] = []
    elif variant == "pointer":
        raw["json"][0].pop("body")
        raw["json"][0].update(moreTokens=3000, keys=["price"], urlRedacted=True)
    elif variant == "degraded":
        raw["data"]["degraded"] = {"reason": "partial", "details": ["state unavailable"]}
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=raw)

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="private-key", client=http)
        result = await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
    assert str(requests[0].url) == "https://api.expand.ai/v1/fetch/json?include=appendix,statejson"
    assert requests[0].headers["x-expand-api-key"] == "private-key"
    assert json.loads(requests[0].content) == {"url": URL, "browserConfig": {"scrollFullPage": True}}
    assert len(result.response.data) == 2
    markdown, structured = result.response.data
    assert markdown.content == raw["markdown"]
    expected = {"meta": raw["meta"], "json": raw["json"]}
    if variant == "degraded":
        expected["data"] = {"degraded": raw["data"]["degraded"]}
    assert json.loads(structured.content) == expected
    assert markdown.url == structured.url == URL + "/resolved"
    assert markdown.title == structured.title == raw["meta"]["title"]
    assert result.billing.provider_request_id == raw["meta"]["snapshotId"]
    assert result.billing.actual_cost_usd is None


async def test_converted_document_accepts_metadata_without_browser_page_fields():
    raw = {
        "meta": {
            "version": 1,
            "url": "https://example.com/report.pdf",
            "capturedAt": "2026-10-16T00:00:00.000Z",
            "snapshotId": "document-1",
            "contentType": "application/pdf",
        },
        "markdown": "# Report\n\nConverted document text.",
        "json": [],
        "data": {"document": {"url": "https://example.com/original.pdf", "contentType": "application/pdf"}},
    }
    async with httpx.AsyncClient(
        base_url="https://api.expand.ai", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))
    ) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        result = await client.fetch_page(FetchPageRequest(provider="expand", url=raw["meta"]["url"]))
    markdown, structured = result.response.data
    assert markdown.content == raw["markdown"]
    assert markdown.url == structured.url == raw["meta"]["url"]
    assert markdown.title is structured.title is None
    assert json.loads(structured.content) == {"meta": raw["meta"], "json": []}


@pytest.mark.parametrize(
    "variant", ["valid", "no_videos", "negative", "fractional_pages", "string", "missing", "null_videos"]
)
async def test_conversion_usage_is_validated_and_retained_only_in_private_billing(variant):
    raw = _response()
    usage = {"documentPages": 12, "ocrPages": 3, "audioSeconds": 12.5, "videos": 1}
    if variant == "no_videos":
        usage.pop("videos")
    elif variant == "negative":
        usage["ocrPages"] = -1
    elif variant == "fractional_pages":
        usage["documentPages"] = 1.5
    elif variant == "string":
        usage["audioSeconds"] = "12.5"
    elif variant == "missing":
        usage.pop("documentPages")
    elif variant == "null_videos":
        usage["videos"] = None
    raw["data"]["usage"] = usage
    async with httpx.AsyncClient(
        base_url="https://api.expand.ai", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))
    ) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        if variant in {"valid", "no_videos"}:
            result = await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert billing_evidence_payload(result.billing)["provider_usage"] == usage
            assert result.billing.actual_cost_usd is None
            assert json.loads(result.response.data[1].content) == {"meta": raw["meta"], "json": raw["json"]}
        else:
            with pytest.raises(ToolProviderError, match="response invalid"):
                await client.fetch_page(FetchPageRequest(provider="expand", url=URL))


@pytest.mark.parametrize(
    "status,body,attempt_count",
    [(401, {"_tag": "AuthFailed"}, 1),
     (429, {"_tag": "TooManyRequests", "retryable": True}, 2),
     (503, {"_tag": "ServiceUnavailable", "retryable": True}, 2),
     (502, {"_tag": "FetchOriginHttpError", "url": URL, "statusCode": 403}, 2)],
)
async def test_failed_fetch_logs_trace_for_each_attempt_without_request_secrets(caplog, status, body, attempt_count):
    caplog.set_level("WARNING", logger="harnyx.tools.expand")
    attempts = []

    def handle(request):
        attempts.append(request)
        return httpx.Response(status, json=body, headers={"x-expand-trace-id": f"trace-{len(attempts)}"})

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai", api_key="private-key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
            include_payloads_in_logs=False,
        )
        with pytest.raises(ToolProviderError) as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
    records = [record for record in caplog.records if record.getMessage() == "expand.request.failed"]
    assert [record.data for record in records] == [
        {"status_code": status, "trace_id": f"trace-{index}"} for index in range(1, attempt_count + 1)
    ]
    assert len(attempts) == attempt_count
    assert "trace-" not in str(error.value)
    assert all("private-key" not in repr(record.__dict__) and URL not in repr(record.__dict__) for record in records)


@pytest.mark.parametrize(
    "invalid", ["missing_json", "bad_meta", "blank", "bad_evidence", "invalid_tag", "origin", "refusal"]
)
async def test_invalid_or_origin_failure_never_returns_a_reordered_partial_success(invalid):
    raw = copy.deepcopy(_response())
    if invalid == "missing_json":
        raw.pop("json")
    elif invalid == "bad_meta":
        raw["meta"].pop("snapshotId")
    elif invalid == "blank":
        raw["markdown"] = " \n "
    elif invalid == "bad_evidence":
        raw["json"][0]["score"] = "7"
    elif invalid == "invalid_tag":
        raw["_tag"] = []
    elif invalid == "origin":
        raw["originError"] = {"_tag": "FetchOriginError", "originStatusCode": 403, "retryable": False}
    else:
        raw = {
            "_tag": "FetchOriginError",
            "url": URL,
            "kind": "http",
            "originStatusCode": 403,
            "contentAvailable": False,
            "retryable": False,
        }
    async with httpx.AsyncClient(
        base_url="https://api.expand.ai", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=raw))
    ) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        with pytest.raises(ToolProviderError) as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
        assert error.value.failure_code == ToolProviderFailureCode.PROVIDER_FAILED
        assert error.value.http_status is None


@pytest.mark.parametrize(
    "status,tag,retryable",
    [
        (502, "FetchCaptureFailed", False),
        (502, "FetchNavigationFailed", False),
        (504, "FetchCaptureTimeout", False),
        (502, "FetchOriginHttpError", None),
        (429, "TooManyRequests", True),
        (529, "FetchCapacityTimeout", True),
        (529, "FetchCapacityTimeout", None),
        (502, "unrecognized", True),
    ],
)
async def test_provider_retry_contract_overrides_status_defaults(monkeypatch, status, tag, retryable):
    attempts = []
    delays = []
    body = {"_tag": tag, "url": URL, "statusCode": 403, "failureType": "pageNeverSettled"}
    if retryable is not None:
        body["retryable"] = retryable

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("harnyx_commons.tools.search_http.asyncio.sleep", sleep)

    def handle(request):
        attempts.append(request)
        return (
            httpx.Response(status, json=body, headers={"Retry-After": "1"})
            if len(attempts) == 1
            else httpx.Response(200, json=_response())
        )

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai",
            api_key="key",
            client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        if status in {429, 529} and retryable is True:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert len(attempts) == 2 and delays == [1.0]
        else:
            with pytest.raises(ToolProviderError) as error:
                await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert len(attempts) == 1 and delays == []
            if tag == "FetchOriginHttpError":
                assert error.value.http_status is None


@pytest.mark.parametrize(
    "tag,body_delay,header_delay,expected",
    [
        ("FetchConcurrencyLimitExceeded", 20, "1", 20),
        ("FetchConcurrencyLimitExceeded", 5, None, 5),
        ("TooManyRequests", 5, "20", 5),
        ("TooManyRequests", 0, "20", 3),
        ("TooManyRequests", None, "5", 5),
        ("ServiceUnavailable", 5, None, 5),
        ("InternalError", 5, None, 5),
        ("FetchCapacityTimeout", 5, None, 5),
    ],
)
async def test_body_retry_delay_overrides_header_and_preserves_backoff(
    monkeypatch, tag, body_delay, header_delay, expected
):
    requests, delays = [], []
    body = {"_tag": tag, "retryable": True, "url": URL, "maxConcurrency": 5}
    if body_delay is not None:
        body["retryAfterSeconds"] = body_delay
    headers = {"Retry-After": header_delay} if header_delay is not None else {}
    status = {"InternalError": 500, "ServiceUnavailable": 503, "FetchCapacityTimeout": 529}.get(tag, 429)

    async def sleep(delay):
        delays.append(delay)

    def handle(request):
        requests.append(request)
        return (
            httpx.Response(status, json=body, headers=headers)
            if len(requests) == 1
            else httpx.Response(200, json=_response())
        )

    monkeypatch.setattr("harnyx_commons.tools.search_http.asyncio.sleep", sleep)
    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai",
            api_key="key",
            client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=3000, max_ms=3000, jitter=0),
        )
        result = await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
    assert len(requests) == 2 and len(result.response.data) == 2
    assert delays == [expected]


@pytest.mark.parametrize("status,expected", [(401, "authentication_failed"), (403, "provider_failed")])
async def test_provider_authentication_is_distinct_from_access_failure(status, expected):
    async with httpx.AsyncClient(
        base_url="https://api.expand.ai",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"_tag": "AuthFailed"})),
    ) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        with pytest.raises(ToolProviderError) as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
        assert error.value.failure_code.value == expected


@pytest.mark.parametrize("operation", ["ocr", "transcription", "video", "unknown", None])
async def test_document_processing_outage_retries_only_the_documented_error_shape(monkeypatch, operation):
    attempts, delays = [], []
    body = {"_tag": "FetchDocumentProcessingUnavailable", "url": URL, "message": "Temporarily unavailable"}
    if operation is not None:
        body["operation"] = operation

    async def sleep(delay):
        delays.append(delay)

    def handle(request):
        attempts.append(request)
        return (
            httpx.Response(503, json=body, headers={"Retry-After": "1"})
            if len(attempts) == 1
            else httpx.Response(200, json=_response())
        )

    monkeypatch.setattr("harnyx_commons.tools.search_http.asyncio.sleep", sleep)
    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        if operation in {"ocr", "transcription", "video"}:
            assert len((await client.fetch_page(FetchPageRequest(provider="expand", url=URL))).response.data) == 2
            assert len(attempts) == 2 and delays == [1.0]
        else:
            with pytest.raises(ToolProviderError):
                await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert len(attempts) == 1 and delays == []


@pytest.mark.parametrize(
    "status,origin_status,retries",
    [(502, 403, True), (502, 500, True), (404, 404, False), (410, 410, False),
     (502, "503", False), (502, None, False)],
)
@pytest.mark.parametrize("recovers", [True, False])
async def test_document_origin_retry_preserves_terminal_refusals_and_source_attribution(
    monkeypatch, status, origin_status, retries, recovers,
):
    attempts, delays = [], []
    body = {"_tag": "FetchOriginHttpError", "url": URL, "contentType": "text/html"}
    if origin_status is not None:
        body["statusCode"] = origin_status

    async def sleep(delay):
        delays.append(delay)

    def handle(request):
        attempts.append(request)
        if len(attempts) > 1 and recovers:
            return httpx.Response(200, json=_response())
        return httpx.Response(status, json=body, headers={"Retry-After": "1"})

    monkeypatch.setattr("harnyx_commons.tools.search_http.asyncio.sleep", sleep)
    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai", api_key="key", client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        if retries and recovers:
            result = await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert len(result.response.data) == 2
        else:
            with pytest.raises(ToolProviderError) as error:
                await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
            assert error.value.http_status is None
            assert error.value.failure_code == ToolProviderFailureCode.PROVIDER_FAILED
        assert len(attempts) == (2 if retries else 1)
        assert delays == ([1.0] if retries else [])


@pytest.mark.parametrize(
    "availability",
    ["private", "removed", "geoRestricted", "ageRestricted", "membersOnly", "live", "upcoming", "unknown"],
)
@pytest.mark.parametrize("with_message", [False, True])
async def test_unavailable_video_is_a_source_failure_without_immediate_retries(availability, with_message):
    body = {"_tag": "FetchVideoUnavailable", "url": URL, "availability": availability}
    if with_message:
        body["message"] = "The video platform will not serve this video"
    attempts = []

    def handle(request):
        attempts.append(request)
        return httpx.Response(422, json=body)

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        with pytest.raises(ToolProviderError, match="fetch source failed") as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
    assert len(attempts) == 1
    assert error.value.provider == "expand"
    assert error.value.http_status is None
    assert error.value.failure_code == ToolProviderFailureCode.PROVIDER_FAILED


@pytest.mark.parametrize("invalid", ["missing_url", "unknown_availability", "null_message"])
async def test_malformed_video_error_keeps_provider_failure_attribution(invalid):
    body = {"_tag": "FetchVideoUnavailable", "url": URL, "availability": "private"}
    if invalid == "missing_url":
        body.pop("url")
    elif invalid == "unknown_availability":
        body["availability"] = "unrecognized"
    else:
        body["message"] = None
    async with httpx.AsyncClient(
        base_url="https://api.expand.ai", transport=httpx.MockTransport(lambda _: httpx.Response(422, json=body))
    ) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        with pytest.raises(ToolProviderError, match="tool provider failed") as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
    assert error.value.http_status == 422


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_interrupted_fetch_closes_stream_and_does_not_retry(stop):
    started, closed = asyncio.Event(), asyncio.Event()
    attempts = []

    class BlockedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    def handle(request):
        attempts.append(request)
        return httpx.Response(200, stream=BlockedStream())

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(base_url="https://api.expand.ai", api_key="key", client=http)
        pending = asyncio.create_task(client.fetch_page(FetchPageRequest(provider="expand", url=URL)))
        await started.wait()
        if stop == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(pending, timeout=0.01)
        assert closed.is_set() and len(attempts) == 1


@pytest.mark.parametrize("body", [b"not json", b'{"_tag": "FetchCapacityTimeout", "retryable": true}'])
async def test_malformed_retryable_error_is_terminal(body):
    attempts = []

    def handle(request):
        attempts.append(request)
        return httpx.Response(529, content=body)

    async with httpx.AsyncClient(base_url="https://api.expand.ai", transport=httpx.MockTransport(handle)) as http:
        client = ExpandClient(
            base_url="https://api.expand.ai",
            api_key="private-key",
            client=http,
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        with pytest.raises(ToolProviderError) as error:
            await client.fetch_page(FetchPageRequest(provider="expand", url=URL))
        assert len(attempts) == 1
        assert "private-key" not in str(error.value)
