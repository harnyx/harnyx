"""Protect account isolation, native schemas and charge retention on bad answers."""

import asyncio
import json

import httpx
import pytest

from harnyx_commons.errors import ToolInvocationCancelledError, ToolProviderError
from harnyx_commons.llm.providers.cloudflare import CloudflareDecisionProvider, CloudflareLlmProvider
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest
from harnyx_miner_sdk.tools.cloudflare import CloudflareCredential
from harnyx_miner_sdk.tools.decision_models import DecisionQueryRequest

pytestmark = pytest.mark.anyio("asyncio")


def _credential() -> CloudflareCredential:
    return CloudflareCredential.model_validate({"account_id": "a" * 32, "api_token": "miner-token"})


async def test_decisions_use_stored_account_token_and_map_boolean_to_native_noul() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/client/v4/accounts/" + "a" * 32 + "/ai/run/@cf/cloudflare/clef-flash"
        assert request.headers["Authorization"] == "Bearer miner-token"
        body = json.loads(request.content)
        assert body["model"] == "clef-flash"
        assert body["questions"]["red"]["type"] == "noul"
        assert body["state"] == {"color": "red"}
        return httpx.Response(200, json={"success": True, "result": {
            "model": "clef-flash", "answers": {"red": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 100, "output_tokens": 0, "cost": 0},
        }})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        result = await provider.query(DecisionQueryRequest.model_validate({
            "provider": "cloudflare", "model": "@cf/cloudflare/clef-flash", "state": {"color": "red"},
            "questions": {"red": {"type": "boolean", "instructions": "Is the color red?"}},
        }))
        assert result.response.answers["red"].probability == 0.99
        assert result.actual_cost_usd == pytest.approx(100 * 0.038 / 1_000_000)
        assert result.reference_cost_usd == result.actual_cost_usd
        assert result.billing.reference_cost_usd == result.billing.actual_cost_usd
        await provider.aclose()
        assert not client.is_closed


@pytest.mark.parametrize("question,answer,valid", [
    ({"type": "number"}, {"type": "number", "value": None, "guess": 100, "p_null": 0.9}, True),
    ({"type": "number"}, {"type": "number", "value": "invalid"}, False),
    ({"type": "estimate", "min": 0, "max": 10},
     {"type": "estimate", "value": None, "median": 100}, False),
    ({"type": "duration", "unit": "days"}, {"type": "duration", "value": 24, "unit": "hours"}, False),
    ({"type": "number", "decimals": 2}, {"type": "number", "value": 1.234}, False),
    ({"type": "estimate", "min": 0, "max": 10},
     {"type": "estimate", "value": 5, "median": 5, "p10": 9, "p90": 1}, False),
    ({"type": "number", "null_threshold": 0.5}, {"type": "number", "value": 100, "p_null": 0.9}, False),
    ({"type": "list", "items": "text"}, {"type": "list", "value": None}, False),
    ({"type": "estimate", "min": 0, "max": 10}, {"type": "estimate", "value": None}, False),
    ({"type": "forecast", "min": 0, "max": 10, "horizon": 2, "interval": "day"},
     {"type": "forecast", "value": None}, False),
    ({"type": "forecast", "min": 0, "max": 10, "horizon": 2, "interval": "day"},
     {"type": "forecast", "value": [None, 4]}, False),
    ({"type": "forecast", "min": 0, "max": 10, "horizon": 1, "interval": "day"},
     {"type": "forecast", "value": [1], "steps": [{"value": 9}]}, False),
    ({"type": "forecast", "min": 0, "max": 10, "horizon": 1, "interval": "day"},
     {"type": "forecast", "value": [1], "p10": [9], "steps": [{"value": 1, "p90": 2}]}, False),
    ({"type": "forecast", "min": 0, "max": 10**16, "horizon": 1, "interval": "day"},
     {"type": "forecast", "value": [9007199254740993], "mean": [9007199254740993],
      "steps": [{"value": 9007199254740993, "mean": 9007199254740993}]}, True),
    ({"type": "forecast", "min": 0, "max": 10**16, "horizon": 1, "interval": "day"},
     {"type": "forecast", "value": [9007199254740992], "steps": [{"value": 9007199254740993}]}, False),
])
async def test_extraction_values_and_failure_billing_follow_native_contract(
    question: dict, answer: dict, valid: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["fields"]["amount"]["instructions"] == "Invoice total"
        assert "questions" not in body
        return httpx.Response(200, json={"success": True, "result": {
            "model": "clairic", "answers": {"amount": answer}, "truncated": False,
            "usage": {"input_tokens": 1000, "output_tokens": 0, "passes": 3},
        }})

    request = DecisionQueryRequest.model_validate({
        "provider": "cloudflare", "model": "@cf/cloudflare/clairic", "state": "An invoice without a total",
        "questions": {"amount": {"instructions": "Invoice total", **question}},
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        if not valid:
            with pytest.raises(ToolProviderError, match="invalid answers") as caught:
                await provider.query(request)
            assert caught.value.billing.actual_cost_usd == pytest.approx(0.0001)
            assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd
            assert caught.value.billing.usage.input_tokens == 1000
        else:
            result = await provider.query(request)
            assert result.response.answers["amount"].value == answer["value"]
            assert result.response.answers["amount"].guess == answer.get("guess")
            assert result.response.answers["amount"].mean == answer.get("mean")
            assert result.response.truncated is False
            assert result.actual_cost_usd == pytest.approx(0.0001)
            assert result.reference_cost_usd == result.actual_cost_usd


@pytest.mark.parametrize("content", ["Hello", 6, 6.25])
async def test_chat_stream_adapts_text_and_settles_cached_usage(content: str | int | float) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/client/v4/accounts/" + "a" * 32 + "/ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer miner-token"
        body = json.loads(request.content)
        assert body["model"] == "@cf/qwen/qwen3.8-27b"
        assert "max_tokens" not in body
        events = [
            {"id": "cf-test", "choices": [{"index": 0, "delta": {"content": content}}]},
            {"id": "cf-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "cf-test", "choices": [], "usage": {
                "prompt_tokens": 100, "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80}, "cost": 0,
            }},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(
            "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        ).encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        response = await provider.invoke(LlmRequest(
            provider="cloudflare", model="@cf/qwen/qwen3.8-27b", temperature=None, max_output_tokens=None,
            messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
        ))
        assert response.raw_text == str(content)
        assert response.usage.prompt_cached_tokens == 80
        assert response.metadata["actual_cost_provider"] == "cloudflare"
        assert response.metadata["actual_cost_usd"] == pytest.approx((20 * 0.45 + 80 * 0.05 + 10 * 3.2) / 1_000_000)
        assert response.metadata["reference_cost_usd"] == response.metadata["actual_cost_usd_total"]


@pytest.mark.parametrize("failure", ["invalid_json", "boolean_content", "nonfinite_content", "cancel"])
async def test_known_chat_usage_survives_broken_stream_and_cancellation(failure: str) -> None:
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            usage = {"id": "cf-billed", "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
            if failure in {"invalid_json", "cancel"}:
                yield f"data: {json.dumps(usage)}\n\n".encode()
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure in {"boolean_content", "nonfinite_content"}:
                content = True if failure == "boolean_content" else float("nan")
                malformed = {**usage, "choices": [{"index": 0, "delta": {"content": content}}]}
                yield f"data: {json.dumps(malformed)}\n\n".encode()
            else:
                yield b"data: not-json\n\n"

    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
    )) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        error = ToolInvocationCancelledError if failure == "cancel" else ToolProviderError
        with pytest.raises(error) as caught:
            await provider.invoke(LlmRequest(
                provider="cloudflare", model="@cf/openai/gpt-oss-20b", temperature=None, max_output_tokens=None,
                messages=(LlmMessage(role="user", content=(LlmMessageContentPart.input_text("Hello"),)),),
                use_case="tool_runtime_invoker",
                retry_policy=RetryPolicy(attempts=1, initial_ms=0, max_ms=0, jitter=0),
            ))
        assert caught.value.billing.actual_cost_usd == pytest.approx(0.000023)
        assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd


async def test_native_choice_missing_distribution_fails_with_retained_usage() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "success": True, "result": {
            "model": "clef", "answers": {"kind": {"type": "choice", "choice": "invoice"}},
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    }))) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        with pytest.raises(ToolProviderError) as caught:
            await provider.query(DecisionQueryRequest.model_validate({
                "provider": "cloudflare", "model": "@cf/cloudflare/clef", "state": "Invoice",
                "questions": {"kind": {"type": "choice", "instructions": "Kind?", "criteria": {
                    "invoice": None, "letter": None,
                }}},
            }))
        assert caught.value.billing.actual_cost_usd == pytest.approx(0.000024)
        assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd


@pytest.mark.parametrize("first_failure", ["empty", "read_error"])
@pytest.mark.parametrize("last_outcome", ["success", "cancel", "read_error"])
async def test_each_billable_attempt_is_counted_once_across_retry_outcomes(
    first_failure: str, last_outcome: str,
) -> None:
    """Partial attempts count too; cumulative updates within an attempt do not count twice."""
    from harnyx_commons.llm.cost_settlement import settled_cost_from_metadata
    attempts = 0

    class Stream(httpx.AsyncByteStream):
        def __init__(self, outcome: str) -> None:
            self.outcome = outcome

        async def __aiter__(self):
            for prompt, completion, cached, reasoning in ((50, 5, 40, 2), (100, 10, 80, 4)):
                event = {"id": str(attempts), "choices": [], "usage": {
                    "prompt_tokens": prompt, "completion_tokens": completion,
                    "prompt_tokens_details": {"cached_tokens": cached},
                    "completion_tokens_details": {"reasoning_tokens": reasoning},
                }}
                yield f"data: {json.dumps(event)}\n\n".encode()
            if self.outcome == "read_error":
                raise httpx.ReadError("stream interrupted")
            if self.outcome == "cancel":
                raise asyncio.CancelledError()
            event = {"choices": [{"index": 0, "delta": {
                "content": "" if self.outcome == "empty" else "OK",
            }}]}
            yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=Stream(first_failure if attempts == 1 else last_outcome))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        request = LlmRequest(
            provider="cloudflare", model="@cf/qwen/qwen3.8-27b", messages=(), temperature=None,
            max_output_tokens=None, use_case="tool_runtime_invoker",
            retry_policy=RetryPolicy(attempts=2, initial_ms=0, max_ms=0, jitter=0),
        )
        if last_outcome == "success":
            response = await provider.invoke(request)
            settled = settled_cost_from_metadata(response.metadata)
            usage = response.usage
            actual, reference = settled.cost_usd, settled.reference_cost_usd
        else:
            error = ToolInvocationCancelledError if last_outcome == "cancel" else ToolProviderError
            with pytest.raises(error) as caught:
                await provider.invoke(request)
            billing = caught.value.billing
            usage = billing.llm_usage
            actual, reference = billing.actual_cost_usd, billing.reference_cost_usd
    assert attempts == 2
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 12
    assert usage.prompt_cached_tokens == 160
    assert usage.reasoning_tokens == 8
    assert actual == pytest.approx(2 * (20 * 0.45 + 80 * 0.05 + 10 * 3.2) / 1_000_000)
    assert reference == actual


@pytest.mark.parametrize("status", [401, 403])
async def test_native_authentication_errors_reach_runtime_classification(status: int) -> None:
    from harnyx_commons.errors import ToolProviderFailureCode
    from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
    from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        invoker = RuntimeToolInvoker(InMemoryReceiptLog(), decision_provider=provider)
        with pytest.raises(ToolProviderError) as caught:
            await invoker.invoke("decision_query", args=(), kwargs={
                "provider": "cloudflare", "model": "@cf/cloudflare/clef", "state": "Invoice",
                "questions": {"paid": {"type": "boolean", "instructions": "Paid?"}},
            })
    assert caught.value.failure_code is ToolProviderFailureCode.AUTHENTICATION_FAILED
    assert caught.value.http_status == status


async def test_thinking_effort_reaches_the_native_request() -> None:
    from harnyx_commons.llm.schema import LlmThinkingConfig

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["reasoning_effort"] == "high"
        event = {"id": "thinking", "choices": [{"index": 0, "delta": {"content": "OK"}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await CloudflareLlmProvider(credential=_credential(), client=client).invoke(LlmRequest(
            provider="cloudflare", model="@cf/openai/gpt-oss-20b", messages=(), temperature=None,
            max_output_tokens=None, thinking=LlmThinkingConfig(enabled=True, effort="high"),
        ))


async def test_unsupported_thinking_budget_fails_before_dispatch() -> None:
    from harnyx_commons.llm.schema import LlmThinkingConfig

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("unsupported thinking controls reached paid transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        with pytest.raises(ValueError, match="does not support"):
            provider._build_request(LlmRequest(
                provider="cloudflare", model="@cf/openai/gpt-oss-20b", messages=(), temperature=None,
                max_output_tokens=None, thinking=LlmThinkingConfig(enabled=True, budget=100),
            ))


@pytest.mark.parametrize("enabled", [False, True])
async def test_native_template_switch_preserves_explicit_thinking_state(enabled: bool) -> None:
    from harnyx_commons.llm.schema import LlmThinkingConfig

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["chat_template_kwargs"]["enable_thinking"] is enabled
        assert "reasoning_effort" not in payload
        event = {"id": "switch", "choices": [{"index": 0, "delta": {"content": "OK"}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await CloudflareLlmProvider(credential=_credential(), client=client).invoke(LlmRequest(
            provider="cloudflare", model="@cf/qwen/qwen3.8-27b", messages=(), temperature=None,
            max_output_tokens=None, thinking=LlmThinkingConfig(enabled=enabled),
        ))


@pytest.mark.parametrize("kind", ["choice", "score"])
@pytest.mark.parametrize("probabilities,score,valid", [
    ([0.9, 0.9], 0.4, False), ([0.1, 0.2], 0.4, False),
    ([0.6, 0.4000000001], 0.4, True), ([0.1, 0.9], 0.1, False), ([0.5, 0.5], 0.5, True),
])
async def test_decision_distributions_require_unit_total_and_retain_billing(
    kind: str, probabilities: list[float], score: float, valid: bool,
) -> None:
    """Malformed distributions must fail without losing independently known charges."""
    options = ["invoice", "letter"] if kind == "choice" else ["0", "1"]
    answer = {
        "type": kind, "probabilities": dict(zip(options, probabilities, strict=True)), "confidence": 0.5,
        **({"choice": "invoice"} if kind == "choice" else {
            "score": score, "legend": {"0": "Small", "1": "Large"},
        }),
    }
    request = DecisionQueryRequest.model_validate({
        "provider": "cloudflare", "model": "@cf/cloudflare/clef-flash", "state": "Invoice",
        "questions": {"kind": {"type": kind, "instructions": "Classify invoice", "criteria": (
            {"invoice": None, "letter": None} if kind == "choice" else ["Small", "Large"]
        )}},
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "success": True, "result": {
            "model": "clef-flash", "answers": {"kind": answer},
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    }))) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        if valid:
            result = await provider.query(request)
            assert result.response.answers["kind"].probabilities == answer["probabilities"]
        else:
            with pytest.raises(ToolProviderError, match="invalid answers") as caught:
                await provider.query(request)
            assert caught.value.billing.actual_cost_usd == pytest.approx(100 * 0.038 / 1_000_000)
            assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd
            assert caught.value.billing.usage.input_tokens == 100


async def test_decision_legend_mismatch_retains_reported_billing() -> None:
    request = DecisionQueryRequest.model_validate({
        "provider": "cloudflare", "model": "@cf/cloudflare/clef-flash", "state": "Invoice",
        "questions": {"size": {"type": "score", "instructions": "Invoice size", "criteria": ["Small", "Large"]}},
    })
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "success": True, "result": {
            "model": "clef-flash", "answers": {"size": {
                "type": "score", "score": 0.5, "legend": {"0": "Large", "1": "Small"},
                "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5,
            }}, "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    }))) as client:
        provider = CloudflareDecisionProvider(credential=_credential(), client=client)
        with pytest.raises(ToolProviderError, match="invalid answers") as caught:
            await provider.query(request)
        assert caught.value.billing.actual_cost_usd == pytest.approx(100 * 0.038 / 1_000_000)
        assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd
        assert caught.value.billing.usage.input_tokens == 100


async def test_native_effort_switch_preserves_explicit_disabled_state() -> None:
    """Disabling reasoning must serialize a supported native off value."""
    from harnyx_commons.llm.schema import LlmThinkingConfig

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "none"
        assert "chat_template_kwargs" not in payload
        event = {"id": "disabled", "choices": [{"index": 0, "delta": {"content": "OK"}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await CloudflareLlmProvider(credential=_credential(), client=client).invoke(LlmRequest(
            provider="cloudflare", model="@cf/moonshotai/kimi-k2.6", messages=(), temperature=None,
            max_output_tokens=None, thinking=LlmThinkingConfig(enabled=False),
        ))


@pytest.mark.parametrize("malformed_usage", [
    None, {}, {"prompt_tokens": 100},
    {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": -1}},
    {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 101}},
    {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": -1},
    {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 11},
    {"prompt_tokens": 100, "completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": -1}},
    {"prompt_tokens": 100, "completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 11}},
    {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": -1},
])
@pytest.mark.parametrize("same_attempt", [False, True])
async def test_invalid_usage_cannot_replace_known_billing_or_borrow_it_from_another_attempt(
    malformed_usage: dict | None, same_attempt: bool,
) -> None:
    """Each native response needs its own complete usage; prior valid evidence still survives rejection."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        known = {"choices": [{"index": 0, "delta": {"content": ""}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        malformed = {"choices": [{"index": 0, "delta": {"content": "OK"}}]}
        if malformed_usage is not None:
            malformed["usage"] = malformed_usage
        events = [known, malformed] if same_attempt else [known if attempts == 1 else malformed]
        return httpx.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events)
                              + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        request = LlmRequest(
            provider="cloudflare", model="@cf/openai/gpt-oss-20b", messages=(), temperature=None,
            max_output_tokens=None, retry_policy=RetryPolicy(
                attempts=1 if same_attempt else 2, initial_ms=0, max_ms=0, jitter=0,
            ),
        )
        if same_attempt and malformed_usage is None:
            # Omitting usage from a content-only event keeps this attempt's valid snapshot.
            response = await provider.invoke(request)
            assert response.usage.prompt_tokens == 100
            assert response.metadata["actual_cost_usd_total"] == pytest.approx(0.000023)
        else:
            with pytest.raises(ToolProviderError) as caught:
                await provider.invoke(request)
            assert caught.value.billing.llm_usage.prompt_tokens == 100
            assert caught.value.billing.llm_usage.completion_tokens == 10
            assert caught.value.billing.actual_cost_usd == pytest.approx(0.000023)
            assert caught.value.billing.reference_cost_usd == caught.value.billing.actual_cost_usd


@pytest.mark.parametrize("billed", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "inactivity"])
async def test_native_owned_deadlines_keep_timeout_classification_and_known_billing(billed: bool, phase: str) -> None:
    """Reported usage must not turn an owned timeout into a generic provider failure."""
    from harnyx_commons.errors import ToolInvocationTimeoutError
    from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
    from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            event = {"choices": [{"index": 0, "delta": {"content": "partial"}}] if phase == "inactivity" else []}
            if billed:
                event["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
            yield f"data: {json.dumps(event)}\n\n".encode()
            await asyncio.Event().wait()

    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
    async with httpx.AsyncClient(transport=transport) as client:
        provider = CloudflareLlmProvider(credential=_credential(), client=client)
        invoker = RuntimeToolInvoker(InMemoryReceiptLog(), llm_provider=provider, llm_provider_name="cloudflare")
        with pytest.raises(ToolInvocationTimeoutError, match=f"LLM {phase} timeout") as caught:
            await invoker.invoke("llm_chat", args=(), kwargs={
                "provider": "cloudflare", "model": "@cf/openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello"}],
                "timeout": {"total": 2, "prefill": 0.05, "inactivity": 0.05},
            })
        if billed:
            billing = caught.value.billing
            assert billing.actual_cost_usd == pytest.approx(0.000023)
            assert billing.reference_cost_usd == billing.actual_cost_usd
            assert billing.llm_usage.prompt_tokens == 100
        else:
            assert caught.value.billing is None
