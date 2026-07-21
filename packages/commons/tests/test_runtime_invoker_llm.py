from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from harnyx_commons.domain.session import ProviderCredentialSource
from harnyx_commons.errors import ProviderCredentialUnavailableError, ToolProviderError, ToolProviderFailureCode
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.provider import LlmProviderConfigurationError, LlmProviderError
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.platform_tool_proxy import platform_tool_proxy_provider_timeout_seconds
from harnyx_commons.tools.executor import ToolInvocationContext, ToolInvocationOutput
from harnyx_commons.tools.runtime_invoker import DEFAULT_TOOL_LLM_TIMEOUT_SECONDS, RuntimeToolInvoker

pytestmark = pytest.mark.anyio("asyncio")


class _CapturingLlmProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="resp-provider-extra",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
            metadata={
                "actual_cost_usd": 0.0001,
                "actual_cost_provider": request.provider,
                "actual_cost_evidence": {"settlement_source": "test"},
            },
        )

    async def aclose(self) -> None:
        return None


def _platform_context() -> ToolInvocationContext:
    return ToolInvocationContext(
        receipt_id="receipt-1",
        session_id=uuid4(),
        active_attempt=0,
        uid=1,
        provider_credential_source=ProviderCredentialSource.PLATFORM,
    )


def _miner_context() -> ToolInvocationContext:
    return ToolInvocationContext(
        receipt_id="receipt-miner",
        session_id=uuid4(),
        active_attempt=0,
        uid=1,
        provider_credential_source=ProviderCredentialSource.MINER,
    )


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("openrouter", "openai/gpt-oss-120b"),
        ("ai_gateway", "thinkingmachines/inkling"),
    ],
)
async def test_platform_credential_session_resolves_requested_provider_without_miner_fallback(
    provider: str,
    model: str,
) -> None:
    platform_provider = _CapturingLlmProvider()
    miner_resolver_calls: list[str] = []
    platform_resolver_calls: list[str] = []

    async def miner_resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        miner_resolver_calls.append(provider)
        return _CapturingLlmProvider()

    async def platform_resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        platform_resolver_calls.append(provider)
        return platform_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider_resolver=miner_resolver,
        platform_llm_provider_resolver=platform_resolver,
    )

    await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": provider,
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        },
        context=_platform_context(),
    )

    assert len(platform_provider.requests) == 1
    assert platform_provider.requests[0].include_payloads_in_observability is False
    assert platform_provider.requests[0].timeout_seconds == platform_tool_proxy_provider_timeout_seconds(
        DEFAULT_TOOL_LLM_TIMEOUT_SECONDS
    )
    assert platform_resolver_calls == [provider]
    assert miner_resolver_calls == []


async def test_context_free_session_uses_matching_direct_provider_without_resolver() -> None:
    direct_provider = _CapturingLlmProvider()
    resolver_calls: list[str] = []

    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        resolver_calls.append(provider)
        return _CapturingLlmProvider()

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=direct_provider,
        llm_provider_name="chutes",
        llm_provider_resolver=resolver,
    )

    await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "chutes",
            "model": "deepseek-ai/DeepSeek-V3.2-TEE",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert len(direct_provider.requests) == 1
    assert resolver_calls == []


async def test_miner_credential_session_uses_miner_resolver_without_direct_fallback() -> None:
    direct_provider = _CapturingLlmProvider()
    miner_provider = _CapturingLlmProvider()
    resolver_calls: list[str] = []

    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        resolver_calls.append(provider)
        return miner_provider

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=direct_provider,
        llm_provider_name="chutes",
        llm_provider_resolver=resolver,
    )

    await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
        },
        context=_miner_context(),
    )

    assert resolver_calls == ["openrouter"]
    assert len(miner_provider.requests) == 1
    assert direct_provider.requests == []


async def test_platform_credential_source_error_is_mapped_at_invoker_boundary() -> None:
    def resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        raise ProviderCredentialUnavailableError(provider)

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_llm_provider_resolver=resolver,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "hi"}],
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.__cause__ is None


async def test_platform_credential_session_does_not_fallback_when_provider_is_missing() -> None:
    resolver_calls: list[str] = []

    async def miner_resolver(provider: str, _context: ToolInvocationContext | None) -> _CapturingLlmProvider:
        resolver_calls.append(provider)
        return _CapturingLlmProvider()

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider_resolver=miner_resolver,
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "openrouter",
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": "hi"}],
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "openrouter"
    assert resolver_calls == []


async def test_platform_llm_authentication_status_is_preserved_as_typed_failure() -> None:
    class AuthenticationFailureProvider:
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            response = httpx.Response(401, request=httpx.Request("POST", "https://provider.test/v1/chat"))
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise LlmProviderError("raw-provider-envelope") from exc
            raise AssertionError("unreachable")

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_llm_provider_resolver=lambda _provider, _context: AuthenticationFailureProvider(),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "deepseek-ai/DeepSeek-V3.2-TEE",
                "messages": [{"role": "user", "content": "hi"}],
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.AUTHENTICATION_FAILED
    assert exc_info.value.provider == "chutes"
    assert exc_info.value.http_status == 401
    assert exc_info.value.__cause__ is None


async def test_platform_llm_blank_credential_configuration_is_typed() -> None:
    class BlankCredentialProvider:
        async def invoke(self, request: LlmRequest) -> LlmResponse:
            raise LlmProviderConfigurationError("raw-blank-credential-detail")

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_llm_provider_resolver=lambda _provider, _context: BlankCredentialProvider(),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "deepseek-ai/DeepSeek-V3.2-TEE",
                "messages": [{"role": "user", "content": "hi"}],
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "chutes"


async def test_platform_embedding_missing_credential_is_typed() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider_name="chutes",
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "embed_text",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "Qwen/Qwen3-Embedding-8B-TEE",
                "texts": ["hello"],
                "input_type": "query",
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "chutes"


async def test_platform_search_missing_credential_is_typed() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        web_search_provider_name="parallel",
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "search_web",
            args=(),
            kwargs={"provider": "parallel", "search_queries": ["hello"]},
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.CREDENTIAL_UNAVAILABLE
    assert exc_info.value.provider == "parallel"


async def test_platform_embedding_authentication_status_is_preserved_as_typed_failure() -> None:
    class AuthenticationFailureProvider:
        async def embed_text(self, request) -> None:
            response = httpx.Response(401, request=httpx.Request("POST", "https://provider.test/v1/embed"))
            response.raise_for_status()

    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        platform_embedding_provider_resolver=lambda _provider, _context: AuthenticationFailureProvider(),
    )

    with pytest.raises(ToolProviderError) as exc_info:
        await invoker.invoke(
            "embed_text",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "Qwen/Qwen3-Embedding-8B-TEE",
                "texts": ["hello"],
                "input_type": "query",
            },
            context=_platform_context(),
        )

    assert exc_info.value.failure_code is ToolProviderFailureCode.AUTHENTICATION_FAILED
    assert exc_info.value.provider == "chutes"
    assert exc_info.value.http_status == 401


async def test_runtime_invoker_lowers_openrouter_provider_extra_to_request_extra() -> None:
    llm_provider = _CapturingLlmProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=llm_provider,
        llm_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
            "provider_extra": {"provider": {"only": ["cerebras"]}},
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(llm_provider.requests) == 1
    request = llm_provider.requests[0]
    assert request.extra == {"provider": {"only": ["cerebras"]}}
    assert request.internal_metadata is None
    assert request.output_mode == "text"
    assert request.max_output_tokens == 128
    assert output.actual_cost_provider == "openrouter"
    assert request.include_payloads_in_observability is True


async def test_runtime_invoker_sets_single_attempt_retry_policy_for_llm_chat() -> None:
    llm_provider = _CapturingLlmProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=llm_provider,
        llm_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(llm_provider.requests) == 1
    assert llm_provider.requests[0].retry_policy == RetryPolicy(
        attempts=1,
        initial_ms=0,
        max_ms=0,
        jitter=0.0,
    )


async def test_runtime_invoker_normalizes_ai_gateway_provider_extra_to_provider_options() -> None:
    llm_provider = _CapturingLlmProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=llm_provider,
        llm_provider_name="ai_gateway",
    )

    output = await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "ai_gateway",
            "model": "zai/glm-5.2-fast",
            "messages": [{"role": "user", "content": "hi"}],
            "provider_extra": {"provider": {"only": ["cerebras"]}},
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(llm_provider.requests) == 1
    request = llm_provider.requests[0]
    assert request.extra == {"providerOptions": {"gateway": {"only": ["cerebras"]}}}
    assert output.actual_cost_provider == "ai_gateway"


async def test_runtime_invoker_rejects_chutes_provider_extra() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=_CapturingLlmProvider(),
        llm_provider_name="chutes",
    )

    with pytest.raises(ValidationError):
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "Qwen/Qwen3.6-27B-TEE",
                "messages": [{"role": "user", "content": "hi"}],
                "provider_extra": {"provider": {"only": ["cerebras"]}},
            },
        )


async def test_runtime_invoker_lowers_complete_tool_loop_without_routing_policy() -> None:
    llm_provider = _CapturingLlmProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=llm_provider,
        llm_provider_name="openrouter",
    )
    reasoning_details = [{"type": "reasoning.encrypted", "data": "opaque"}]

    await invoker.invoke(
        "llm_chat",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "name": "lookup_weather",
                            "arguments": '{"city":"Paris"}',
                        }
                    ],
                    "reasoning_details": reasoning_details,
                },
                {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature":19}'},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup_weather", "strict": True},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "parallel_tool_calls": True,
            "provider_extra": {"provider": {"only": ["cerebras"]}},
        },
    )

    request = llm_provider.requests[0]
    assert request.messages[1].tool_calls is not None
    assert request.messages[1].tool_calls[0].id == "call-1"
    assert request.messages[1].reasoning_details == tuple(reasoning_details)
    assert request.messages[2].content[0].tool_call_id == "call-1"
    assert request.tool_choice == {"type": "function", "function": {"name": "lookup_weather"}}
    assert request.parallel_tool_calls is True
    assert request.extra == {"provider": {"only": ["cerebras"]}}
    assert "require_parameters" not in request.extra["provider"]


@pytest.mark.parametrize("field", ("include", "response_format"))
async def test_runtime_invoker_rejects_removed_fields_before_provider(field: str) -> None:
    llm_provider = _CapturingLlmProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        llm_provider=llm_provider,
        llm_provider_name="chutes",
    )

    with pytest.raises(ValidationError, match=field):
        await invoker.invoke(
            "llm_chat",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hi"}],
                field: [] if field == "include" else 10,
            },
        )

    assert llm_provider.requests == []
