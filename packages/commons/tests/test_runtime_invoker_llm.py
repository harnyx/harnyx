from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools.executor import ToolInvocationOutput
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker

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
            "provider_extra": {"provider": {"only": ["cerebras"]}},
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(llm_provider.requests) == 1
    request = llm_provider.requests[0]
    assert request.extra == {"provider": {"only": ["cerebras"]}}
    assert request.internal_metadata is None
    assert request.output_mode == "text"
    assert output.actual_cost_provider == "openrouter"


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


async def test_runtime_invoker_lowers_ai_gateway_provider_options_extra_to_request_extra() -> None:
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
            "provider_extra": {"providerOptions": {"gateway": {"only": ["cerebras"]}}},
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
