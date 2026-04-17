from __future__ import annotations

import pytest

from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.tools.runtime_invoker import ALLOWED_TOOL_MODELS

pytestmark = pytest.mark.anyio("asyncio")


class StubProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:  # pragma: no cover - simple stub
        self.requests.append(request)
        return LlmResponse(
            id="stub",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                        tool_calls=None,
                    ),
                ),
            ),
            usage=LlmUsage(),
        )


async def test_adapter_prefers_provider_specific_entry() -> None:
    aliases = {
        "vertex-maas:openai/gpt-oss-20b-TEE": "publishers/openai/models/gpt-oss-20b-maas",
        "openai/gpt-oss-20b-TEE": "publishers/openai/models/gpt-oss-20b-maas-global",
    }
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex-maas", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex-maas",
        model="openai/gpt-oss-20b-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "publishers/openai/models/gpt-oss-20b-maas"


async def test_adapter_falls_back_to_global_entry() -> None:
    aliases = {"openai/gpt-oss-20b-TEE": "publishers/openai/models/gpt-oss-20b-maas"}
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex-maas", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex-maas",
        model="openai/gpt-oss-20b-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "publishers/openai/models/gpt-oss-20b-maas"


async def test_bedrock_adapter_uses_provider_specific_tee_alias() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="bedrock", delegate=delegate)

    request = LlmRequest(
        provider="bedrock",
        model="openai/gpt-oss-20b-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "openai.gpt-oss-20b-1:0"


async def test_bedrock_adapter_does_not_fall_back_to_global_alias() -> None:
    aliases = {"openai/gpt-oss-20b-TEE": "wrong-global-alias"}
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="bedrock", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="bedrock",
        model="openai/gpt-oss-20b-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "openai/gpt-oss-20b-TEE"


async def test_bedrock_adapter_uses_provider_specific_kimi_alias() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="bedrock", delegate=delegate)

    request = LlmRequest(
        provider="bedrock",
        model="moonshotai/Kimi-K2.5-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "moonshotai.kimi-k2.5"


async def test_bedrock_adapter_does_not_fall_back_to_global_kimi_alias() -> None:
    aliases = {"moonshotai/Kimi-K2.5-TEE": "wrong-global-alias"}
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="bedrock", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="bedrock",
        model="moonshotai/Kimi-K2.5-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "moonshotai/Kimi-K2.5-TEE"


@pytest.mark.parametrize("model", ALLOWED_TOOL_MODELS)
async def test_adapter_applies_default_vertex_maas_aliases(model: str) -> None:
    expected_aliases = {
        "openai/gpt-oss-20b-TEE": "publishers/openai/models/gpt-oss-20b-maas",
        "openai/gpt-oss-120b-TEE": "publishers/openai/models/gpt-oss-120b-maas",
        "Qwen/Qwen3-Next-80B-A3B-Instruct": "publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas",
    }
    expected = expected_aliases[model]
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex-maas", delegate=delegate)

    request = LlmRequest(
        provider="vertex-maas",
        model=model,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == expected


@pytest.mark.parametrize("model", ALLOWED_TOOL_MODELS)
async def test_adapter_leaves_open_model_ids_unchanged_for_vertex(model: str) -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex", delegate=delegate)

    request = LlmRequest(
        provider="vertex",
        model=model,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == model
