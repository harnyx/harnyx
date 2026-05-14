from __future__ import annotations

import pytest

from harnyx_commons.llm.adapter import LlmProviderAdapter, canonical_model_for_provider_model
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

pytestmark = pytest.mark.anyio("asyncio")

VERTEX_ALIASED_TOOL_MODELS = {
    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek-ai/deepseek-v3.1-maas",
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek-ai/deepseek-v3.2-maas",
    "openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    "openai/gpt-oss-120b": "publishers/openai/models/gpt-oss-120b-maas",
    "zai-org/GLM-5-TEE": "zai-org/glm-5-maas",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "publishers/qwen/models/qwen3-next-80b-a3b-instruct-maas",
}
GEMMA_CHUTES_MODEL = "google/gemma-4-31B-turbo-TEE"
GEMMA_CLOUD_RUN_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run-turbo"
GEMMA_CLOUD_RUN_NATIVE_MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"
QWEN36_CHUTES_MODEL = "Qwen/Qwen3.6-27B-TEE"
QWEN36_CLOUD_RUN_ROUTE_TARGET = "custom-openai-compatible:qwen36-cloud-run"
QWEN36_CLOUD_RUN_NATIVE_MODEL = "Qwen/Qwen3.6-27B-FP8"


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
        "vertex:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
        "openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas-global",
    }
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex",
        model="openai/gpt-oss-20b",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "publishers/openai/models/gpt-oss-20b-maas"


async def test_adapter_falls_back_to_global_entry() -> None:
    aliases = {"openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas"}
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex",
        model="openai/gpt-oss-20b",
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


async def test_bedrock_adapter_uses_provider_specific_minimax_m2_5_alias() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="bedrock", delegate=delegate)

    request = LlmRequest(
        provider="bedrock",
        model="MiniMaxAI/MiniMax-M2.5-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "minimax.minimax-m2.5"


async def test_vertex_adapter_uses_provider_specific_deepseek_v3_2_alias() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex", delegate=delegate)

    request = LlmRequest(
        provider="vertex",
        model="deepseek-ai/DeepSeek-V3.2-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "deepseek-ai/deepseek-v3.2-maas"


async def test_vertex_adapter_uses_provider_specific_qwen3_235b_alias() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex", delegate=delegate)

    request = LlmRequest(
        provider="vertex",
        model="Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "qwen3-235b-a22b-instruct-2507-maas"


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


@pytest.mark.parametrize(("model", "expected"), VERTEX_ALIASED_TOOL_MODELS.items())
async def test_adapter_applies_default_vertex_aliases(model: str, expected: str) -> None:
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

    assert delegate.requests[0].model == expected


def test_canonical_model_for_provider_model_reverses_vertex_tool_alias() -> None:
    assert (
        canonical_model_for_provider_model(
            provider_name="vertex",
            model="deepseek-ai/deepseek-v3.2-maas",
        )
        == "deepseek-ai/DeepSeek-V3.2-TEE"
    )


def test_canonical_model_for_provider_model_reverses_custom_openai_compatible_tool_alias() -> None:
    assert (
        canonical_model_for_provider_model(
            provider_name=GEMMA_CLOUD_RUN_ROUTE_TARGET,
            model=GEMMA_CLOUD_RUN_NATIVE_MODEL,
        )
        == GEMMA_CHUTES_MODEL
    )


def test_canonical_model_for_provider_model_reverses_qwen36_custom_openai_compatible_tool_alias() -> None:
    assert (
        canonical_model_for_provider_model(
            provider_name=QWEN36_CLOUD_RUN_ROUTE_TARGET,
            model=QWEN36_CLOUD_RUN_NATIVE_MODEL,
        )
        == QWEN36_CHUTES_MODEL
    )


def test_canonical_model_for_provider_model_returns_unknown_model_unchanged() -> None:
    assert (
        canonical_model_for_provider_model(
            provider_name="vertex",
            model="unmapped-provider-model",
        )
        == "unmapped-provider-model"
    )


@pytest.mark.parametrize("model", VERTEX_ALIASED_TOOL_MODELS)
async def test_adapter_leaves_open_model_ids_unchanged_for_chutes(model: str) -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="chutes", delegate=delegate)

    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == model


async def test_adapter_maps_gemma_chutes_id_to_turbo_cloud_run_native_model() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name=GEMMA_CLOUD_RUN_ROUTE_TARGET, delegate=delegate)

    request = LlmRequest(
        provider=GEMMA_CLOUD_RUN_ROUTE_TARGET,
        model=GEMMA_CHUTES_MODEL,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == GEMMA_CLOUD_RUN_NATIVE_MODEL


async def test_adapter_maps_qwen36_chutes_id_to_cloud_run_native_model() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name=QWEN36_CLOUD_RUN_ROUTE_TARGET, delegate=delegate)

    request = LlmRequest(
        provider=QWEN36_CLOUD_RUN_ROUTE_TARGET,
        model=QWEN36_CHUTES_MODEL,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == QWEN36_CLOUD_RUN_NATIVE_MODEL


async def test_adapter_leaves_gemma_chutes_id_unchanged_for_chutes() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="chutes", delegate=delegate)

    request = LlmRequest(
        provider="chutes",
        model=GEMMA_CHUTES_MODEL,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == GEMMA_CHUTES_MODEL


async def test_adapter_leaves_qwen36_chutes_id_unchanged_for_chutes() -> None:
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="chutes", delegate=delegate)

    request = LlmRequest(
        provider="chutes",
        model=QWEN36_CHUTES_MODEL,
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == QWEN36_CHUTES_MODEL
