from __future__ import annotations

import pytest

from caster_commons.llm.adapter import LlmProviderAdapter
from caster_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from caster_commons.tools.runtime_invoker import ALLOWED_TOOL_MODELS

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
        "vertex-maas:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
        "openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas-global",
    }
    delegate = StubProvider()
    provider = LlmProviderAdapter(provider_name="vertex-maas", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex-maas",
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
    provider = LlmProviderAdapter(provider_name="vertex-maas", delegate=delegate, model_aliases=aliases)

    request = LlmRequest(
        provider="vertex-maas",
        model="openai/gpt-oss-20b",
        messages=(),
        temperature=None,
        max_output_tokens=None,
        output_mode="text",
    )

    await provider.invoke(request)

    assert delegate.requests[0].model == "publishers/openai/models/gpt-oss-20b-maas"


@pytest.mark.parametrize("model", ALLOWED_TOOL_MODELS)
async def test_adapter_applies_default_vertex_aliases(model: str) -> None:
    expected = model.replace("openai/", "publishers/openai/models/") + "-maas"
    for provider_name in ("vertex", "vertex-maas"):
        delegate = StubProvider()
        provider = LlmProviderAdapter(provider_name=provider_name, delegate=delegate)

        request = LlmRequest(
            provider=provider_name,
            model=model,
            messages=(),
            temperature=None,
            max_output_tokens=None,
            output_mode="text",
        )

        await provider.invoke(request)

        assert delegate.requests[0].model == expected
