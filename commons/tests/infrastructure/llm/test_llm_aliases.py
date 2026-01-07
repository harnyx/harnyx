from __future__ import annotations

import pytest

from caster_commons.llm.providers.aliases import (
    DEFAULT_MODEL_ALIASES,
    AliasingLlmProvider,
    LlmModelAliasResolver,
)
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


def test_alias_resolver_prefers_provider_specific_entry() -> None:
    resolver = LlmModelAliasResolver({
        "vertex-maas:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
    })
    result = resolver.resolve("openai/gpt-oss-20b", provider="vertex-maas")
    assert result == "publishers/openai/models/gpt-oss-20b-maas"


def test_alias_resolver_falls_back_to_global() -> None:
    resolver = LlmModelAliasResolver(
        {"openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas"},
    )
    assert (
        resolver.resolve("openai/gpt-oss-20b", provider="openai")
        == "publishers/openai/models/gpt-oss-20b-maas"
    )


async def test_aliasing_provider_rewrites_model_before_invocation() -> None:
    resolver = LlmModelAliasResolver(
        {
            "vertex:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
            "vertex-maas:openai/gpt-oss-20b": "publishers/openai/models/gpt-oss-20b-maas",
        }
    )
    delegate = StubProvider()
    provider = AliasingLlmProvider(provider_name="vertex-maas", delegate=delegate, resolver=resolver)

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


@pytest.mark.parametrize("model", ALLOWED_TOOL_MODELS)
def test_all_allowed_models_have_vertex_alias(model: str) -> None:
    key_vertex = f"vertex:{model.strip().lower()}"
    key_maas = f"vertex-maas:{model.strip().lower()}"
    assert key_vertex in DEFAULT_MODEL_ALIASES
    assert key_maas in DEFAULT_MODEL_ALIASES
