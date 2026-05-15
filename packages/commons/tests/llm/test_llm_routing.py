from __future__ import annotations

from dataclasses import dataclass

import pytest

from harnyx_commons.llm.routing import (
    ResolvedLlmRoute,
    RoutedLlmProvider,
    parse_llm_model_provider_overrides,
    resolve_llm_route,
)
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)


def test_parse_llm_model_provider_overrides_accepts_surface_scoped_json() -> None:
    parsed = parse_llm_model_provider_overrides(
        '{"generator":{"sample-routed-model":"bedrock"},"scoring":{"openai/gpt-oss-120b-TEE":"bedrock"}}'
    )

    assert parsed == {
        "generator": {"sample-routed-model": "bedrock"},
        "scoring": {"openai/gpt-oss-120b-TEE": "bedrock"},
    }


def test_parse_llm_model_provider_overrides_accepts_custom_openai_compatible_target() -> None:
    parsed = parse_llm_model_provider_overrides(
        (
            '{"tool":{'
            '"google/gemma-4-31B-turbo-TEE":"custom-openai-compatible:gemma4-cloud-run-turbo",'
            '"Qwen/Qwen3.6-27B-TEE":"custom-openai-compatible:qwen36-cloud-run"'
            "}}"
        ),
        custom_openai_compatible_endpoint_ids={"gemma4-cloud-run-turbo", "qwen36-cloud-run"},
    )

    assert parsed == {
        "tool": {
            "google/gemma-4-31B-turbo-TEE": "custom-openai-compatible:gemma4-cloud-run-turbo",
            "Qwen/Qwen3.6-27B-TEE": "custom-openai-compatible:qwen36-cloud-run",
        }
    }


def test_parse_llm_model_provider_overrides_rejects_unknown_custom_endpoint() -> None:
    with pytest.raises(ValueError, match="unknown custom OpenAI-compatible endpoint 'missing'"):
        parse_llm_model_provider_overrides(
            '{"tool":{"google/gemma-4-31B-turbo-TEE":"custom-openai-compatible:missing"}}',
            custom_openai_compatible_endpoint_ids={"gemma4-cloud-run-turbo"},
        )


def test_parse_llm_model_provider_overrides_rejects_unknown_surface() -> None:
    with pytest.raises(ValueError, match="surface 'unknown' is not supported"):
        parse_llm_model_provider_overrides('{"unknown":{"sample-routed-model":"bedrock"}}')


def test_resolve_llm_route_falls_back_to_default_provider() -> None:
    route = resolve_llm_route(
        surface="generator",
        default_provider="vertex",
        model="sample-routed-model",
        overrides={},
        allowed_providers={"bedrock", "vertex"},
    )

    assert route == ResolvedLlmRoute(surface="generator", provider="vertex", model="sample-routed-model")


def test_resolve_llm_route_rejects_provider_not_allowed_for_surface() -> None:
    with pytest.raises(ValueError, match="reference override provider 'bedrock' is not supported"):
        resolve_llm_route(
            surface="reference",
            default_provider="vertex",
            model="sample-routed-model",
            overrides={"reference": {"sample-routed-model": "bedrock"}},
            allowed_providers={"vertex"},
        )


def test_resolve_llm_route_allows_custom_target_only_when_enabled() -> None:
    overrides = {
        "tool": {
            "google/gemma-4-31B-turbo-TEE": "custom-openai-compatible:gemma4-cloud-run-turbo",
            "Qwen/Qwen3.6-27B-TEE": "custom-openai-compatible:qwen36-cloud-run",
        }
    }

    route = resolve_llm_route(
        surface="tool",
        default_provider="chutes",
        model="google/gemma-4-31B-turbo-TEE",
        overrides=overrides,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )

    assert route == ResolvedLlmRoute(
        surface="tool",
        provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        model="google/gemma-4-31B-turbo-TEE",
    )
    qwen_route = resolve_llm_route(
        surface="tool",
        default_provider="chutes",
        model="Qwen/Qwen3.6-27B-TEE",
        overrides=overrides,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )
    assert qwen_route == ResolvedLlmRoute(
        surface="tool",
        provider="custom-openai-compatible:qwen36-cloud-run",
        model="Qwen/Qwen3.6-27B-TEE",
    )
    with pytest.raises(ValueError, match="not supported"):
        resolve_llm_route(
            surface="tool",
            default_provider="chutes",
            model="google/gemma-4-31B-turbo-TEE",
            overrides=overrides,
            allowed_providers={"chutes", "vertex"},
        )


def test_custom_route_target_is_canonicalized() -> None:
    parsed = parse_llm_model_provider_overrides(
        '{"tool":{"google/gemma-4-31B-turbo-TEE":"custom-openai-compatible: gemma4-cloud-run-turbo"}}',
        custom_openai_compatible_endpoint_ids={"gemma4-cloud-run-turbo"},
    )

    assert parsed["tool"]["google/gemma-4-31B-turbo-TEE"] == "custom-openai-compatible:gemma4-cloud-run-turbo"


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
def test_parse_llm_model_provider_overrides_rejects_internal_openrouter_target(model: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        parse_llm_model_provider_overrides(f'{{"tool":{{"{model}":"openrouter"}}}}')


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
def test_resolve_llm_route_routes_chutes_selected_openrouter_only_model_to_openrouter(model: str) -> None:
    route = resolve_llm_route(
        surface="tool",
        default_provider="chutes",
        model=model,
        overrides={},
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )

    assert route == ResolvedLlmRoute(surface="tool", provider="openrouter", model=model)


def test_resolve_llm_route_custom_gpt_oss_20b_override_wins_over_openrouter_fallback() -> None:
    route = resolve_llm_route(
        surface="tool",
        default_provider="chutes",
        model="openai/gpt-oss-20b",
        overrides={"tool": {"openai/gpt-oss-20b": "custom-openai-compatible:gpt-oss-20b-gke"}},
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )

    assert route == ResolvedLlmRoute(
        surface="tool",
        provider="custom-openai-compatible:gpt-oss-20b-gke",
        model="openai/gpt-oss-20b",
    )


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
def test_resolve_llm_route_routes_chutes_override_openrouter_only_model_to_openrouter(model: str) -> None:
    route = resolve_llm_route(
        surface="tool",
        default_provider="vertex",
        model=model,
        overrides={"tool": {model: "chutes"}},
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )

    assert route == ResolvedLlmRoute(surface="tool", provider="openrouter", model=model)


@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
def test_resolve_llm_route_does_not_special_case_non_chutes_selection(model: str) -> None:
    route = resolve_llm_route(
        surface="tool",
        default_provider="vertex",
        model=model,
        overrides={},
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
    )

    assert route == ResolvedLlmRoute(surface="tool", provider="vertex", model=model)


@dataclass(slots=True)
class _RecordingProvider:
    seen_requests: list[LlmRequest]

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.seen_requests.append(request)
        return LlmResponse(
            id="resp-1",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="output_text", text="ok"),),
                    ),
                ),
            ),
            usage=LlmUsage(),
            metadata={"raw_response": {"ok": True}},
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.anyio("asyncio")
async def test_routed_provider_rewrites_request_provider_before_delegating() -> None:
    delegate = _RecordingProvider(seen_requests=[])

    provider = RoutedLlmProvider(
        surface="generator",
        default_provider="vertex",
        overrides={"generator": {"sample-routed-model": "bedrock"}},
        allowed_providers={"bedrock", "vertex"},
        resolve_provider=lambda _: delegate,
    )

    response = await provider.invoke(
        LlmRequest(
            provider="vertex",
            model="sample-routed-model",
            messages=(),
            temperature=None,
            max_output_tokens=128,
        )
    )

    assert delegate.seen_requests[0].provider == "bedrock"
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "bedrock"
    assert response.metadata["effective_model"] == "sample-routed-model"
    assert response.metadata["selected_provider"] == "bedrock"
    assert response.metadata["selected_model"] == "sample-routed-model"


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("model", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"))
async def test_routed_provider_preserves_delegate_effective_route_metadata(model: str) -> None:
    delegate = _RecordingProvider(seen_requests=[])

    provider = RoutedLlmProvider(
        surface="tool",
        default_provider="chutes",
        overrides={},
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
        resolve_provider=lambda _: delegate,
    )

    response = await provider.invoke(
        LlmRequest(
            provider="chutes",
            model=model,
            messages=(),
            temperature=None,
            max_output_tokens=128,
        )
    )

    assert delegate.seen_requests[0].provider == "openrouter"
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "openrouter"
    assert response.metadata["effective_model"] == model
    assert response.metadata["selected_provider"] == "openrouter"
    assert response.metadata["selected_model"] == model
