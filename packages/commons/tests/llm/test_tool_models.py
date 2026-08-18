from __future__ import annotations

import pytest

from harnyx_commons.llm.tool_models import (
    ALLOWED_TOOL_MODELS,
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    TOOL_MODEL_THINKING_CAPABILITIES,
    ToolModelThinkingCapability,
    ToolModelThinkingField,
    ToolModelThinkingProvider,
    model_thinking_capability,
    parse_miner_selected_llm_provider_model,
    parse_tool_model,
    resolve_tool_model,
    tool_model_thinking_capability,
)


def test_tool_model_thinking_capabilities_share_the_canonical_model_owner() -> None:
    deepseek = model_thinking_capability("deepseek-ai/deepseek-v3.2-tee", provider_name="chutes")
    glm = model_thinking_capability("zai-org/GLM-5-TEE", provider_name="vertex")
    qwen36_chutes = model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="chutes",
    )
    qwen38_chutes = model_thinking_capability(
        "Qwen/Qwen3.8-27B-TEE",
        provider_name="chutes",
    )
    qwen36 = model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="custom-openai-compatible:qwen36-cloud-run",
    )
    gemma_chutes = model_thinking_capability("google/gemma-4-31B-turbo-TEE", provider_name="chutes")
    gemma_custom = model_thinking_capability(
        "google/gemma-4-31B-turbo-TEE",
        provider_name="custom-openai-compatible:gemma4-cloud-run-turbo",
    )

    assert resolve_tool_model("deepseek-ai/deepseek-v3.2-tee") == "deepseek-ai/DeepSeek-V3.2-TEE"
    assert resolve_tool_model("openai/gpt-oss-20b") == "openai/gpt-oss-20b"
    assert resolve_tool_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert resolve_tool_model("qwen/qwen3.6-27b-tee") == "Qwen/Qwen3.6-27B-TEE"
    assert resolve_tool_model("Qwen/Qwen3-Next-80B-A3B-Instruct") is None
    assert resolve_tool_model("deepseek-ai/deepseek-v3.1-tee") is None
    assert deepseek is not None
    assert deepseek.chat_template_kwargs(enabled=True) == {"thinking": True}
    assert glm is not None
    assert glm.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert qwen36 is not None
    assert qwen36.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert qwen36_chutes is not None
    assert qwen36_chutes.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
    assert qwen38_chutes is not None
    assert qwen38_chutes.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert gemma_chutes is not None
    assert gemma_chutes.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert gemma_custom is not None
    assert gemma_custom.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
    assert model_thinking_capability("openai/gpt-oss-20b", provider_name="chutes") is None
    assert model_thinking_capability("openai/gpt-oss-120b", provider_name="chutes") is None
    assert model_thinking_capability("openai/gpt-oss-20b", provider_name="openrouter") is None
    assert model_thinking_capability("openai/gpt-oss-120b", provider_name="openrouter") is None


def test_benchmark_model_thinking_capability_does_not_expand_tool_authorization() -> None:
    model = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"

    assert model not in ALLOWED_TOOL_MODELS
    assert model not in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model(model)
    with pytest.raises(ValueError, match="not supported for miner-selected provider 'chutes'"):
        parse_miner_selected_llm_provider_model(provider="chutes", model=model)
    capability = model_thinking_capability(model.lower(), provider_name="chutes")
    assert capability is not None
    assert capability.chat_template_kwargs(enabled=True) == {"enable_thinking": True}


def test_legacy_tool_thinking_exports_remain_authorization_restricted() -> None:
    field: ToolModelThinkingField = "chat_template_kwargs.enable_thinking"
    provider: ToolModelThinkingProvider = "chutes"
    capability = ToolModelThinkingCapability(field)

    assert capability.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
    assert set(TOOL_MODEL_THINKING_CAPABILITIES) <= set(ALLOWED_TOOL_MODELS)
    assert tool_model_thinking_capability(
        "google/gemma-4-31B-turbo-TEE",
        provider_name=provider,
    ) == model_thinking_capability(
        "google/gemma-4-31B-turbo-TEE",
        provider_name=provider,
    )
    assert (
        tool_model_thinking_capability(
            "deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
            provider_name=provider,
        )
        is None
    )


def test_miner_selected_chutes_supports_only_chutes_models() -> None:
    assert (
        parse_miner_selected_llm_provider_model(
            provider="chutes",
            model="deepseek-ai/DeepSeek-V3.2-TEE",
        ).model
        == "deepseek-ai/DeepSeek-V3.2-TEE"
    )
    assert (
        parse_miner_selected_llm_provider_model(
            provider=" chutes ",
            model=" Qwen/Qwen3.6-27B-TEE ",
        ).model
        == "Qwen/Qwen3.6-27B-TEE"
    )
    assert (
        parse_miner_selected_llm_provider_model(
            provider="chutes",
            model="moonshotai/Kimi-K2.6-TEE",
        ).model
        == "moonshotai/Kimi-K2.6-TEE"
    )
    assert (
        parse_miner_selected_llm_provider_model(
            provider="chutes",
            model="Qwen/Qwen3.8-27B-TEE",
        ).model
        == "Qwen/Qwen3.8-27B-TEE"
    )


def test_miner_selected_chutes_rejects_openrouter_only_models() -> None:
    for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "deepseek/deepseek-v3.2"):
        with pytest.raises(ValueError, match="not supported for miner-selected provider 'chutes'"):
            parse_miner_selected_llm_provider_model(provider="chutes", model=model)


@pytest.mark.parametrize(
    "model",
    (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "deepseek/deepseek-v3.2",
        "z-ai/glm-5",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro",
        "z-ai/glm-5.2",
        "thinkingmachines/inkling",
        "qwen/qwen3.5-397b-a17b",
        "meta/muse-glimmer-30b",
    ),
)
def test_miner_selected_openrouter_uses_native_model_ids_without_translation(
    model: str,
) -> None:
    resolved = parse_miner_selected_llm_provider_model(provider="openrouter", model=model)

    assert resolved.provider == "openrouter"
    assert resolved.model == model


@pytest.mark.parametrize(
    "model",
    (
        "thinkingmachines/inkling",
        "zai/glm-5.2-fast",
        "openai/gpt-oss-20b",
        "zai/glm-4.7",
        "google/gemma-4-31b-it",
        "openai/gpt-oss-120b",
        "minimax/minimax-m2.7",
        "zai/glm-4.7-flash",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro",
        "meta/muse-glimmer-30b",
        "alibaba/qwen3.8-27b",
    ),
)
def test_miner_selected_ai_gateway_uses_native_model_ids_without_translation(model: str) -> None:
    resolved = parse_miner_selected_llm_provider_model(provider="ai_gateway", model=model)

    assert resolved.provider == "ai_gateway"
    assert resolved.model == model


def test_new_chutes_provider_models_are_not_internal_canonical_models() -> None:
    for model in (
        "moonshotai/Kimi-K2.6-TEE",
        "zai-org/GLM-5.2-TEE",
        "Qwen/Qwen3.5-397B-A17B-TEE",
    ):
        assert model in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]
        assert model not in ALLOWED_TOOL_MODELS
        with pytest.raises(ValueError, match="not allowed for validator tools"):
            parse_tool_model(model)
        assert model_thinking_capability(model, provider_name="chutes") is None
        assert model_thinking_capability(model, provider_name="vertex") is None
        assert model_thinking_capability(model, provider_name="custom-openai-compatible") is None


def test_qwen38_chutes_model_does_not_expand_internal_tool_authorization() -> None:
    model = "Qwen/Qwen3.8-27B-TEE"

    assert model in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]
    assert model not in ALLOWED_TOOL_MODELS
    with pytest.raises(ValueError, match="not allowed for validator tools"):
        parse_tool_model(model)


def test_retired_chutes_model_is_not_in_miner_selected_namespace() -> None:
    assert "zai-org/GLM-5-TEE" not in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]


def test_miner_selected_ai_gateway_rejects_retired_qwen37_plus() -> None:
    with pytest.raises(ValueError, match="not supported for miner-selected provider 'ai_gateway'"):
        parse_miner_selected_llm_provider_model(
            provider="ai_gateway",
            model="alibaba/qwen3.7-plus",
        )


def test_miner_selected_provider_model_sets_are_provider_namespaces() -> None:
    assert set(MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]).isdisjoint(
        MINER_SELECTED_LLM_PROVIDER_MODELS["openrouter"]
    )
    assert set(MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]).isdisjoint(
        MINER_SELECTED_LLM_PROVIDER_MODELS["ai_gateway"]
    )


def test_miner_selected_openrouter_rejects_chutes_model_ids() -> None:
    for model in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]:
        with pytest.raises(ValueError, match="not supported for miner-selected provider 'openrouter'"):
            parse_miner_selected_llm_provider_model(provider="openrouter", model=model)


def test_miner_selected_ai_gateway_rejects_chutes_model_ids() -> None:
    for model in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]:
        with pytest.raises(ValueError, match="not supported for miner-selected provider 'ai_gateway'"):
            parse_miner_selected_llm_provider_model(provider="ai_gateway", model=model)


def test_miner_selected_openrouter_supports_openrouter_only_gpt_models() -> None:
    assert (
        parse_miner_selected_llm_provider_model(
            provider="openrouter",
            model="openai/gpt-oss-20b",
        ).provider
        == "openrouter"
    )


def test_miner_selected_ai_gateway_rejects_openrouter_only_non_gateway_models() -> None:
    for model in ("deepseek/deepseek-v3.2", "z-ai/glm-5", "qwen/qwen3.6-27b"):
        with pytest.raises(ValueError, match="not supported for miner-selected provider 'ai_gateway'"):
            parse_miner_selected_llm_provider_model(provider="ai_gateway", model=model)


def test_openrouter_native_model_ids_are_not_valid_for_chutes() -> None:
    with pytest.raises(ValueError, match="not supported for miner-selected provider 'chutes'"):
        parse_miner_selected_llm_provider_model(provider="chutes", model="qwen/qwen3.6-27b")


def test_unknown_miner_selected_llm_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="miner-selected llm provider 'vertex' is not supported"):
        parse_miner_selected_llm_provider_model(provider="vertex", model="openai/gpt-oss-20b")
