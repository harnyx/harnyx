from __future__ import annotations

import pytest

from harnyx_commons.llm.tool_models import (
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    parse_miner_selected_llm_provider_model,
    resolve_tool_model,
    tool_model_thinking_capability,
)


def test_tool_model_thinking_capabilities_share_the_canonical_model_owner() -> None:
    deepseek = tool_model_thinking_capability("deepseek-ai/deepseek-v3.2-tee", provider_name="chutes")
    glm = tool_model_thinking_capability("zai-org/GLM-5-TEE", provider_name="vertex")
    qwen36_chutes = tool_model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="chutes",
    )
    qwen36 = tool_model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="custom-openai-compatible:qwen36-cloud-run",
    )
    gemma_chutes = tool_model_thinking_capability("google/gemma-4-31B-turbo-TEE", provider_name="chutes")
    gemma_custom = tool_model_thinking_capability(
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
    assert gemma_chutes is not None
    assert gemma_chutes.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert gemma_custom is not None
    assert gemma_custom.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
    assert tool_model_thinking_capability("openai/gpt-oss-20b", provider_name="chutes") is None
    assert tool_model_thinking_capability("openai/gpt-oss-120b", provider_name="chutes") is None
    assert tool_model_thinking_capability("openai/gpt-oss-20b", provider_name="openrouter") is None
    assert tool_model_thinking_capability("openai/gpt-oss-120b", provider_name="openrouter") is None


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
        "google/gemma-4-31b-it",
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
    ),
)
def test_miner_selected_ai_gateway_uses_native_model_ids_without_translation(model: str) -> None:
    resolved = parse_miner_selected_llm_provider_model(provider="ai_gateway", model=model)

    assert resolved.provider == "ai_gateway"
    assert resolved.model == model


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
