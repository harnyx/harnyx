from __future__ import annotations

import pytest

from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmUsage
from caster_commons.observability import langfuse

_LANGFUSE_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


@pytest.fixture(autouse=True)
def _reset_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langfuse, "_LANGFUSE_CLIENT", None)


def _request() -> LlmRequest:
    return LlmRequest(
        provider="openai",
        model="gpt-5-mini",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
    )


def test_read_config_returns_none_when_all_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LANGFUSE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    assert langfuse._read_config() is None


def test_read_config_raises_runtime_error_for_partial_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Langfuse configuration is partial"):
        langfuse._read_config()


def test_read_config_returns_mapping_for_full_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", " https://langfuse.example ")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", " pk-test ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", " sk-test ")

    assert langfuse._read_config() == {
        "LANGFUSE_HOST": "https://langfuse.example",
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
    }


def test_start_llm_generation_returns_none_scope_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _LANGFUSE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    scope = langfuse.start_llm_generation(
        trace_id="trace-id",
        provider_label="openai",
        request=_request(),
    )
    with scope as generation:
        assert generation is None


def test_update_generation_best_effort_swallows_update_exception() -> None:
    class RaisingGeneration:
        def update(self, **kwargs: object) -> None:
            raise RuntimeError("update failed")

    langfuse.update_generation_best_effort(
        RaisingGeneration(),
        output={"ok": True},
        usage=LlmUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        metadata={"provider": "openai"},
    )
