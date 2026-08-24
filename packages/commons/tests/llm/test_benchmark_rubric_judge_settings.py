from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.config.benchmark_rubric_judge import BenchmarkRubricJudgeLlmSettings


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": "vertex"},
    ],
)
def test_benchmark_rubric_judge_settings_require_provider_and_model_together(
    payload: dict[str, object],
) -> None:
    with pytest.raises(
        ValidationError,
        match="BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER and BENCHMARK_RUBRIC_JUDGE_LLM_MODEL must be set together",
    ):
        BenchmarkRubricJudgeLlmSettings.model_validate(payload)


def test_benchmark_rubric_judge_settings_strip_model() -> None:
    settings = BenchmarkRubricJudgeLlmSettings.model_validate({"provider": "vertex", "model": " gemini-3-pro-preview "})

    assert settings.model == "gemini-3-pro-preview"


def test_benchmark_rubric_judge_settings_do_not_fallback_from_benchmark_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("BENCHMARK_LLM_MODEL", "benchmark-model")
    monkeypatch.delenv("BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("BENCHMARK_RUBRIC_JUDGE_LLM_MODEL", raising=False)

    settings = BenchmarkRubricJudgeLlmSettings(_env_file=None)

    assert settings.provider is None
    assert settings.model == ""


def test_benchmark_rubric_judge_settings_ignore_max_output_tokens_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_MODEL", "rubric-model")
    monkeypatch.setenv("BENCHMARK_RUBRIC_JUDGE_LLM_MAX_OUTPUT_TOKENS", "256")

    settings = BenchmarkRubricJudgeLlmSettings(_env_file=None)

    assert not hasattr(settings, "max_output_tokens")
