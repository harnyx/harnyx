from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pytest

from harnyx_commons.clients import CHUTES
from harnyx_commons.llm.provider_factory import (
    build_cached_llm_provider_registry,
    build_routed_llm_provider,
)
from harnyx_commons.llm.providers.openrouter import OPENROUTER_BASE_URL
from harnyx_commons.llm.retry_utils import RetryPolicy
from harnyx_validator.application.similarity_judge import (
    SimilarityJudge,
    SimilarityJudgeConfig,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.settings import Settings
from validator.tests.benchmark.similarity_judge_benchmark import (
    BenchmarkIdentity,
    InvocationRecordingProvider,
    load_cases,
    run_benchmark,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.expensive,
    pytest.mark.similarity_judge_benchmark,
    pytest.mark.anyio("asyncio"),
]

_CASES_PATH = (
    Path(__file__).parent / "data" / "similarity_judge_benchmark_cases.jsonl"
)
_OUTPUT_ROOT = Path(".localdev/similarity-judge-benchmark")
_GEMMA_MODEL = "google/gemma-4-31B-turbo-TEE"
_GEMMA_ENDPOINT_ID = "gemma4-cloud-run-turbo"
_GEMMA_ROUTE_TARGET = f"custom-openai-compatible:{_GEMMA_ENDPOINT_ID}"
_GEMMA_SERVICE_URL = "https://gemma-4-31b-turbo-obbrpx3ppa-uc.a.run.app"
_GLM_MODEL = "zai-org/GLM-5.2-TEE"
_KIMI_MODEL = "moonshotai/Kimi-K3-TEE"
_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731-TEE"


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    test_id: str
    model: str
    route_target: str
    endpoint_id: str
    normalized_base_url: str
    endpoint_config: dict[str, object] | None = None

    @property
    def provider_overrides(self) -> dict[str, object]:
        if self.route_target == "chutes":
            return {}
        return {"duplication_detection": {self.model: self.route_target}}


def _gemma_cloud_run_endpoint_config() -> dict[str, object]:
    return {
        "id": _GEMMA_ENDPOINT_ID,
        "base_url": f"{_GEMMA_SERVICE_URL}/v1",
        "auth": {
            "type": "google_id_token",
            "audience": _GEMMA_SERVICE_URL,
            "credential_source": "service_account_json_b64_env",
            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        },
    }


_BENCHMARK_TARGETS = (
    BenchmarkTarget(
        test_id="gemma4-cloud-run",
        model=_GEMMA_MODEL,
        route_target=_GEMMA_ROUTE_TARGET,
        endpoint_id=_GEMMA_ENDPOINT_ID,
        normalized_base_url=f"{_GEMMA_SERVICE_URL}/v1",
        endpoint_config=_gemma_cloud_run_endpoint_config(),
    ),
    BenchmarkTarget(
        test_id="glm-5.2-chutes",
        model=_GLM_MODEL,
        route_target="chutes",
        endpoint_id="chutes",
        normalized_base_url=CHUTES.base_url,
    ),
    BenchmarkTarget(
        test_id="kimi-k3-chutes",
        model=_KIMI_MODEL,
        route_target="chutes",
        endpoint_id="chutes",
        normalized_base_url=CHUTES.base_url,
    ),
    BenchmarkTarget(
        test_id="deepseek-v4-flash-0731-openrouter",
        model=_DEEPSEEK_MODEL,
        route_target="openrouter",
        endpoint_id="openrouter",
        normalized_base_url=OPENROUTER_BASE_URL,
    ),
)


def _repository_sha() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git executable is required to record the benchmark repository SHA")
    completed = subprocess.run(  # noqa: S603 - fixed git command, resolved executable
        [git_executable, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_source_hashes() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[4]
    relative_paths = (
        "Taskfile.yml",
        "pytest.ini",
        "public/pytest.ini",
        "apps/platform/src/harnyx_platform/application/services/miner_task_similarity.py",
        "public/packages/commons/src/harnyx_commons/llm/providers/chutes.py",
        "public/packages/commons/src/harnyx_commons/llm/providers/chutes_codec.py",
        "public/packages/commons/src/harnyx_commons/llm/providers/openrouter.py",
        "public/packages/commons/src/harnyx_commons/llm/providers/thinking.py",
        "public/packages/commons/src/harnyx_commons/llm/tool_models.py",
        "scripts/test/run_integration_tests.sh",
        "public/validator/src/harnyx_validator/application/similarity_judge.py",
        "public/validator/src/harnyx_validator/runtime/bootstrap.py",
        "public/validator/tests/benchmark/data/similarity_judge_benchmark_cases.jsonl",
        "public/validator/tests/benchmark/similarity_judge_benchmark.py",
        "public/validator/tests/benchmark/test_similarity_judge_benchmark_live.py",
    )
    return {
        relative_path: _file_sha256(repo_root / relative_path)
        for relative_path in relative_paths
    }


@pytest.mark.parametrize(
    "target",
    _BENCHMARK_TARGETS,
    ids=lambda target: target.test_id,
)
async def test_fixed_dataset_similarity_benchmark(target: BenchmarkTarget) -> None:
    base_settings = Settings.load()
    settings = base_settings.model_copy(
        update={
            "llm": base_settings.llm.model_copy(
                update={
                    "openai_compatible_endpoints_json": json.dumps(
                        [] if target.endpoint_config is None else [target.endpoint_config]
                    ),
                    "llm_model_provider_overrides_json": json.dumps(
                        target.provider_overrides
                    ),
                    "similarity_llm_provider": "chutes",
                    "similarity_llm_model_override": target.model,
                }
            )
        }
    )
    similarity_route = bootstrap._resolve_similarity_judge_route(settings)
    assert similarity_route.provider == target.route_target
    assert similarity_route.model == target.model

    registry = build_cached_llm_provider_registry(
        llm_settings=settings.llm,
        bedrock_settings=settings.bedrock,
        vertex_settings=settings.vertex,
    )
    routed_provider = build_routed_llm_provider(
        surface="duplication_detection",
        default_provider=settings.llm.similarity_llm_provider,
        llm_settings=settings.llm,
        allowed_providers={"bedrock", "chutes", "openrouter", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )
    recording_provider = InvocationRecordingProvider(routed_provider)
    judge = SimilarityJudge(
        llm_provider=recording_provider,
        config=SimilarityJudgeConfig(
            provider=settings.llm.similarity_llm_provider,
            model=similarity_route.model,
            fallback_models=(),
            temperature=0.0,
            max_output_tokens=settings.llm.similarity_llm_max_output_tokens,
            reasoning_effort=bootstrap._SCORING_LLM_REASONING_EFFORT,
            timeout_seconds=float(settings.llm.similarity_llm_timeout_seconds),
            retry_policy=RetryPolicy(
                attempts=1,
                initial_ms=0,
                max_ms=0,
                jitter=0.0,
            ),
            request_extra_by_model=bootstrap._similarity_request_extra_by_model(
                (similarity_route,)
            ),
        ),
    )
    identity = BenchmarkIdentity(
        repository_sha=_repository_sha(),
        validator_package_version=version("harnyx-validator"),
        requested_model=target.model,
        route_target=target.route_target,
        endpoint_id=target.endpoint_id,
        normalized_base_url=target.normalized_base_url,
        immutable_serving_revision=None,
        benchmark_source_sha256=_benchmark_source_hashes(),
        temperature=0.0,
        retry_attempts=1,
        fallback_models=(),
    )

    try:
        summary = await run_benchmark(
            groups=load_cases(_CASES_PATH),
            judge=judge,
            recording_provider=recording_provider,
            output_root=_OUTPUT_ROOT,
            identity=identity,
        )
    finally:
        await registry.aclose()

    print(summary.model_dump_json(indent=2))
    assert summary.provider_failure_count == 0
    assert summary.pairwise_overclassification_count == 0
    assert summary.final_overclassification_count == 0
    assert summary.pairwise_multi_step_underclassification_count == 0
    assert summary.final_multi_step_underclassification_count == 0
    assert summary.false_novel_count == 0
