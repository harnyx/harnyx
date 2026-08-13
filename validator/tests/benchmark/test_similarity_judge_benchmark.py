from __future__ import annotations

import ast
import builtins
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from harnyx_commons.llm.provider import LlmProviderError, LlmRetryExhaustedError
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.llm.tool_models import parse_miner_selected_llm_provider_model
from harnyx_commons.miner_task_similarity import SimilarityJudgeRequest, SimilarityJudgeResult
from harnyx_miner_sdk.query import Query, Response
from harnyx_miner_sdk.tools.search_models import FetchPageResponse, SearchWebSearchResponse
from validator.tests.benchmark.similarity_judge_benchmark import (
    FINAL_LABELS,
    PAIRWISE_LABELS,
    BenchmarkCandidateGroup,
    BenchmarkIdentity,
    InvocationRecordingProvider,
    PairwiseGold,
    _multi_step_underclassification_count,
    _overclassification_count,
    aggregate_classification,
    eligible_comparisons,
    load_cases,
    normalize_pairwise,
    run_benchmark,
    summarize_metrics,
)

_CASES_PATH = (
    Path(__file__).parent / "data" / "similarity_judge_benchmark_cases.jsonl"
)
_REALISM_SAMPLE_PATH = (
    Path(__file__).parent / "data" / "similarity_judge_realism_sample.json"
)
_EXPECTED_ARTIFACT_IDS = {
    "duplicate-model-and-rename": (
        UUID("20000000-0000-4000-8000-000000000001"),
        UUID("30000000-0000-4000-8000-000000000001"),
        UUID("30000000-0000-4000-8000-000000000002"),
        UUID("30000000-0000-4000-8000-000000000003"),
    ),
    "duplicate-unused-component": (
        UUID("20000000-0000-4000-8000-000000000002"),
        UUID("30000000-0000-4000-8000-000000000004"),
        UUID("30000000-0000-4000-8000-000000000005"),
    ),
    "duplicate-with-fallback-boundary": (
        UUID("20000000-0000-4000-8000-000000000003"),
        UUID("30000000-0000-4000-8000-000000000006"),
        UUID("30000000-0000-4000-8000-000000000007"),
    ),
    "near-extra-ranking-tool": (
        UUID("20000000-0000-4000-8000-000000000004"),
        UUID("30000000-0000-4000-8000-000000000008"),
        UUID("30000000-0000-4000-8000-000000000009"),
        UUID("30000000-0000-4000-8000-000000000010"),
    ),
    "near-production-lane-b-payload-guard": (
        UUID("20000000-0000-4000-8000-000000000005"),
        UUID("30000000-0000-4000-8000-000000000011"),
        UUID("30000000-0000-4000-8000-000000000012"),
        UUID("30000000-0000-4000-8000-000000000033"),
    ),
    "near-local-verification": (
        UUID("20000000-0000-4000-8000-000000000006"),
        UUID("30000000-0000-4000-8000-000000000013"),
        UUID("30000000-0000-4000-8000-000000000014"),
    ),
    "notable-explicit-stages": (
        UUID("20000000-0000-4000-8000-000000000007"),
        UUID("30000000-0000-4000-8000-000000000015"),
        UUID("30000000-0000-4000-8000-000000000016"),
    ),
    "notable-parallel-retrieval": (
        UUID("20000000-0000-4000-8000-000000000008"),
        UUID("30000000-0000-4000-8000-000000000017"),
        UUID("30000000-0000-4000-8000-000000000018"),
        UUID("30000000-0000-4000-8000-000000000019"),
    ),
    "notable-reordered-same-root": (
        UUID("20000000-0000-4000-8000-000000000009"),
        UUID("30000000-0000-4000-8000-000000000020"),
        UUID("30000000-0000-4000-8000-000000000021"),
    ),
    "novel-json-contract-solver": (
        UUID("20000000-0000-4000-8000-000000000010"),
        UUID("30000000-0000-4000-8000-000000000022"),
        UUID("30000000-0000-4000-8000-000000000023"),
    ),
    "novel-vfs-investigation-state": (
        UUID("20000000-0000-4000-8000-000000000011"),
        UUID("30000000-0000-4000-8000-000000000024"),
        UUID("30000000-0000-4000-8000-000000000025"),
    ),
    "novel-entity-attribute-graph": (
        UUID("20000000-0000-4000-8000-000000000012"),
        UUID("30000000-0000-4000-8000-000000000026"),
        UUID("30000000-0000-4000-8000-000000000027"),
    ),
    "duplicate-prompt-only": (
        UUID("20000000-0000-4000-8000-000000000013"),
        UUID("30000000-0000-4000-8000-000000000029"),
        UUID("30000000-0000-4000-8000-000000000030"),
    ),
    "notable-unreachable-parallel-controller": (
        UUID("20000000-0000-4000-8000-000000000014"),
        UUID("30000000-0000-4000-8000-000000000031"),
        UUID("30000000-0000-4000-8000-000000000032"),
    ),
    "notable-production-conflict-ledger": (
        UUID("20000000-0000-4000-8000-000000000015"),
        UUID("30000000-0000-4000-8000-000000000034"),
        UUID("30000000-0000-4000-8000-000000000035"),
    ),
    "near-production-constraint-ledger": (
        UUID("20000000-0000-4000-8000-000000000016"),
        UUID("30000000-0000-4000-8000-000000000036"),
        UUID("30000000-0000-4000-8000-000000000037"),
    ),
    "notable-production-draft-audit-repair": (
        UUID("20000000-0000-4000-8000-000000000017"),
        UUID("30000000-0000-4000-8000-000000000038"),
        UUID("30000000-0000-4000-8000-000000000039"),
    ),
}
_SDK_BOUNDARY_CALLS = {"fetch_page", "llm_chat", "search_web", "tooling_info"}


@dataclass(frozen=True)
class _PostprocessFailure:
    response: LlmResponse
    error: Exception


class _ScriptedProvider:
    def __init__(self) -> None:
        self.next_outcome: LlmResponse | Exception | None = None
        self.requests: list[AbstractLlmRequest] = []

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        self.requests.append(request)
        outcome = self.next_outcome
        self.next_outcome = None
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            raise RuntimeError("test did not configure a provider outcome")
        return outcome

    async def aclose(self) -> None:
        return None


class _ScriptedJudge:
    def __init__(
        self,
        *,
        provider: InvocationRecordingProvider,
        delegate: _ScriptedProvider,
        classifications: Mapping[UUID, PairwiseGold],
        outcomes: Mapping[UUID, LlmResponse | Exception | _PostprocessFailure] | None = None,
    ) -> None:
        self._provider = provider
        self._delegate = delegate
        self._classifications = classifications
        self._outcomes = dict(outcomes or {})

    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult:
        outcome = self._outcomes.get(
            request.reference_artifact_id,
            _response(response_id=str(request.reference_artifact_id)),
        )
        if isinstance(outcome, _PostprocessFailure):
            self._delegate.next_outcome = outcome.response
        else:
            self._delegate.next_outcome = outcome
        response = await self._provider.invoke(_llm_request(request))
        if isinstance(outcome, _PostprocessFailure):
            raise outcome.error
        pairwise = self._classifications[request.reference_artifact_id]
        raw_classification = "novel" if pairwise == "architectural_replacement" else pairwise
        return SimilarityJudgeResult(
            classification=raw_classification,
            reasoning=f"scripted {pairwise}",
            reasoning_tokens=response.usage.reasoning_tokens,
            model="google/gemma-4-31B-turbo-TEE",
            provider="custom-openai-compatible:gemma4-cloud-run-turbo",
        )


def _llm_request(request: SimilarityJudgeRequest) -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model="google/gemma-4-31B-turbo-TEE",
        messages=(
            LlmMessage(
                role="system",
                content=(LlmMessageContentPart.input_text("similarity benchmark prompt"),),
            ),
            LlmMessage(
                role="user",
                content=(
                    LlmMessageContentPart.input_text(
                        json.dumps({"reference_artifact_id": str(request.reference_artifact_id)})
                    ),
                ),
            ),
        ),
        output_mode="text",
        temperature=0.0,
        max_output_tokens=256,
        retry_policy=None,
        use_case="miner_task_similarity_judge",
    )


def _response(*, response_id: str = "response-1") -> LlmResponse:
    return LlmResponse(
        id=response_id,
        choices=(),
        usage=LlmUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        metadata={
            "selected_provider": "custom-openai-compatible:gemma4-cloud-run-turbo",
            "selected_model": "google/gemma-4-31B-turbo-TEE",
            "raw_response": {"id": response_id, "model": "served-gemma-revision"},
        },
        postprocessed={"classification": "duplicate"},
        finish_reason="stop",
    )


def _identity() -> BenchmarkIdentity:
    return BenchmarkIdentity(
        repository_sha="test-sha",
        validator_package_version="test-version",
        requested_model="google/gemma-4-31B-turbo-TEE",
        route_target="custom-openai-compatible:gemma4-cloud-run-turbo",
        endpoint_id="gemma4-cloud-run-turbo",
        normalized_base_url="https://gemma.example/v1",
        immutable_serving_revision=None,
        benchmark_source_sha256={"dataset": "test-dataset-sha"},
        temperature=0.0,
        retry_attempts=1,
        fallback_models=(),
    )


def _groups() -> tuple[BenchmarkCandidateGroup, ...]:
    return load_cases(_CASES_PATH)


def _classifications(
    groups: tuple[BenchmarkCandidateGroup, ...],
) -> dict[UUID, PairwiseGold]:
    return {
        reference.artifact_id: reference.expected_pairwise
        for group in groups
        for reference in eligible_comparisons(group)
        if reference.expected_pairwise is not None
    }


async def _run(
    tmp_path: Path,
    *,
    classifications: Mapping[UUID, PairwiseGold] | None = None,
    outcomes: Mapping[UUID, LlmResponse | Exception | _PostprocessFailure] | None = None,
):
    groups = _groups()
    scripted_classifications = _classifications(groups)
    if classifications is not None:
        scripted_classifications.update(classifications)
    delegate = _ScriptedProvider()
    recording_provider = InvocationRecordingProvider(delegate)
    judge = _ScriptedJudge(
        provider=recording_provider,
        delegate=delegate,
        classifications=scripted_classifications,
        outcomes=outcomes,
    )
    summary = await run_benchmark(
        groups=groups,
        judge=judge,
        recording_provider=recording_provider,
        output_root=tmp_path,
        identity=_identity(),
    )
    return summary, delegate, groups


def _pair_rows(summary) -> list[dict[str, object]]:
    path = Path(summary.run_directory) / "pair_results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_checked_in_dataset_protects_gold_coverage_and_aggregation_contract() -> None:
    groups = _groups()

    assert len(groups) >= 12
    assert set(group.expected_final for group in groups) == set(FINAL_LABELS)
    assert all(len(eligible_comparisons(group)) >= 2 for group in groups)
    assert all(
        group.expected_final
        == aggregate_classification(
            [
                reference.expected_pairwise
                for reference in eligible_comparisons(group)
                if reference.expected_pairwise is not None
            ]
        )
        for group in groups
    )


def test_checked_in_dataset_has_realistic_executable_artifact_contract() -> None:
    groups = _groups()
    actual_artifact_ids = {
        group.case_id: (
            group.candidate.artifact_id,
            *(reference.artifact_id for reference in group.references),
        )
        for group in groups
    }
    references = [reference for group in groups for reference in group.references]
    eligible = [
        reference
        for group in groups
        for reference in eligible_comparisons(group)
    ]
    pairwise_support = Counter(
        reference.expected_pairwise for reference in eligible
    )
    final_support = Counter(group.expected_final for group in groups)

    assert actual_artifact_ids == _EXPECTED_ARTIFACT_IDS
    assert len(references) == 38
    assert len(eligible) == 34
    assert all(7 <= pairwise_support[label] <= 10 for label in PAIRWISE_LABELS)
    assert all(3 <= final_support[label] <= 6 for label in FINAL_LABELS)

    artifacts = [group.candidate for group in groups]
    artifacts.extend(references)
    assert len(artifacts) == 55
    for artifact in artifacts:
        assert "PROFILE_LABEL" not in artifact.script
        assert not any(group.case_id in artifact.script for group in groups)
        compile(artifact.script, str(artifact.artifact_id), "exec")
        module = ast.parse(artifact.script)
        imported_names = {
            alias.asname or alias.name
            for node in module.body
            if isinstance(node, ast.ImportFrom)
            and node.module in {
                "harnyx_miner_sdk.api",
                "harnyx_miner_sdk.decorators",
                "harnyx_miner_sdk.query",
            }
            for alias in node.names
        }
        assert {"Query", "Response", "entrypoint"} <= imported_names
        assert imported_names & _SDK_BOUNDARY_CALLS

        string_constants = {
            target.id: node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        llm_chat_calls = [
            call
            for call in ast.walk(module)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "llm_chat"
        ]
        assert llm_chat_calls
        for call in llm_chat_calls:
            keywords = {
                keyword.arg: keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
            }
            provider = _static_string(keywords.get("provider"), string_constants)
            model = _static_string(keywords.get("model"), string_constants)
            selected_model = parse_miner_selected_llm_provider_model(
                provider=provider,
                model=model,
            )
            assert selected_model.provider == provider
            assert selected_model.model == model

        definitions = {
            node.name: node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        top_level_names = set(definitions)
        top_level_names.update(
            node.name for node in module.body if isinstance(node, ast.ClassDef)
        )
        top_level_names.update(imported_names)
        unresolved_bare_calls = {
            call.func.id
            for call in ast.walk(module)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id not in top_level_names
            and not hasattr(builtins, call.func.id)
        }
        assert unresolved_bare_calls == set()
        entrypoints = [
            node
            for node in definitions.values()
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "query"
            and _has_entrypoint_decorator(node)
        ]
        assert len(entrypoints) == 1
        entrypoint = entrypoints[0]
        assert _annotation_name(entrypoint.args.args[0].annotation) == "Query"
        assert _annotation_name(entrypoint.returns) == "Response"

        call_graph = _top_level_function_call_graph(artifact.script)
        reachable_names = {"query", *_reachable_functions(call_graph, "query")}
        reachable_nodes = [definitions[name] for name in reachable_names]
        sdk_calls = [
            call
            for function in reachable_nodes
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in _SDK_BOUNDARY_CALLS
        ]
        assert len(sdk_calls) >= 2
        assert any(
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for function in reachable_nodes
            for node in ast.walk(function)
        )
        assert any(
            isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try))
            for function in reachable_nodes
            for node in ast.walk(function)
        )
        assert any(
            isinstance(call.func, ast.Name) and call.func.id == "Response"
            for function in reachable_nodes
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
        )


def test_realism_sample_retains_auditable_production_provenance() -> None:
    sample = json.loads(_REALISM_SAMPLE_PATH.read_text())
    artifacts = sample["artifacts"]

    assert sample["source_batch_id"] == "cd9101db-44f8-518a-bcb8-eadf6f656dca"
    assert sample["population_artifact_count"] == 262
    assert sample["selected_ranks"] == [1, 34, 67, 99, 132, 165, 197, 230, 262]
    assert [artifact["rank"] for artifact in artifacts] == sample["selected_ranks"]
    assert len({artifact["artifact_id"] for artifact in artifacts}) == 9
    assert all(len(artifact["content_hash"]) == 64 for artifact in artifacts)
    assert min(artifact["size_bytes"] for artifact in artifacts) == 47_980
    assert max(artifact["size_bytes"] for artifact in artifacts) == 682_847


@pytest.mark.anyio
async def test_checked_in_artifact_ordinary_paths_execute_with_sdk_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def search_web(
        search_queries: str,
        /,
        *,
        provider: str,
        num: int | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        assert search_queries
        assert provider == "parallel"
        assert num is not None
        return SimpleNamespace(
            response=SearchWebSearchResponse(
                data=[
                    {"link": "https://example.test/one", "snippet": "First source"},
                    {"link": "https://example.test/two", "snippet": "Second source"},
                ]
            )
        )

    async def fetch_page(
        url: str,
        /,
        *,
        provider: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        assert provider == "parallel"
        return SimpleNamespace(
            response=FetchPageResponse(
                data=[{"url": url, "content": f"Evidence fetched from {url}"}]
            )
        )

    async def llm_chat(
        *,
        provider: str,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        assert messages
        selected_model = parse_miner_selected_llm_provider_model(
            provider=provider,
            model=model,
        )
        assert selected_model.provider == provider
        assert selected_model.model == model
        return SimpleNamespace(llm=SimpleNamespace(raw_text="KEEP"))

    monkeypatch.setattr(
        "harnyx_miner_sdk.decorators.entrypoint",
        lambda _name: lambda function: function,
    )
    artifacts = [group.candidate for group in _groups()]
    artifacts.extend(reference for group in _groups() for reference in group.references)
    for artifact in artifacts:
        namespace: dict[str, object] = {}
        exec(  # noqa: S102 - benchmark artifacts are checked-in test data
            compile(artifact.script, str(artifact.artifact_id), "exec"),
            namespace,
        )
        namespace.update(
            search_web=search_web,
            fetch_page=fetch_page,
            llm_chat=llm_chat,
        )

        query_entrypoint = namespace["query"]
        assert callable(query_entrypoint)
        response = await query_entrypoint(
            Query(text="Which Record Was Published in 2026?")
        )

        assert isinstance(response, Response)
        assert response.text


def test_production_false_novel_case_protects_reachability_over_dead_architecture() -> None:
    group = next(
        group
        for group in _groups()
        if group.case_id == "notable-unreachable-parallel-controller"
    )
    production_reference = next(
        reference
        for reference in eligible_comparisons(group)
        if reference.expected_pairwise == "near_duplicate"
    )

    assert group.expected_final == "near_duplicate"
    assert group.production_evidence is not None
    assert str(group.production_evidence.source_batch_id) == (
        "3258ff1c-8e73-4f7d-b7f4-dc32c943f4d4"
    )
    assert group.production_evidence.observed_production_classifications == (
        "novel",
        "novel",
        "novel",
    )
    assert production_reference.hotkey != group.candidate.hotkey
    assert production_reference.gold_explanation is not None
    assert "_parallel_research is never called" in (
        production_reference.gold_explanation.primary_controller
    )
    call_graph = _top_level_function_call_graph(group.candidate.script)
    assert {"_fixed_research", "_solve", "_parallel_research"} <= set(call_graph)
    assert call_graph["_solve"] == {"_fixed_research", "_synthesize"}
    assert "_parallel_research" not in _reachable_functions(call_graph, "query")
    assert not production_reference.candidate_diff.endswith("\n")
    assert production_reference.candidate_diff.startswith(
        f"--- reference/{production_reference.artifact_id}\n"
        f"+++ candidate/{group.candidate.artifact_id}\n"
    )


def test_production_distillations_cover_true_redesigns_and_same_root_changes() -> None:
    expected_provenance = {
        "near-production-lane-b-payload-guard": (
            "a7045f3e-bcc0-4e0f-b109-cd1911b12d9f",
            "near_duplicate",
        ),
        "notable-unreachable-parallel-controller": (
            "efff9f12-e877-4934-8cae-9497dc5b9c58",
            "near_duplicate",
        ),
        "notable-production-conflict-ledger": (
            "2fdce5bf-0abd-46b4-acf2-51b071722316",
            "notable_change",
        ),
        "near-production-constraint-ledger": (
            "32e5807a-0649-413d-aa56-b6fde053dc21",
            "near_duplicate",
        ),
        "notable-production-draft-audit-repair": (
            "108e4176-5c75-4ce6-8f8d-1abfaaf1fcff",
            "notable_change",
        ),
        "novel-json-contract-solver": (
            "12ac5428-27a1-4527-8224-485c18197fe9",
            "novel",
        ),
        "novel-vfs-investigation-state": (
            "a50c4f26-1161-4e94-9172-3692b6c1d0c0",
            "novel",
        ),
        "novel-entity-attribute-graph": (
            "d4378bee-9ba1-4600-9093-875ee5fdff12",
            "novel",
        ),
    }
    groups = {group.case_id: group for group in _groups()}

    for case_id, (artifact_id, expected_final) in expected_provenance.items():
        group = groups[case_id]
        assert group.expected_final == expected_final
        assert group.production_evidence is not None
        assert str(group.production_evidence.source_batch_id) == (
            "3258ff1c-8e73-4f7d-b7f4-dc32c943f4d4"
        )
        assert str(group.production_evidence.candidate_artifact_id) == artifact_id
        assert group.production_evidence.distilled_reproduction is True
        if (
            case_id.startswith("notable-production-")
            or case_id
            in {
                "notable-unreachable-parallel-controller",
                "near-production-constraint-ledger",
            }
        ):
            assert set(group.production_evidence.observed_production_classifications) == {
                "novel"
            }


def test_novel_gold_cases_expose_reachable_implementations_not_architecture_names() -> None:
    required_reachable_functions = {
        "novel-json-contract-solver": {"validate_contract", "execute_contract"},
        "novel-vfs-investigation-state": {"line_id"},
        "novel-entity-attribute-graph": {"holes", "merge_records"},
    }

    for group in _groups():
        if group.expected_final != "novel":
            continue
        module = ast.parse(group.candidate.script)
        definitions = {
            node.name: node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        entrypoint = definitions["query"]
        call_graph = _top_level_function_call_graph(group.candidate.script)

        assert len(entrypoint.body) > 1
        assert required_reachable_functions[group.case_id] <= _reachable_functions(
            call_graph,
            "query",
        )
        reachable_nodes = [
            definitions[name]
            for name in {"query", *_reachable_functions(call_graph, "query")}
        ]
        assert any(
            isinstance(node, (ast.If, ast.For, ast.While))
            for function in reachable_nodes
            for node in ast.walk(function)
        )
        top_level_names = set(definitions)
        top_level_names.update(
            alias.asname or alias.name.split(".", maxsplit=1)[0]
            for node in module.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        )
        unresolved_bare_calls = {
            call.func.id
            for call in ast.walk(module)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id not in top_level_names
            and not hasattr(builtins, call.func.id)
        }
        assert unresolved_bare_calls == set()
        assert group.production_evidence is not None
        assert all(
            len(ast.parse(reference.script).body) >= 1
            and len(reference.script.splitlines()) >= 5
            for reference in eligible_comparisons(group)
        )


def _top_level_function_call_graph(script: str) -> dict[str, set[str]]:
    module = ast.parse(script)
    definitions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return {
        name: {
            call.func.id
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in definitions
        }
        for name, function in definitions.items()
    }


def _has_entrypoint_decorator(function: ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "entrypoint"
        and len(decorator.args) == 1
        and isinstance(decorator.args[0], ast.Constant)
        and decorator.args[0].value == "query"
        for decorator in function.decorator_list
    )


def _annotation_name(annotation: ast.expr | None) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    return None


def _static_string(
    expression: ast.expr | None,
    constants: Mapping[str, str],
) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.Name):
        return constants.get(expression.id)
    return None


def _reachable_functions(call_graph: Mapping[str, set[str]], entrypoint: str) -> set[str]:
    reachable: set[str] = set()
    pending = [entrypoint]
    while pending:
        current = pending.pop()
        for called in call_graph.get(current, set()):
            if called in reachable:
                continue
            reachable.add(called)
            pending.append(called)
    return reachable


@pytest.mark.anyio
async def test_same_hotkey_references_never_reach_the_judge_or_aggregation(
    tmp_path: Path,
) -> None:
    summary, delegate, groups = await _run(tmp_path)
    expected_eligible_count = sum(len(eligible_comparisons(group)) for group in groups)
    same_hotkey_ids = {
        str(reference.artifact_id)
        for group in groups
        for reference in group.references
        if reference.hotkey == group.candidate.hotkey
    }

    assert len(delegate.requests) == expected_eligible_count
    assert summary.final.accuracy == 1.0
    assert not any(
        row["similarity_request"]["reference_artifact_id"] in same_hotkey_ids
        for row in _pair_rows(summary)
    )


def test_least_changed_cross_hotkey_pair_prevents_final_novel() -> None:
    assert (
        aggregate_classification(["architectural_replacement", "notable_change"])
        == "notable_change"
    )
    assert (
        aggregate_classification(
            ["architectural_replacement", "architectural_replacement"]
        )
        == "novel"
    )
    assert normalize_pairwise("novel") == "architectural_replacement"


def test_provider_failures_remain_in_metric_denominators() -> None:
    summary = summarize_metrics(
        expected_labels=["duplicate", "near_duplicate"],
        observed_labels=["provider_failure", "near_duplicate"],
        class_labels=PAIRWISE_LABELS,
    )

    assert summary.total == 2
    assert summary.accuracy == 0.5
    assert summary.confusion_matrix["duplicate"]["provider_failure"] == 1
    assert summary.per_class["duplicate"].support == 1
    assert summary.per_class["duplicate"].recall == 0.0


def test_overclassification_gate_is_asymmetric() -> None:
    rank = {label: index for index, label in enumerate(FINAL_LABELS)}

    assert (
        _overclassification_count(
            expected_labels=["duplicate", "near_duplicate", "notable_change", "novel"],
            observed_labels=["near_duplicate", "notable_change", "novel", "novel"],
            rank=rank,
        )
        == 3
    )
    assert (
        _overclassification_count(
            expected_labels=["near_duplicate", "notable_change", "novel"],
            observed_labels=["duplicate", "near_duplicate", "notable_change"],
            rank=rank,
        )
        == 0
    )


def test_multi_step_underclassification_count_enforces_approved_tolerance() -> None:
    rank = {label: index for index, label in enumerate(PAIRWISE_LABELS)}

    assert (
        _multi_step_underclassification_count(
            expected_labels=[
                "duplicate",
                "near_duplicate",
                "notable_change",
                "architectural_replacement",
                "architectural_replacement",
                "architectural_replacement",
            ],
            observed_labels=[
                "duplicate",
                "duplicate",
                "near_duplicate",
                "notable_change",
                "near_duplicate",
                "duplicate",
            ],
            rank=rank,
        )
        == 2
    )
    assert (
        _multi_step_underclassification_count(
            expected_labels=["architectural_replacement"],
            observed_labels=["provider_failure"],
            rank=rank,
        )
        == 0
    )


@pytest.mark.anyio
async def test_run_benchmark_records_multi_step_underclassifications(
    tmp_path: Path,
) -> None:
    groups = _groups()
    novel_group = next(group for group in groups if group.expected_final == "novel")
    reference_id = eligible_comparisons(novel_group)[0].artifact_id

    summary, _, _ = await _run(
        tmp_path,
        classifications={reference_id: "near_duplicate"},
    )

    assert summary.pairwise_multi_step_underclassification_count == 1
    assert summary.final_multi_step_underclassification_count == 1
    persisted = json.loads(
        (Path(summary.run_directory) / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted["pairwise_multi_step_underclassification_count"] == 1
    assert persisted["final_multi_step_underclassification_count"] == 1


@pytest.mark.anyio
async def test_retry_exhaustion_retains_its_attached_response_before_failure_gate(
    tmp_path: Path,
) -> None:
    groups = _groups()
    reference_id = eligible_comparisons(groups[0])[0].artifact_id
    response = _response(response_id="retry-response")
    error = LlmRetryExhaustedError("retry exhausted", response=response, attempts=1)

    summary, _, _ = await _run(tmp_path, outcomes={reference_id: error})
    failed_row = next(row for row in _pair_rows(summary) if row["observed_pairwise"] == "provider_failure")

    assert summary.provider_failure_count == 1
    assert summary.pairwise_multi_step_underclassification_count == 0
    assert summary.final_multi_step_underclassification_count == 0
    assert failed_row["error"]["type"] == "LlmRetryExhaustedError"
    assert failed_row["llm_response"]["id"] == "retry-response"
    assert failed_row["llm_response"]["raw_response"]["id"] == "retry-response"


@pytest.mark.anyio
async def test_provider_error_retains_its_attached_response_before_failure_gate(
    tmp_path: Path,
) -> None:
    groups = _groups()
    reference_id = eligible_comparisons(groups[0])[0].artifact_id
    response = _response(response_id="provider-response")
    error = LlmProviderError("provider failed", response=response)

    summary, _, _ = await _run(tmp_path, outcomes={reference_id: error})
    failed_row = next(row for row in _pair_rows(summary) if row["observed_pairwise"] == "provider_failure")

    assert summary.provider_failure_count == 1
    assert failed_row["error"]["type"] == "LlmProviderError"
    assert failed_row["llm_response"]["id"] == "provider-response"


@pytest.mark.anyio
async def test_structured_output_failure_retains_current_invocation_response(
    tmp_path: Path,
) -> None:
    groups = _groups()
    reference_id = eligible_comparisons(groups[0])[0].artifact_id
    response = _response(response_id="malformed-response")
    outcome = _PostprocessFailure(
        response=response,
        error=RuntimeError("structured output validation failed"),
    )

    summary, _, _ = await _run(tmp_path, outcomes={reference_id: outcome})
    failed_row = next(row for row in _pair_rows(summary) if row["observed_pairwise"] == "provider_failure")

    assert failed_row["similarity_request"]["reference_artifact_id"] == str(reference_id)
    assert failed_row["error"] == {
        "message": "structured output validation failed",
        "type": "RuntimeError",
    }
    assert failed_row["llm_response"]["id"] == "malformed-response"
    assert failed_row["llm_response"]["raw_response"]["model"] == "served-gemma-revision"
    run_directory = Path(summary.run_directory)
    assert (run_directory / "manifest.json").is_file()
    assert (run_directory / "candidate_results.jsonl").is_file()
    assert (run_directory / "summary.json").is_file()


@pytest.mark.anyio
async def test_failure_without_response_does_not_reuse_previous_pair_response(
    tmp_path: Path,
) -> None:
    groups = _groups()
    first, second = eligible_comparisons(groups[0])
    outcomes = {
        first.artifact_id: _response(response_id="earlier-success"),
        second.artifact_id: LlmProviderError("failed before response"),
    }

    summary, _, _ = await _run(tmp_path, outcomes=outcomes)
    failed_row = next(row for row in _pair_rows(summary) if row["observed_pairwise"] == "provider_failure")

    assert failed_row["similarity_request"]["reference_artifact_id"] == str(second.artifact_id)
    assert failed_row["llm_response"] is None
