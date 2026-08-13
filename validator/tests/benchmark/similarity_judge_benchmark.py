"""Fixed-dataset harness for the opt-in live similarity-judge benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from difflib import unified_diff
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from harnyx_commons.llm.provider import (
    LlmProviderError,
    LlmProviderPort,
    LlmRetryExhaustedError,
)
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse
from harnyx_commons.miner_task_similarity import (
    SimilarityClassification,
    SimilarityJudgeRequest,
    SimilarityJudgeResult,
)

PairwiseGold = Literal[
    "duplicate",
    "near_duplicate",
    "notable_change",
    "architectural_replacement",
]
FinalGold = Literal["duplicate", "near_duplicate", "notable_change", "novel"]
ObservedLabel = PairwiseGold | FinalGold | Literal["provider_failure"]

PAIRWISE_LABELS: tuple[PairwiseGold, ...] = (
    "duplicate",
    "near_duplicate",
    "notable_change",
    "architectural_replacement",
)
FINAL_LABELS: tuple[FinalGold, ...] = (
    "duplicate",
    "near_duplicate",
    "notable_change",
    "novel",
)
_PAIRWISE_RANK = {label: rank for rank, label in enumerate(PAIRWISE_LABELS)}
_FINAL_RANK = {label: rank for rank, label in enumerate(FINAL_LABELS)}
_REQUIRED_BOUNDARY_TAGS = frozenset(
    {
        "prompt-model-parameter-only",
        "renamed-equivalent-code",
        "dead-or-unused-architecture",
        "extra-tools-or-steps",
        "local-pipeline-improvement",
        "retries-or-fallback-only",
        "substantial-same-root-reorganization",
        "complete-architecture-replacement",
    }
)


class GoldExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    primary_controller: str = Field(min_length=1)
    evidence_state_and_flow: str = Field(min_length=1)
    answer_production_path: str = Field(min_length=1)
    ordinary_case_reachability: str = Field(min_length=1)
    adjacent_class_rejection: str = Field(min_length=1)


class BenchmarkArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: UUID
    miner_uid: int = Field(ge=0)
    hotkey: str = Field(min_length=1)
    script: str = Field(min_length=1)


class BenchmarkReference(BenchmarkArtifact):
    candidate_diff: str = Field(min_length=1)
    expected_pairwise: PairwiseGold | None
    gold_explanation: GoldExplanation | None


class ProductionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_batch_id: UUID
    candidate_artifact_id: UUID
    selected_reference_artifact_id: UUID
    observed_production_classifications: tuple[SimilarityClassification, ...] = Field(
        min_length=1
    )
    distilled_reproduction: bool
    evidence_summary: str = Field(min_length=1)


class BenchmarkCandidateGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    batch_id: UUID
    candidate: BenchmarkArtifact
    references: tuple[BenchmarkReference, ...] = Field(min_length=2)
    expected_final: FinalGold
    boundary_tags: tuple[str, ...] = Field(min_length=1)
    production_evidence: ProductionEvidence | None = None

    @model_validator(mode="after")
    def _validate_comparison_contract(self) -> BenchmarkCandidateGroup:
        reference_ids = [reference.artifact_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError(f"{self.case_id}: reference artifact IDs must be unique")
        if self.candidate.artifact_id in reference_ids:
            raise ValueError(f"{self.case_id}: candidate cannot also be a reference")

        eligible_gold: list[PairwiseGold] = []
        for reference in self.references:
            expected_diff = build_candidate_diff(
                reference.script,
                self.candidate.script,
                reference_artifact_id=reference.artifact_id,
                candidate_artifact_id=self.candidate.artifact_id,
            )
            if reference.candidate_diff != expected_diff:
                raise ValueError(
                    f"{self.case_id}: candidate_diff for {reference.artifact_id} "
                    "does not match the stored source artifacts"
                )
            if reference.hotkey == self.candidate.hotkey:
                if reference.expected_pairwise is not None or reference.gold_explanation is not None:
                    raise ValueError(
                        f"{self.case_id}: same-hotkey decoys must not carry gold labels"
                    )
                continue
            if reference.expected_pairwise is None or reference.gold_explanation is None:
                raise ValueError(
                    f"{self.case_id}: different-hotkey references require gold labels and explanations"
                )
            eligible_gold.append(reference.expected_pairwise)

        if len(eligible_gold) < 2:
            raise ValueError(f"{self.case_id}: at least two different-hotkey references are required")
        if aggregate_classification(eligible_gold) != self.expected_final:
            raise ValueError(f"{self.case_id}: expected_final disagrees with pairwise gold labels")
        return self


class BenchmarkIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_sha: str = Field(min_length=1)
    validator_package_version: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    route_target: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    normalized_base_url: str = Field(min_length=1)
    immutable_serving_revision: str | None = None
    benchmark_source_sha256: dict[str, str]
    temperature: float
    retry_attempts: int = Field(ge=1)
    fallback_models: tuple[str, ...]


class ClassMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: float
    recall: float
    f1: float
    support: int


class MetricsSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accuracy: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    total: int


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    run_directory: str
    pairwise: MetricsSummary
    final: MetricsSummary
    provider_failure_count: int
    pairwise_overclassification_count: int
    pairwise_overclassification_rate: float
    final_overclassification_count: int
    final_overclassification_rate: float
    pairwise_multi_step_underclassification_count: int
    final_multi_step_underclassification_count: int
    false_novel_count: int
    false_novel_rate: float
    missed_novel_count: int
    missed_novel_rate: float


class SimilarityJudgePort(Protocol):
    async def judge(self, request: SimilarityJudgeRequest) -> SimilarityJudgeResult: ...


@dataclass
class InvocationRecord:
    request: AbstractLlmRequest | None = None
    returned_response: LlmResponse | None = None


class InvocationRecordingProvider(LlmProviderPort):
    """Retain provider evidence for exactly one active judge invocation."""

    def __init__(self, delegate: LlmProviderPort) -> None:
        self._delegate = delegate
        self._active: InvocationRecord | None = None

    def begin_invocation(self) -> InvocationRecord:
        if self._active is not None:
            raise RuntimeError("previous benchmark provider invocation is still active")
        self._active = InvocationRecord()
        return self._active

    def finish_invocation(self, invocation: InvocationRecord) -> None:
        if self._active is not invocation:
            raise RuntimeError("benchmark provider invocation scope does not match")
        self._active = None

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        if self._active is None:
            raise RuntimeError("provider call occurred outside a benchmark invocation scope")
        if self._active.request is not None:
            raise RuntimeError("benchmark invocation made more than one provider call")
        self._active.request = request
        response = await self._delegate.invoke(request)
        self._active.returned_response = response
        return response

    async def aclose(self) -> None:
        await self._delegate.aclose()


def build_candidate_diff(
    reference_script: str,
    candidate_script: str,
    *,
    reference_artifact_id: UUID,
    candidate_artifact_id: UUID,
) -> str:
    lines = unified_diff(
        reference_script.splitlines(),
        candidate_script.splitlines(),
        fromfile=f"reference/{reference_artifact_id}",
        tofile=f"candidate/{candidate_artifact_id}",
        lineterm="",
    )
    rendered = "\n".join(lines).strip()
    return rendered or "(no textual diff)"


def eligible_comparisons(group: BenchmarkCandidateGroup) -> tuple[BenchmarkReference, ...]:
    comparisons = tuple(
        reference
        for reference in group.references
        if reference.hotkey != group.candidate.hotkey
    )
    if not comparisons:
        raise ValueError(f"{group.case_id}: no different-hotkey references")
    return comparisons


def normalize_pairwise(classification: SimilarityClassification) -> PairwiseGold:
    if classification == "novel":
        return "architectural_replacement"
    return classification


def aggregate_classification(labels: Sequence[PairwiseGold]) -> FinalGold:
    if not labels:
        raise ValueError("cannot aggregate an empty pairwise label collection")
    least_changed = min(labels, key=_PAIRWISE_RANK.__getitem__)
    if least_changed == "architectural_replacement":
        return "novel"
    return least_changed


def load_cases(path: Path) -> tuple[BenchmarkCandidateGroup, ...]:
    groups: list[BenchmarkCandidateGroup] = []
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        normalized = line.strip()
        if not normalized:
            continue
        group = BenchmarkCandidateGroup.model_validate_json(normalized)
        if group.case_id in seen_case_ids:
            raise ValueError(f"duplicate benchmark case_id on line {line_number}: {group.case_id}")
        seen_case_ids.add(group.case_id)
        groups.append(group)
    validate_case_collection(groups)
    return tuple(groups)


def validate_case_collection(groups: Sequence[BenchmarkCandidateGroup]) -> None:
    if len(groups) < 12:
        raise ValueError("benchmark dataset requires at least 12 candidate groups")

    final_counts = Counter(group.expected_final for group in groups)
    missing_final_support = {
        label: 3 - final_counts[label] for label in FINAL_LABELS if final_counts[label] < 3
    }
    if missing_final_support:
        raise ValueError(f"benchmark final-class support is incomplete: {missing_final_support}")

    pairwise_counts: Counter[PairwiseGold] = Counter()
    same_hotkey_group_count = 0
    boundary_tags: set[str] = set()
    for group in groups:
        boundary_tags.update(group.boundary_tags)
        if any(reference.hotkey == group.candidate.hotkey for reference in group.references):
            same_hotkey_group_count += 1
        pairwise_counts.update(
            cast(PairwiseGold, reference.expected_pairwise)
            for reference in eligible_comparisons(group)
        )
    missing_pairwise_support = {
        label: 4 - pairwise_counts[label] for label in PAIRWISE_LABELS if pairwise_counts[label] < 4
    }
    if missing_pairwise_support:
        raise ValueError(
            f"benchmark pairwise-class support is incomplete: {missing_pairwise_support}"
        )
    if same_hotkey_group_count < 4:
        raise ValueError("benchmark dataset requires at least four same-hotkey decoy groups")
    missing_boundaries = sorted(_REQUIRED_BOUNDARY_TAGS - boundary_tags)
    if missing_boundaries:
        raise ValueError(f"benchmark dataset is missing boundary coverage: {missing_boundaries}")


def summarize_metrics(
    *,
    expected_labels: Sequence[str],
    observed_labels: Sequence[str],
    class_labels: Sequence[str],
) -> MetricsSummary:
    if len(expected_labels) != len(observed_labels):
        raise ValueError("expected and observed label counts differ")
    observed_columns = (*class_labels, "provider_failure")
    confusion = {
        expected: {observed: 0 for observed in observed_columns} for expected in class_labels
    }
    for expected, observed in zip(expected_labels, observed_labels, strict=True):
        if expected not in confusion:
            raise ValueError(f"unsupported expected label: {expected}")
        if observed not in confusion[expected]:
            raise ValueError(f"unsupported observed label: {observed}")
        confusion[expected][observed] += 1

    total = len(expected_labels)
    correct = sum(confusion[label][label] for label in class_labels)
    per_class: dict[str, ClassMetrics] = {}
    for label in class_labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[expected][label] for expected in class_labels if expected != label
        )
        false_negative = sum(
            count for observed, count in confusion[label].items() if observed != label
        )
        support = sum(confusion[label].values())
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=_safe_ratio(2 * precision * recall, precision + recall),
            support=support,
        )
    return MetricsSummary(
        accuracy=_safe_ratio(correct, total),
        per_class=per_class,
        confusion_matrix=confusion,
        total=total,
    )


async def run_benchmark(
    *,
    groups: Sequence[BenchmarkCandidateGroup],
    judge: SimilarityJudgePort,
    recording_provider: InvocationRecordingProvider,
    output_root: Path,
    identity: BenchmarkIdentity,
) -> BenchmarkSummary:
    validate_case_collection(groups)
    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    pair_path = run_directory / "pair_results.jsonl"
    candidate_path = run_directory / "candidate_results.jsonl"
    manifest_path = run_directory / "manifest.json"
    summary_path = run_directory / "summary.json"
    manifest: dict[str, object] = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "prompt_sha256": None,
        **identity.model_dump(mode="json"),
    }
    _write_json(manifest_path, manifest)

    pair_expected: list[str] = []
    pair_observed: list[str] = []
    final_expected: list[str] = []
    final_observed: list[str] = []
    provider_failure_count = 0

    for group in groups:
        group_labels: list[PairwiseGold] = []
        group_failed = False
        for reference in eligible_comparisons(group):
            expected = cast(PairwiseGold, reference.expected_pairwise)
            pair_expected.append(expected)
            request = SimilarityJudgeRequest(
                batch_id=group.batch_id,
                candidate_artifact_id=group.candidate.artifact_id,
                reference_artifact_id=reference.artifact_id,
                candidate_miner_uid=group.candidate.miner_uid,
                reference_miner_uid=reference.miner_uid,
                reference_script=reference.script,
                candidate_diff=reference.candidate_diff,
            )
            invocation = recording_provider.begin_invocation()
            call_started = time.perf_counter()
            try:
                result = await judge.judge(request)
            except Exception as error:
                elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
                response = _failure_response(error, invocation)
                row = _pair_failure_row(
                    group=group,
                    reference=reference,
                    expected=expected,
                    request=request,
                    llm_request=invocation.request,
                    response=response,
                    error=error,
                    elapsed_ms=elapsed_ms,
                )
                _append_json_line(pair_path, row)
                pair_observed.append("provider_failure")
                provider_failure_count += 1
                group_failed = True
            else:
                elapsed_ms = round((time.perf_counter() - call_started) * 1000, 2)
                if invocation.request is None or invocation.returned_response is None:
                    raise RuntimeError("similarity judge did not execute exactly one provider call")
                observed = normalize_pairwise(result.classification)
                pair_observed.append(observed)
                group_labels.append(observed)
                _append_json_line(
                    pair_path,
                    _pair_success_row(
                        group=group,
                        reference=reference,
                        expected=expected,
                        request=request,
                        llm_request=invocation.request,
                        response=invocation.returned_response,
                        result=result,
                        observed=observed,
                        elapsed_ms=elapsed_ms,
                    ),
                )
                if manifest["prompt_sha256"] is None:
                    manifest["prompt_sha256"] = _prompt_sha256(invocation.request)
            finally:
                recording_provider.finish_invocation(invocation)

        expected_final = group.expected_final
        final_expected.append(expected_final)
        if group_failed:
            observed_final: str = "provider_failure"
            candidate_row = {
                "case_id": group.case_id,
                "expected_final": expected_final,
                "observed_final": observed_final,
                "eligible_reference_artifact_ids": [
                    str(reference.artifact_id) for reference in eligible_comparisons(group)
                ],
                "error": "one or more eligible pairwise calls failed",
            }
        else:
            observed_final = aggregate_classification(group_labels)
            candidate_row = {
                "case_id": group.case_id,
                "expected_final": expected_final,
                "observed_final": observed_final,
                "eligible_reference_artifact_ids": [
                    str(reference.artifact_id) for reference in eligible_comparisons(group)
                ],
                "pairwise_observed": group_labels,
                "error": None,
            }
        final_observed.append(observed_final)
        _append_json_line(candidate_path, candidate_row)

    pairwise_metrics = summarize_metrics(
        expected_labels=pair_expected,
        observed_labels=pair_observed,
        class_labels=PAIRWISE_LABELS,
    )
    final_metrics = summarize_metrics(
        expected_labels=final_expected,
        observed_labels=final_observed,
        class_labels=FINAL_LABELS,
    )
    pairwise_overclassification_count = _overclassification_count(
        expected_labels=pair_expected,
        observed_labels=pair_observed,
        rank=_PAIRWISE_RANK,
    )
    final_overclassification_count = _overclassification_count(
        expected_labels=final_expected,
        observed_labels=final_observed,
        rank=_FINAL_RANK,
    )
    pairwise_multi_step_underclassification_count = _multi_step_underclassification_count(
        expected_labels=pair_expected,
        observed_labels=pair_observed,
        rank=_PAIRWISE_RANK,
    )
    final_multi_step_underclassification_count = _multi_step_underclassification_count(
        expected_labels=final_expected,
        observed_labels=final_observed,
        rank=_FINAL_RANK,
    )
    false_novel_count = sum(
        expected != "novel" and observed == "novel"
        for expected, observed in zip(final_expected, final_observed, strict=True)
    )
    missed_novel_count = sum(
        expected == "novel" and observed != "novel"
        for expected, observed in zip(final_expected, final_observed, strict=True)
    )
    non_novel_count = sum(expected != "novel" for expected in final_expected)
    novel_count = sum(expected == "novel" for expected in final_expected)
    summary = BenchmarkSummary(
        run_id=run_id,
        run_directory=str(run_directory),
        pairwise=pairwise_metrics,
        final=final_metrics,
        provider_failure_count=provider_failure_count,
        pairwise_overclassification_count=pairwise_overclassification_count,
        pairwise_overclassification_rate=_safe_ratio(
            pairwise_overclassification_count,
            len(pair_expected),
        ),
        final_overclassification_count=final_overclassification_count,
        final_overclassification_rate=_safe_ratio(
            final_overclassification_count,
            len(final_expected),
        ),
        pairwise_multi_step_underclassification_count=(
            pairwise_multi_step_underclassification_count
        ),
        final_multi_step_underclassification_count=(
            final_multi_step_underclassification_count
        ),
        false_novel_count=false_novel_count,
        false_novel_rate=_safe_ratio(false_novel_count, non_novel_count),
        missed_novel_count=missed_novel_count,
        missed_novel_rate=_safe_ratio(missed_novel_count, novel_count),
    )
    _write_json(summary_path, summary.model_dump(mode="json"))
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_json(manifest_path, manifest)
    return summary


def _overclassification_count(
    *,
    expected_labels: Sequence[str],
    observed_labels: Sequence[str],
    rank: Mapping[str, int],
) -> int:
    return sum(
        observed != "provider_failure" and rank[observed] > rank[expected]
        for expected, observed in zip(expected_labels, observed_labels, strict=True)
    )


def _multi_step_underclassification_count(
    *,
    expected_labels: Sequence[str],
    observed_labels: Sequence[str],
    rank: Mapping[str, int],
) -> int:
    return sum(
        observed != "provider_failure" and rank[expected] - rank[observed] > 1
        for expected, observed in zip(expected_labels, observed_labels, strict=True)
    )


def _failure_response(
    error: Exception,
    invocation: InvocationRecord,
) -> LlmResponse | None:
    if isinstance(error, (LlmRetryExhaustedError, LlmProviderError)):
        return error.response
    return invocation.returned_response


def _pair_success_row(
    *,
    group: BenchmarkCandidateGroup,
    reference: BenchmarkReference,
    expected: PairwiseGold,
    request: SimilarityJudgeRequest,
    llm_request: AbstractLlmRequest,
    response: LlmResponse,
    result: SimilarityJudgeResult,
    observed: PairwiseGold,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        **_pair_base_row(
            group=group,
            reference=reference,
            expected=expected,
            request=request,
            llm_request=llm_request,
            response=response,
            elapsed_ms=elapsed_ms,
        ),
        "observed_pairwise": observed,
        "raw_production_classification": result.classification,
        "reasoning": result.reasoning,
        "result_model": result.model,
        "result_provider": result.provider,
        "error": None,
    }


def _pair_failure_row(
    *,
    group: BenchmarkCandidateGroup,
    reference: BenchmarkReference,
    expected: PairwiseGold,
    request: SimilarityJudgeRequest,
    llm_request: AbstractLlmRequest | None,
    response: LlmResponse | None,
    error: Exception,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        **_pair_base_row(
            group=group,
            reference=reference,
            expected=expected,
            request=request,
            llm_request=llm_request,
            response=response,
            elapsed_ms=elapsed_ms,
        ),
        "observed_pairwise": "provider_failure",
        "raw_production_classification": None,
        "reasoning": None,
        "result_model": None,
        "result_provider": None,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
    }


def _pair_base_row(
    *,
    group: BenchmarkCandidateGroup,
    reference: BenchmarkReference,
    expected: PairwiseGold,
    request: SimilarityJudgeRequest,
    llm_request: AbstractLlmRequest | None,
    response: LlmResponse | None,
    elapsed_ms: float,
) -> dict[str, object]:
    llm_request_payload = _llm_request_payload(llm_request)
    return {
        "case_id": group.case_id,
        "candidate_hotkey": group.candidate.hotkey,
        "reference_hotkey": reference.hotkey,
        "expected_pairwise": expected,
        "elapsed_ms": elapsed_ms,
        "similarity_request": _json_safe(request),
        "llm_request": llm_request_payload,
        "llm_request_sha256": _sha256_json(llm_request_payload),
        "llm_response": _llm_response_payload(response),
    }


def _llm_request_payload(request: AbstractLlmRequest | None) -> dict[str, object] | None:
    if request is None:
        return None
    return {
        "provider": request.provider,
        "model": request.model,
        "messages": _json_safe(request.messages),
        "output_mode": request.output_mode,
        "output_schema": (
            request.output_schema.__name__ if request.output_schema is not None else None
        ),
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "reasoning_effort": request.reasoning_effort,
        "timeout_seconds": request.timeout_seconds,
        "retry_policy": _json_safe(request.retry_policy),
        "use_case": request.use_case,
    }


def _llm_response_payload(response: LlmResponse | None) -> dict[str, object] | None:
    if response is None:
        return None
    metadata = dict(response.metadata or {})
    return {
        "id": response.id,
        "choices": _json_safe(response.choices),
        "usage": _json_safe(response.usage),
        "metadata": _json_safe(metadata),
        "raw_response": _json_safe(metadata.get("raw_response")),
        "postprocessed": _json_safe(response.postprocessed),
        "finish_reason": response.finish_reason,
        "raw_text": response.raw_text,
        "selected_provider": metadata.get("selected_provider"),
        "selected_model": metadata.get("selected_model"),
    }


def _prompt_sha256(request: AbstractLlmRequest) -> str:
    first_message = request.messages[0]
    return hashlib.sha256(
        json.dumps(_json_safe(first_message), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _sha256_json(payload: object) -> str | None:
    if payload is None:
        return None
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        handle.write("\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
