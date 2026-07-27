from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from harnyx_commons.domain_tweak_generation.adk_runner import (
    DomainTweakAdkRunner,
    DomainTweakAdkTurn,
    DomainTweakAdkTurnExecutor,
)
from harnyx_commons.domain_tweak_generation.batch_pipeline import (
    DomainTweakBatchGenerationPipeline,
    _question_attempt_windows,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkRunConfig,
    DomainTweakBatchGenerationConfig,
    DomainTweakQuestionPhasePolicy,
)
from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakPairInput,
)
from harnyx_commons.tools.extraction_models import ExtractedPage, ExtractPagesRequest, ExtractPagesResponse
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _ExtractionProvider:
    requests: list[ExtractPagesRequest] = field(default_factory=list)

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        return SearchProviderResult(
            response=ExtractPagesResponse(
                pages=tuple(
                    ExtractedPage(
                        url=url,
                        title="Official results",
                        content=f"Official evidence for {url}: the candidate satisfies both predicates.",
                    )
                    for url in request.urls
                )
            ),
            billing=ProviderBillingMetadata(
                actual_cost_provider="parallel",
                source="request_body",
                billable_units=len(request.urls),
                actual_cost_usd=0.001 * len(request.urls),
                service="extract",
            ),
        )


@dataclass
class _Executor:
    abandon_pairs: set[str] = field(default_factory=set)
    rejected_form_pairs: set[str] = field(default_factory=set)
    duplicate_question_pairs: dict[str, str] = field(default_factory=dict)
    not_materializable_pairs: set[str] = field(default_factory=set)
    reference_delay_seconds: float = 0.0
    reference_calls: Counter[str] = field(default_factory=Counter)
    calls: list[tuple[str, str]] = field(default_factory=list)
    active_reference_calls: int = 0
    max_active_reference_calls: int = 0
    repair_structured_semantic_coverage: bool = False
    semantic_pair_id: str | None = None
    semantic_window_id: str | None = None
    semantic_structured: bool = False

    async def __call__(self, **kwargs: object) -> DomainTweakAdkTurn:
        phase = str(kwargs["phase"])
        prompt = str(kwargs["prompt"])
        attempt_index = int(kwargs["attempt_index"])
        if (
            phase == "semantic_support_gate"
            and attempt_index > 0
            and self.repair_structured_semantic_coverage
        ):
            if self.semantic_pair_id is None:
                raise AssertionError("semantic retry did not retain its initial pair")
            pair_id = self.semantic_pair_id
        else:
            pair_id = _pair_id_from_prompt(prompt)
        self.calls.append((phase, pair_id))
        if phase == "form_blueprint":
            payload = _blueprint_payload()
        elif phase == "question_generation":
            payload = _question_payload(
                pair_id,
                question_pair_id=self.duplicate_question_pairs.get(pair_id, pair_id),
            )
        elif phase == "form_review":
            payload = _form_review_payload(form_match=pair_id not in self.rejected_form_pairs)
        elif phase == "reference_answer_generation":
            self.reference_calls[pair_id] += 1
            self.active_reference_calls += 1
            self.max_active_reference_calls = max(
                self.max_active_reference_calls,
                self.active_reference_calls,
            )
            try:
                if self.reference_delay_seconds:
                    await asyncio.sleep(self.reference_delay_seconds)
                payload = (
                    _abandon_payload()
                    if pair_id in self.abandon_pairs
                    else _reference_payload(_window_id_from_prompt(prompt))
                )
            finally:
                self.active_reference_calls -= 1
        elif phase == "structured_output_materialization":
            payload = (
                _not_materializable_payload()
                if pair_id in self.not_materializable_pairs
                else _materialization_payload(_window_id_from_prompt(prompt))
            )
        elif phase == "semantic_support_gate":
            if attempt_index == 0:
                self.semantic_pair_id = pair_id
                self.semantic_window_id = _window_id_from_prompt(prompt)
                self.semantic_structured = '"disposition": "materialized"' in prompt
            if self.semantic_window_id is None:
                raise AssertionError("semantic phase did not retain its acquired window")
            payload = _semantic_payload(
                self.semantic_window_id,
                structured=self.semantic_structured,
                omit_structured_window=(
                    self.repair_structured_semantic_coverage and attempt_index == 0
                ),
            )
        else:
            raise AssertionError(f"unexpected phase {phase}")
        return DomainTweakAdkTurn(final_text=json.dumps(payload), events=())


@dataclass
class _TerminalFailureExecutor:
    failure_phase: str
    failure: ToolProviderError = field(
        default_factory=lambda: ToolProviderError(
            "credential rejected",
            provider="google",
            failure_code=ToolProviderFailureCode.AUTHENTICATION_FAILED,
        )
    )
    sibling_started: asyncio.Event = field(default_factory=asyncio.Event)
    sibling_cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    sibling_completed: asyncio.Event = field(default_factory=asyncio.Event)
    release_sibling: asyncio.Event = field(default_factory=asyncio.Event)
    active_failure_phase_calls: int = 0

    async def __call__(self, **kwargs: object) -> DomainTweakAdkTurn:
        phase = str(kwargs["phase"])
        prompt = str(kwargs["prompt"])
        pair_id = _pair_id_from_prompt(prompt)
        if phase == self.failure_phase:
            self.active_failure_phase_calls += 1
            try:
                if pair_id == "PAIR-001":
                    await self.sibling_started.wait()
                    raise self.failure
                if pair_id == "PAIR-002":
                    self.sibling_started.set()
                    try:
                        await self.release_sibling.wait()
                    except asyncio.CancelledError:
                        self.sibling_cancelled.set()
                        raise
            finally:
                self.active_failure_phase_calls -= 1

        if phase == "form_blueprint":
            payload = _blueprint_payload()
        elif phase == "question_generation":
            payload = _question_payload(pair_id, question_pair_id=pair_id)
        elif phase == "form_review":
            payload = _form_review_payload(form_match=True)
        elif phase == "reference_answer_generation":
            payload = _reference_payload(_window_id_from_prompt(prompt))
        elif phase == "structured_output_materialization":
            payload = _materialization_payload(_window_id_from_prompt(prompt))
        elif phase == "semantic_support_gate":
            payload = _semantic_payload(
                _window_id_from_prompt(prompt),
                structured='"disposition": "materialized"' in prompt,
            )
        else:
            raise AssertionError(f"unexpected phase {phase}")

        if pair_id == "PAIR-002" and (
            (self.failure_phase == "form_blueprint" and phase == "form_review")
            or (self.failure_phase == "reference_answer_generation" and phase == "semantic_support_gate")
        ):
            self.sibling_completed.set()
        return DomainTweakAdkTurn(final_text=json.dumps(payload), events=())


async def test_failed_candidate_is_replaced_from_unused_pair_without_same_pair_retry() -> None:
    executor = _Executor(abandon_pairs={"PAIR-001"})
    result = await _batch(executor).generate_batch(
        _pairs(4),
        DomainTweakBatchGenerationConfig(target_count=2),
    )

    assert [item.reviewed_question.pair_input.pair_id for item in result.finalized_tasks] == [
        "PAIR-002",
        "PAIR-003",
    ]
    assert [item.reviewed_question.pair_input.pair_id for item in result.discarded_candidates] == ["PAIR-001"]
    assert executor.reference_calls == Counter({"PAIR-001": 1, "PAIR-002": 1, "PAIR-003": 1})
    assert result.candidate_finalization_attempt_count == 3
    assert not result.underfilled


async def test_failed_structured_slot_is_replaced_by_a_fresh_structured_candidate() -> None:
    executor = _Executor(not_materializable_pairs={"PAIR-002"})
    result = await _batch(executor).generate_batch(
        _pairs(4),
        DomainTweakBatchGenerationConfig(
            target_count=2,
            structured_target_count=1,
        ),
    )

    assert [item.reviewed_question.pair_input.pair_id for item in result.finalized_tasks] == ["PAIR-001", "PAIR-003"]
    assert result.finalized_tasks[0].task.query.output_schema is None
    assert result.finalized_tasks[1].task.query.output_schema is not None
    assert result.discarded_candidates[0].requested_response_mode == "structured"
    assert result.discarded_candidates[0].reason == "structured_output_not_materializable"


async def test_ten_task_generation_finalizes_five_plain_and_five_structured() -> None:
    result = await _batch(_Executor()).generate_batch(
        _pairs(10),
        DomainTweakBatchGenerationConfig(
            target_count=10,
            structured_target_count=5,
        ),
    )

    assert len(result.finalized_tasks) == 10
    assert sum(item.task.query.output_schema is None for item in result.finalized_tasks) == 5
    assert sum(item.task.query.output_schema is not None for item in result.finalized_tasks) == 5


async def test_batch_wiring_retries_structured_semantic_coverage_feedback() -> None:
    executor = _Executor(repair_structured_semantic_coverage=True)
    result = await _batch(executor).generate_batch(
        _pairs(1),
        DomainTweakBatchGenerationConfig(
            target_count=1,
            structured_target_count=1,
        ),
    )

    assert len(result.finalized_tasks) == 1
    semantic_result = result.finalized_tasks[0].semantic_support_result
    assert semantic_result is not None
    assert len(semantic_result.attempts) == 2
    assert semantic_result.attempts[0].validation_feedback == (
        "candidates[] evidence windows must exactly match its binding",
    )


async def test_all_discards_stop_at_existing_four_times_global_pair_cap() -> None:
    executor = _Executor(abandon_pairs={f"PAIR-{index:03d}" for index in range(1, 9)})
    result = await _batch(executor).generate_batch(
        _pairs(8),
        DomainTweakBatchGenerationConfig(target_count=1),
    )

    assert result.underfilled
    assert result.candidate_finalization_attempt_count == 4
    assert len(result.discarded_candidates) == 4
    assert set(executor.reference_calls.values()) == {1}
    assert set(executor.reference_calls) == {"PAIR-001", "PAIR-002", "PAIR-003", "PAIR-004"}


async def test_form_rejection_uses_fresh_pair_and_never_repair_turns_same_pair() -> None:
    executor = _Executor(rejected_form_pairs={"PAIR-001"})
    result = await _batch(executor).generate_batch(
        _pairs(3),
        DomainTweakBatchGenerationConfig(target_count=1),
    )

    assert result.finalized_tasks[0].reviewed_question.pair_input.pair_id == "PAIR-002"
    assert result.rejected_attempts[0].pair_input.pair_id == "PAIR-001"
    assert result.rejected_attempts[0].reason == "form_rejected"
    assert executor.reference_calls == Counter({"PAIR-002": 1})
    assert Counter(phase for phase, pair_id in executor.calls if pair_id == "PAIR-001") == Counter(
        {"form_blueprint": 1, "question_generation": 1, "form_review": 1}
    )


async def test_canonical_duplicate_question_is_rejected_before_finalization() -> None:
    executor = _Executor(duplicate_question_pairs={"PAIR-002": "PAIR-001"})
    result = await _batch(executor).generate_batch(
        _pairs(4),
        DomainTweakBatchGenerationConfig(target_count=2),
    )

    assert [item.reviewed_question.pair_input.pair_id for item in result.finalized_tasks] == [
        "PAIR-001",
        "PAIR-003",
    ]
    assert any(item.reason == "duplicate_question" for item in result.rejected_attempts)
    assert "PAIR-002" not in executor.reference_calls


async def test_question_and_reference_work_remain_bounded_to_eight_concurrent_calls() -> None:
    executor = _Executor(reference_delay_seconds=0.01)
    result = await _batch(executor).generate_batch(
        _pairs(9),
        DomainTweakBatchGenerationConfig(target_count=9),
    )

    assert len(result.finalized_tasks) == 9
    assert executor.max_active_reference_calls == 8


@pytest.mark.parametrize(
    "failure_phase",
    ("form_blueprint", "reference_answer_generation"),
)
async def test_terminal_failure_cancels_and_drains_sibling_provider_work(
    failure_phase: str,
) -> None:
    executor = _TerminalFailureExecutor(failure_phase=failure_phase)
    try:
        with pytest.raises(ToolProviderError) as exc_info:
            await _batch(executor).generate_batch(
                _pairs(2),
                DomainTweakBatchGenerationConfig(target_count=2),
            )

        assert exc_info.value is executor.failure
        await asyncio.wait_for(executor.sibling_cancelled.wait(), timeout=0.1)
        assert executor.active_failure_phase_calls == 0
    finally:
        executor.release_sibling.set()
        if not executor.sibling_cancelled.is_set():
            await asyncio.wait_for(executor.sibling_completed.wait(), timeout=1)


def test_question_attempt_windows_remain_deterministic_and_contiguous() -> None:
    pairs = _pairs(12)
    policy = DomainTweakQuestionPhasePolicy(
        target_attempt_multiplier=3,
        underfill_extra_passes=3,
        hard_attempt_cap_multiplier=4,
    )

    windows = _question_attempt_windows(pairs, policy, target_count=3)

    assert [[item.pair_id for item in window] for window in windows] == [
        [f"PAIR-{index:03d}" for index in range(1, 10)],
        ["PAIR-010"],
        ["PAIR-011"],
        ["PAIR-012"],
    ]
    assert DomainTweakBatchGenerationConfig(target_count=3).candidate_attempt_cap == 12


def _batch(executor: DomainTweakAdkTurnExecutor) -> DomainTweakBatchGenerationPipeline:
    return DomainTweakBatchGenerationPipeline(
        base_config=DomainTweakAdkRunConfig(model="gemini-test", max_retries=0),
        source_provider=_ExtractionProvider(),
        runner=DomainTweakAdkRunner(turn_executor=executor),
    )


def _pairs(count: int) -> tuple[DomainTweakPairInput, ...]:
    return tuple(
        DomainTweakPairInput(
            pair_id=f"PAIR-{index:03d}",
            deepsearchqa_form_target=f"FORM PAIR-{index:03d}: Which candidates satisfy both?",
            deepresearch9k_domain_target=f"DOMAIN PAIR-{index:03d}",
            timestamp=datetime(2026, 7, 21, tzinfo=UTC),
        )
        for index in range(1, count + 1)
    )


def _blueprint_payload() -> dict[str, object]:
    return {
        "status": "proceed",
        "operation": "Filter a closed candidate set by two retrieved predicates.",
        "load_bearing_invariants": ["closed universe", "two predicates"],
        "non_load_bearing_surface_features": [],
        "retrieval_boundary": "Sources supply predicate values.",
        "answer_shape": "Exhaustive list.",
        "semantic_ambiguities": [],
        "no_generate_reason": None,
    }


def _question_payload(pair_id: str, *, question_pair_id: str) -> dict[str, object]:
    return {
        "status": "ready",
        "question": f"Question {question_pair_id}: which candidates satisfy both predicates?",
        "short_answer": f"Answer {question_pair_id}",
        "solution_steps": ["Read the table.", "Intersect the qualifying sets."],
        "claims": [
            {
                "claim_id": "answer",
                "claim": f"Answer {question_pair_id} satisfies both predicates.",
                "role": "answer_determining",
                "support_mode": "external_source",
                "support_explanation": "The official table records both values.",
            }
        ],
        "evidence_declarations": [
            {
                "evidence_id": "evidence-1",
                "source_url": f"https://example.com/{pair_id.lower()}",
                "source_title": "Official results",
                "source_locator": "Results table",
                "claimed_excerpt": "satisfies both predicates",
                "supported_claim_ids": ["answer"],
                "support_explanation": "The table records both predicate values.",
            }
        ],
        "no_generate_reason": None,
    }


def _form_review_payload(*, form_match: bool) -> dict[str, object]:
    return {
        "form_match": form_match,
        "reviewer_feedback": "The form is preserved." if form_match else "The form changed.",
        "question_requirements": [
            {
                "requirement_id": "metric",
                "category": "metric_or_field_relation",
                "requirement": "Both predicates must hold.",
                "required_relation": "derived_calculation",
            }
        ],
        "requirement_category_audit": [
            {
                "category": category,
                "present": category == "metric_or_field_relation",
                "explanation": "Present." if category == "metric_or_field_relation" else "Absent.",
            }
            for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
        ],
    }


def _reference_payload(window_id: str) -> dict[str, object]:
    return {
        "status": "finalized",
        "answer_disposition": "unchanged",
        "proposed_short_answer": "Answer",
        "reference_answer_text": "The candidate satisfies both predicates.",
        "claims": [
            {
                "claim_id": "answer",
                "claim": "The candidate satisfies both predicates.",
                "role": "answer_determining",
                "evidence_window_ids": [window_id],
                "support_explanation": "The acquired official table supports the claim.",
            }
        ],
        "citation_window_ids": [window_id],
        "abandon_reason": None,
    }


def _abandon_payload() -> dict[str, object]:
    return {
        "status": "abandon",
        "answer_disposition": None,
        "proposed_short_answer": None,
        "reference_answer_text": None,
        "claims": [],
        "citation_window_ids": [],
        "abandon_reason": "The proposed path requires a materially new metric.",
    }


def _semantic_payload(
    window_id: str,
    *,
    structured: bool = False,
    omit_structured_window: bool = False,
) -> dict[str, object]:
    return {
        "status": "pass",
        "requirement_findings": [
            {
                "requirement_id": "metric",
                "support_status": "supported",
                "evidence_window_ids": [window_id],
                "explanation": "The table supports both predicates.",
            }
        ],
        "claim_findings": [
            {
                "claim_id": "answer",
                "support_status": "supported",
                "evidence_window_ids": [window_id],
                "explanation": "The table supports the answer claim.",
            }
        ],
        "structured_field_findings": (
            [
                {
                    "schema_path": "candidates[]",
                    "support_status": "supported",
                    "requirement_ids": ["metric"],
                    "claim_ids": ["answer"],
                    "evidence_window_ids": [] if omit_structured_window else [window_id],
                    "explanation": "The requested list value is grounded by the answer claim.",
                }
            ]
            if structured
            else []
        ),
        "unmanifested_material_claims": [],
        "abandon_reason": None,
    }


def _materialization_payload(window_id: str) -> dict[str, object]:
    return {
        "disposition": "materialized",
        "rationale": "The question requests one candidate list.",
        "output_schema_json": json.dumps(
            {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["candidates"],
                "additionalProperties": False,
            }
        ),
        "structured_output_json": json.dumps({"candidates": ["Answer"]}),
        "field_bindings": [
            {
                "schema_path": "candidates[]",
                "answer_evidence": "The grounded reference identifies the candidate.",
                "requirement_ids": ["metric"],
                "claim_ids": ["answer"],
                "evidence_window_ids": [window_id],
            }
        ],
    }


def _not_materializable_payload() -> dict[str, object]:
    return {
        "disposition": "not_materializable",
        "rationale": "The requested value cannot fit the bounded subset.",
        "output_schema_json": None,
        "structured_output_json": None,
        "field_bindings": [],
    }


def _pair_id_from_prompt(prompt: str) -> str:
    match = re.search(r"PAIR-\d{3}", prompt, flags=re.IGNORECASE)
    if match is None:
        raise AssertionError("prompt did not contain a pair marker")
    return match.group(0).upper()


def _window_id_from_prompt(prompt: str) -> str:
    match = re.search(
        r'"(?:window_id|evidence_window_ids)"\s*:\s*(?:\[\s*)?"(window_[^"]+)"',
        prompt,
    )
    if match is None:
        raise AssertionError("prompt did not contain an acquired window")
    return match.group(1)
