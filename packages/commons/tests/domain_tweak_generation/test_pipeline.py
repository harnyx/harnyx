from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from harnyx_commons.domain_tweak_generation.adk_runner import DomainTweakAdkRunner, DomainTweakAdkTurn
from harnyx_commons.domain_tweak_generation.pipeline import DomainTweakGenerationPipeline
from harnyx_commons.domain_tweak_generation.source_evidence import BatchSourceEvidence
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkRunConfig,
    DomainTweakDiscardedCandidate,
    DomainTweakFinalizedTask,
    DomainTweakReviewedQuestion,
    DomainTweakSourceEvidencePolicy,
)
from harnyx_commons.miner_task_generation import (
    DOMAIN_TWEAK_REQUIREMENT_CATEGORIES,
    DomainTweakClaim,
    DomainTweakEvidenceDeclaration,
    DomainTweakFormBlueprint,
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionPacket,
    DomainTweakQuestionRequirement,
    DomainTweakRequirementCategoryAudit,
)
from harnyx_commons.tools.extraction_models import ExtractedPage, ExtractPagesRequest, ExtractPagesResponse
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata, SearchProviderResult

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _ExtractionProvider:
    delay_seconds: float = 0.0
    requests: list[ExtractPagesRequest] = field(default_factory=list)

    async def extract_pages(
        self,
        request: ExtractPagesRequest,
    ) -> SearchProviderResult[ExtractPagesResponse]:
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return SearchProviderResult(
            response=ExtractPagesResponse(
                pages=tuple(
                    ExtractedPage(
                        url=url,
                        title="Official results",
                        content="The official table shows that A satisfies both predicates.",
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
    form_match: bool = True
    reference_mode: str = "finalized"
    semantic_mode: str = "pass"
    calls: list[dict[str, object]] = field(default_factory=list)

    async def __call__(self, **kwargs: object) -> DomainTweakAdkTurn:
        self.calls.append(kwargs)
        phase = str(kwargs["phase"])
        prompt = str(kwargs["prompt"])
        if phase == "form_blueprint":
            payload = _blueprint().model_dump(mode="json")
        elif phase == "question_generation":
            payload = _packet().model_dump(mode="json")
        elif phase == "form_review":
            payload = _review(form_match=self.form_match).model_dump(mode="json")
        elif phase == "reference_answer_generation":
            window_id = _window_id_from_prompt(prompt)
            payload = self._reference_payload(window_id)
        elif phase == "semantic_support_gate":
            window_id = _window_id_from_prompt(prompt)
            payload = self._semantic_payload(window_id)
        else:
            raise AssertionError(f"unexpected phase {phase}")
        return DomainTweakAdkTurn(final_text=json.dumps(payload), events=())

    def _reference_payload(self, window_id: str) -> dict[str, object]:
        if self.reference_mode == "abandon":
            return {
                "status": "abandon",
                "answer_disposition": None,
                "proposed_short_answer": None,
                "reference_answer_text": None,
                "claims": [],
                "citation_window_ids": [],
                "abandon_reason": "The proposed path requires a materially different metric.",
            }
        selected_window = "window-invented" if self.reference_mode == "unknown_window" else window_id
        return {
            "status": "finalized",
            "answer_disposition": "unchanged",
            "proposed_short_answer": "A",
            "reference_answer_text": "A satisfies both predicates.  Exact spacing is retained.",
            "claims": [
                {
                    "claim_id": "answer",
                    "claim": "A satisfies both predicates.",
                    "role": "answer_determining",
                    "evidence_window_ids": [selected_window],
                    "support_explanation": "The acquired table records both predicate values.",
                }
            ],
            "citation_window_ids": [selected_window],
            "abandon_reason": None,
        }

    def _semantic_payload(self, window_id: str) -> dict[str, object]:
        supported = self.semantic_mode == "pass"
        return {
            "status": "pass" if supported or self.semantic_mode == "missing_requirement" else "abandon",
            "requirement_findings": (
                []
                if self.semantic_mode == "missing_requirement"
                else [
                    {
                        "requirement_id": "metric",
                        "support_status": "supported" if supported else "unsupported",
                        "evidence_window_ids": [window_id],
                        "explanation": "The window establishes both predicates.",
                    }
                ]
            ),
            "claim_findings": [
                {
                    "claim_id": "answer",
                    "support_status": "supported" if supported else "unsupported",
                    "evidence_window_ids": [window_id],
                    "explanation": "The window establishes the answer claim.",
                }
            ],
            "unmanifested_material_claims": [],
            "abandon_reason": (
                None
                if supported or self.semantic_mode == "missing_requirement"
                else "The evidence does not establish the exact metric."
            ),
        }


async def test_question_pipeline_uses_blueprint_then_one_qg_and_one_form_review() -> None:
    executor = _Executor()
    pipeline = _pipeline(executor)

    result = await pipeline.generate_reviewed_question(_pair())

    assert result.form_blueprint.terminal_status == "validated"
    assert result.question_generation is not None
    assert result.form_review is not None
    assert [call["phase"] for call in executor.calls] == [
        "form_blueprint",
        "question_generation",
        "form_review",
    ]
    assert [call["search_enabled"] for call in executor.calls] == [False, True, False]


async def test_form_rejection_does_not_repair_or_retry_same_pair() -> None:
    executor = _Executor(form_match=False)
    result = await _pipeline(executor).generate_reviewed_question(_pair())

    assert result.form_review is not None
    assert result.form_review.terminal_status == "form_rejected"
    assert len(executor.calls) == 3


async def test_finalize_hydrates_only_acquired_window_and_preserves_answer_text_exactly() -> None:
    executor = _Executor()
    provider = _ExtractionProvider()
    reviewed = _reviewed()
    pipeline = _pipeline(executor, provider=provider)

    outcome = await pipeline.finalize_task(reviewed)

    assert isinstance(outcome, DomainTweakFinalizedTask)
    assert outcome.task.reference_answer.text == "A satisfies both predicates.  Exact spacing is retained."
    assert len(outcome.task.reference_answer.citations) == 1
    assert outcome.task.reference_answer.citations[0].url == "https://example.com/source"
    reference_call = next(call for call in executor.calls if call["phase"] == "reference_answer_generation")
    gate_call = next(call for call in executor.calls if call["phase"] == "semantic_support_gate")
    assert reference_call["search_enabled"] is True
    assert reference_call["function_tool_names"] == ("read_cached_source", "acquire_sources")
    assert gate_call["search_enabled"] is False
    assert gate_call["function_tool_names"] == ()


async def test_unacquired_reference_window_is_discarded_before_semantic_gate() -> None:
    executor = _Executor(reference_mode="unknown_window")
    outcome = await _pipeline(executor, provider=_ExtractionProvider()).finalize_task(_reviewed())

    assert isinstance(outcome, DomainTweakDiscardedCandidate)
    assert outcome.terminal_stage == "deterministic_evidence_delivery"
    assert outcome.reason == "delivery_validation_failed"
    assert not any(call["phase"] == "semantic_support_gate" for call in executor.calls)


async def test_semantic_abandon_discards_complete_reference_answer() -> None:
    executor = _Executor(semantic_mode="abandon")
    outcome = await _pipeline(executor, provider=_ExtractionProvider()).finalize_task(_reviewed())

    assert isinstance(outcome, DomainTweakDiscardedCandidate)
    assert outcome.terminal_stage == "semantic_support_gate"
    assert outcome.reason == "semantic_support_abandon"


async def test_semantic_gate_missing_exact_requirement_id_is_observed_as_validation_failure() -> None:
    executor = _Executor(semantic_mode="missing_requirement")
    outcome = await _pipeline(executor, provider=_ExtractionProvider()).finalize_task(_reviewed())

    assert isinstance(outcome, DomainTweakDiscardedCandidate)
    assert outcome.terminal_stage == "semantic_support_gate"
    assert outcome.reason == "validation_failed"
    assert outcome.stage_summaries[-1].outcome == "validation_failed"


async def test_initial_acquisition_obeys_the_stage_hard_timeout() -> None:
    executor = _Executor()
    outcome = await _pipeline(
        executor,
        provider=_ExtractionProvider(delay_seconds=60),
        timeout_seconds=0.01,
    ).finalize_task(_reviewed())

    assert isinstance(outcome, DomainTweakDiscardedCandidate)
    assert outcome.terminal_stage == "initial_evidence_acquisition"
    assert outcome.reason == "timeout"
    assert outcome.stage_summaries[0].outcome == "timeout"
    assert not executor.calls


def _pipeline(
    executor: _Executor,
    *,
    provider: _ExtractionProvider | None = None,
    timeout_seconds: float = 600,
) -> DomainTweakGenerationPipeline:
    source_evidence = (
        BatchSourceEvidence(
            provider=provider,
            policy=DomainTweakSourceEvidencePolicy(),
            client_model="gemini-test",
        )
        if provider is not None
        else None
    )
    return DomainTweakGenerationPipeline(
        config=DomainTweakAdkRunConfig(
            model="gemini-test",
            max_retries=0,
            phase_timeout_seconds=timeout_seconds,
        ),
        runner=DomainTweakAdkRunner(turn_executor=executor),
        source_evidence=source_evidence,
    )


def _reviewed() -> DomainTweakReviewedQuestion:
    blueprint = _blueprint()
    packet = _packet()
    review = _review()
    return DomainTweakReviewedQuestion(
        pair_input=_pair(),
        form_blueprint=blueprint,
        question_packet=packet,
        form_review=review,
        form_blueprint_result=_phase_result("form_blueprint", blueprint),
        question_generation_result=_phase_result("question_generation", packet),
        form_review_result=_phase_result("form_review", review),
    )


def _phase_result(phase: str, parsed_output: object):
    from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkPhaseResult

    return DomainTweakAdkPhaseResult(
        phase=phase,
        terminal_status="validated",
        parsed_output=parsed_output,
    )


def _pair() -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which candidates satisfy both predicates?",
        deepresearch9k_domain_target="Public results tables",
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
    )


def _blueprint() -> DomainTweakFormBlueprint:
    return DomainTweakFormBlueprint(
        status="proceed",
        operation="Filter a closed candidate set by two retrieved predicates.",
        load_bearing_invariants=("closed universe", "two predicates"),
        non_load_bearing_surface_features=(),
        retrieval_boundary="Sources supply predicate values.",
        answer_shape="Exhaustive list.",
        semantic_ambiguities=(),
        no_generate_reason=None,
    )


def _packet() -> DomainTweakQuestionPacket:
    return DomainTweakQuestionPacket(
        status="ready",
        question="Which candidates satisfy both predicates?",
        short_answer="A",
        solution_steps=("Read the table.", "Intersect the qualifying sets."),
        claims=(
            DomainTweakClaim(
                claim_id="answer",
                claim="A satisfies both predicates.",
                role="answer_determining",
                support_mode="external_source",
                support_explanation="The official table records both values.",
            ),
        ),
        evidence_declarations=(
            DomainTweakEvidenceDeclaration(
                evidence_id="evidence-1",
                source_url="https://example.com/source",
                source_title="Official results",
                source_locator="Results table",
                claimed_excerpt="A satisfies both predicates.",
                supported_claim_ids=("answer",),
                support_explanation="The table records both predicate values.",
            ),
        ),
        no_generate_reason=None,
    )


def _review(*, form_match: bool = True) -> DomainTweakFormReview:
    return DomainTweakFormReview(
        form_match=form_match,
        reviewer_feedback=("The form is preserved." if form_match else "The answer shape changed."),
        question_requirements=(
            DomainTweakQuestionRequirement(
                requirement_id="metric",
                category="metric_or_field_relation",
                requirement="Both predicates must hold.",
                required_relation="derived_calculation",
            ),
        ),
        requirement_category_audit=tuple(
            DomainTweakRequirementCategoryAudit(
                category=category,
                present=category == "metric_or_field_relation",
                explanation=("Present." if category == "metric_or_field_relation" else "Absent."),
            )
            for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
        ),
    )


def _window_id_from_prompt(prompt: str) -> str:
    match = re.search(r'"window_id": "(window_[^"]+)"', prompt)
    if match is None:
        raise AssertionError("prompt did not contain an acquired window")
    return match.group(1)
