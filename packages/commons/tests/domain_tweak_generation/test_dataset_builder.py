from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain_tweak_generation.dataset_builder import DomainTweakMinerTaskDatasetBuilder
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhaseResult,
    DomainTweakBatchGenerationConfig,
    DomainTweakBatchGenerationResult,
    DomainTweakDiscardedCandidate,
    DomainTweakFinalizedTask,
    DomainTweakReviewedQuestion,
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
    MinerTaskDatasetRequest,
    MinerTaskModelSpec,
)

pytestmark = pytest.mark.anyio("asyncio")


@dataclass
class _PairSource:
    pairs: tuple[DomainTweakPairInput, ...]
    requested_counts: list[int] = field(default_factory=list)

    async def load_pair_inputs(
        self,
        *,
        batch_id: UUID,
        timestamp: datetime,
        requested_count: int,
    ) -> tuple[DomainTweakPairInput, ...]:
        _ = batch_id, timestamp
        self.requested_counts.append(requested_count)
        return self.pairs[:requested_count]


@dataclass
class _BatchPipeline:
    result: DomainTweakBatchGenerationResult
    configs: list[DomainTweakBatchGenerationConfig] = field(default_factory=list)

    async def generate_batch(
        self,
        pair_inputs: tuple[DomainTweakPairInput, ...],
        config: DomainTweakBatchGenerationConfig,
    ) -> DomainTweakBatchGenerationResult:
        _ = pair_inputs
        self.configs.append(config)
        return self.result


async def test_builder_requests_existing_four_times_pair_cap_and_returns_exact_target() -> None:
    reviewed = _reviewed()
    finalized = _finalized(reviewed)
    pair_source = _PairSource((_pair(),) * 8)
    pipeline = _BatchPipeline(
        DomainTweakBatchGenerationResult(
            target_count=1,
            selected_questions=(reviewed,),
            finalized_tasks=(finalized,),
            underfilled=False,
        )
    )
    builder = DomainTweakMinerTaskDatasetBuilder(
        pair_source=pair_source,
        batch_pipeline=pipeline,
    )

    tasks = await builder.build(_request(minimum_task_total=1))

    assert tasks == (finalized.task,)
    assert pair_source.requested_counts == [4]
    assert pipeline.configs[0].target_count == 1


async def test_builder_reports_typed_discard_when_exact_target_is_underfilled() -> None:
    reviewed = _reviewed()
    pipeline = _BatchPipeline(
        DomainTweakBatchGenerationResult(
            target_count=1,
            selected_questions=(reviewed,),
            discarded_candidates=(
                DomainTweakDiscardedCandidate(
                    reviewed_question=reviewed,
                    terminal_stage="semantic_support_gate",
                    reason="semantic_support_abandon",
                ),
            ),
            candidate_finalization_attempt_count=1,
            underfilled=True,
        )
    )
    builder = DomainTweakMinerTaskDatasetBuilder(
        pair_source=_PairSource((_pair(),)),
        batch_pipeline=pipeline,
    )

    with pytest.raises(RuntimeError, match="source-aware candidate finalization was discarded"):
        await builder.build(_request(minimum_task_total=1))


async def test_builder_requires_fixed_generation_timestamp() -> None:
    builder = DomainTweakMinerTaskDatasetBuilder(
        pair_source=_PairSource((_pair(),)),
        batch_pipeline=_BatchPipeline(DomainTweakBatchGenerationResult(target_count=1, underfilled=True)),
    )

    with pytest.raises(ValueError, match="requires request.created_at"):
        await builder.build_with_result(_request(minimum_task_total=1, created_at=None))


def _request(
    *,
    minimum_task_total: int,
    created_at: datetime | None = datetime(2026, 7, 21, tzinfo=UTC),
) -> MinerTaskDatasetRequest:
    spec = MinerTaskModelSpec(
        provider="vertex",
        model="unused",
        temperature=None,
        max_output_tokens=None,
    )
    return MinerTaskDatasetRequest(
        batch_id=UUID("00000000-0000-0000-0000-000000000042"),
        created_at=created_at,
        minimum_task_total=minimum_task_total,
        generation_task_buffer=0,
        generation_spec=spec,
        reference_spec=spec,
    )


def _pair() -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id="pair-001",
        deepsearchqa_form_target="Which candidates satisfy both?",
        deepresearch9k_domain_target="Public tables",
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
    )


def _reviewed() -> DomainTweakReviewedQuestion:
    blueprint = DomainTweakFormBlueprint(
        status="proceed",
        operation="Filter a set.",
        load_bearing_invariants=("two predicates",),
        non_load_bearing_surface_features=(),
        retrieval_boundary="Sources supply values.",
        answer_shape="List.",
        semantic_ambiguities=(),
        no_generate_reason=None,
    )
    packet = DomainTweakQuestionPacket(
        status="ready",
        question="Which candidates satisfy both?",
        short_answer="A",
        solution_steps=("Read the values.",),
        claims=(
            DomainTweakClaim(
                claim_id="answer",
                claim="A satisfies both.",
                role="answer_determining",
                support_mode="external_source",
                support_explanation="The source records both values.",
            ),
        ),
        evidence_declarations=(
            DomainTweakEvidenceDeclaration(
                evidence_id="evidence",
                source_url="https://example.com/source",
                source_title="Source",
                source_locator=None,
                claimed_excerpt="A satisfies both.",
                supported_claim_ids=("answer",),
                support_explanation="The source records both values.",
            ),
        ),
        no_generate_reason=None,
    )
    review = DomainTweakFormReview(
        form_match=True,
        reviewer_feedback="The form is preserved.",
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
                explanation="Present." if category == "metric_or_field_relation" else "Absent.",
            )
            for category in DOMAIN_TWEAK_REQUIREMENT_CATEGORIES
        ),
    )
    return DomainTweakReviewedQuestion(
        pair_input=_pair(),
        form_blueprint=blueprint,
        question_packet=packet,
        form_review=review,
        form_blueprint_result=DomainTweakAdkPhaseResult(
            phase="form_blueprint", terminal_status="validated", parsed_output=blueprint
        ),
        question_generation_result=DomainTweakAdkPhaseResult(
            phase="question_generation", terminal_status="validated", parsed_output=packet
        ),
        form_review_result=DomainTweakAdkPhaseResult(
            phase="form_review", terminal_status="validated", parsed_output=review
        ),
    )


def _finalized(reviewed: DomainTweakReviewedQuestion) -> DomainTweakFinalizedTask:
    return DomainTweakFinalizedTask(
        reviewed_question=reviewed,
        reference_answer_result=DomainTweakAdkPhaseResult(
            phase="reference_answer_generation", terminal_status="validated"
        ),
        semantic_support_result=DomainTweakAdkPhaseResult(phase="semantic_support_gate", terminal_status="validated"),
        task=MinerTask(
            task_id=UUID("00000000-0000-0000-0000-000000000001"),
            query=Query(text="Which candidates satisfy both?"),
            reference_answer=ReferenceAnswer(text="A satisfies both."),
        ),
    )
