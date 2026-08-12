import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from typing import cast
from uuid import UUID

import pytest

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.domain.tool_usage import LlmUsageSummary, ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import known_zero_actual_cost_tool_usage
from harnyx_commons.domain_tweak_generation import (
    AcceptedRouteContext,
    BatchTerminalGenerationError,
    CandidateFailure,
    CandidateStageError,
    DomainTweakFinalizedTask,
    PortfolioAllocation,
    PortfolioCallEvent,
    PortfolioPacket,
    ShortfallRefillPipeline,
    SlotAttemptEvent,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation import refill_pipeline as refill_pipeline_module


class _PortfolioRunner:
    def __init__(self) -> None:
        self.group_sizes: list[int] = []
        self.accepted_route_context_counts: list[int] = []

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        prompt = str(kwargs["prompt"])
        payload = _prompt_payload(prompt)
        rows = cast(list[dict[str, object]], payload["slots"])
        self.group_sizes.append(len(rows))
        self.accepted_route_context_counts.append(len(payload["already_accepted_routes_to_avoid"]))
        packet = PortfolioPacket(
            allocations=tuple(
                PortfolioAllocation(
                    slot=row["slot"],
                    ecosystems=("a", "b", "c", "d", "e"),
                )
                for row in rows
            )
        )
        return StageRunResult(packet, 1.0, known_zero_actual_cost_tool_usage(), actual_llm_cost_usd=0.0)


class _CandidatePipeline:
    def __init__(self, outcomes: Sequence[DomainTweakFinalizedTask | CandidateFailure]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.preferences: list[tuple[int, str]] = []

    async def run(self, allocation: PortfolioAllocation, *, capability_preference: str):
        self.calls += 1
        self.preferences.append((allocation.slot, capability_preference))
        return self.outcomes.pop(0)


class _OneFailedPortfolioRunner(_PortfolioRunner):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        prompt = str(kwargs["prompt"])
        rows = _prompt_rows(prompt)
        self.group_sizes.append(len(rows))
        usage = _usage(1.0)
        if len(rows) == 3 and not self.failed:
            self.failed = True
            raise CandidateStageError(
                "transient_provider",
                "portfolio",
                "temporary pressure",
                retry_after_seconds=0.25,
                tool_usage=usage,
            )
        return StageRunResult(
            PortfolioPacket(
                allocations=tuple(
                    PortfolioAllocation(
                        slot=row["slot"],
                        ecosystems=("a", "b", "c", "d", "e"),
                    )
                    for row in rows
                )
            ),
            1.0,
            usage,
        )


class _OneInvalidPortfolioRunner(_PortfolioRunner):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        prompt = str(kwargs["prompt"])
        rows = _prompt_rows(prompt)
        self.group_sizes.append(len(rows))
        self.calls += 1
        usage = _usage(2.75)
        slot = rows[0]["slot"] + 1 if self.calls == 1 else rows[0]["slot"]
        return StageRunResult(
            PortfolioPacket(
                allocations=(
                    PortfolioAllocation(
                        slot=slot,
                        ecosystems=("a", "b", "c", "d", "e"),
                    ),
                )
            ),
            1.0,
            usage,
        )


class _BatchTerminalCandidatePipeline:
    def __init__(self) -> None:
        self.sibling_started = asyncio.Event()
        self.sibling_cancelled = asyncio.Event()
        self.calls = 0

    async def run(self, _allocation: PortfolioAllocation, *, capability_preference: str):
        del capability_preference
        self.calls += 1
        if self.calls == 1:
            await self.sibling_started.wait()
            raise BatchTerminalGenerationError(
                "provider_auth",
                "credentials rejected",
                stage="reference",
                tool_usage=_usage(2.5),
            )
        self.sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.sibling_cancelled.set()
            raise


class _CompletedSuccessRaceCandidatePipeline:
    def __init__(self) -> None:
        self.release_success = asyncio.Event()
        self.success_returned = asyncio.Event()
        self.calls = 0

    async def run(self, _allocation: PortfolioAllocation, *, capability_preference: str):
        del capability_preference
        self.calls += 1
        if self.calls == 1:
            raise BatchTerminalGenerationError(
                "provider_auth",
                "credentials rejected",
                stage="reference",
            )
        await self.release_success.wait()
        self.success_returned.set()
        return _success(1)


class _BatchTerminalPortfolioRunner:
    def __init__(self) -> None:
        self.sibling_started = asyncio.Event()
        self.sibling_cancelled = asyncio.Event()
        self.calls = 0

    async def run_stage(self, **_kwargs: object) -> StageRunResult:
        self.calls += 1
        if self.calls == 1:
            await self.sibling_started.wait()
            raise BatchTerminalGenerationError(
                "provider_auth",
                "credentials rejected",
                stage="portfolio",
                tool_usage=_usage(1.5),
            )
        self.sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.sibling_cancelled.set()
            raise


class _CompletedPortfolioRaceRunner:
    def __init__(self) -> None:
        self.release_success = asyncio.Event()
        self.success_returned = asyncio.Event()
        self.calls = 0

    async def run_stage(self, **kwargs: object) -> StageRunResult:
        self.calls += 1
        if self.calls == 1:
            raise BatchTerminalGenerationError(
                "provider_auth",
                "credentials rejected",
                stage="portfolio",
                tool_usage=_usage(1.5),
            )
        await self.release_success.wait()
        prompt = str(kwargs["prompt"])
        rows = _prompt_rows(prompt)
        self.success_returned.set()
        return StageRunResult(
            PortfolioPacket(
                allocations=tuple(
                    PortfolioAllocation(
                        slot=row["slot"],
                        ecosystems=("a", "b", "c", "d", "e"),
                    )
                    for row in rows
                )
            ),
            1.0,
            _usage(2.0),
        )


class _UnexpectedPortfolioRunner:
    async def run_stage(self, **_kwargs: object) -> StageRunResult:
        raise RuntimeError("broken portfolio adapter")


class _WrongTypePortfolioRunner:
    async def run_stage(self, **_kwargs: object) -> StageRunResult:
        output = PortfolioAllocation(slot=99, ecosystems=("a", "b", "c", "d", "e"))
        return StageRunResult(output, 1.0, known_zero_actual_cost_tool_usage(), actual_llm_cost_usd=0.0)


def _success(index: int) -> DomainTweakFinalizedTask:
    return DomainTweakFinalizedTask(
        task=MinerTask(
            task_id=UUID(int=index + 1),
            query=Query(text=f"question {index}"),
            reference_answer=ReferenceAnswer(text=f"answer {index}"),
        ),
        route_context=AcceptedRouteContext(
            subject=f"subject {index}",
            route_summary=f"route {index}",
            source_urls=(f"https://example.com/{index}",),
        ),
        tool_usage=known_zero_actual_cost_tool_usage(),
    )


def _prompt_rows(prompt: str) -> list[dict[str, object]]:
    payload = _prompt_payload(prompt)
    return cast(list[dict[str, object]], payload["slots"])


def _prompt_payload(prompt: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(prompt.split("\n", 1)[1]))


def _failure() -> CandidateFailure:
    return CandidateFailure(
        "reasoning_no_generate",
        "question_generation",
        (),
        known_zero_actual_cost_tool_usage(),
        failure_reason="complete route unavailable",
        source_failure_id="source_failure:1",
    )


def _usage(cost: float) -> ToolUsageSummary:
    return ToolUsageSummary(
        llm=LlmUsageSummary(actual_cost=cost),
        actual_total_cost_usd=cost,
        actual_cost_by_provider={"vertex": cost},
    )


@pytest.mark.anyio
async def test_refill_launches_only_current_shortfall_until_exact_completion() -> None:
    """Future failure: the old N-multiple over-generation strategy must not return."""
    runner = _PortfolioRunner()
    candidates = _CandidatePipeline((_success(0), _failure(), _failure(), _success(1), _failure(), _success(2)))
    accepted_slots: list[int] = []

    async def accept(slot: int, task: DomainTweakFinalizedTask) -> None:
        accepted_slots.append(slot)
        assert task.route_context is None
        assert "route_context" not in task.model_dump(mode="json")

    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    ).generate_batch(
        target_count=3,
        on_finalized_task=accept,
    )

    assert runner.group_sizes == [3, 2, 1]
    assert runner.accepted_route_context_counts == [0, 1, 2]
    assert candidates.calls == 6
    assert accepted_slots == [0, 1, 2]
    assert all(item.route_context is None for item in result.finalized_tasks)
    assert result.slot_attempt_count == 6
    assert result.failure_counts == {"reasoning_no_generate": 3}
    assert candidates.preferences == [
        (0, "general_deep_research"),
        (1, "false_premise_correction"),
        (2, "source_conflict_time_uncertainty"),
        (1, "false_premise_correction"),
        (2, "source_conflict_time_uncertainty"),
        (2, "source_conflict_time_uncertainty"),
    ]


@pytest.mark.anyio
async def test_fourteen_output_slots_receive_fixed_preference_counts_without_quota_logic() -> None:
    """Future failure: capability preferences must use ceil-per-kind then truncate in fixed order."""
    candidates = _CandidatePipeline(tuple(_success(index) for index in range(14)))

    result = await ShortfallRefillPipeline(
        runner=_PortfolioRunner(),  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    ).generate_batch(target_count=14)

    assert len(result.finalized_tasks) == 14
    assert Counter(preference for _, preference in candidates.preferences) == {
        "general_deep_research": 3,
        "false_premise_correction": 3,
        "source_conflict_time_uncertainty": 3,
        "evidence_grounded_calculation_or_proof": 3,
        "structured_field_semantics": 2,
    }
    assert [preference for _, preference in sorted(candidates.preferences)] == [
        "general_deep_research",
        "general_deep_research",
        "general_deep_research",
        "false_premise_correction",
        "false_premise_correction",
        "false_premise_correction",
        "source_conflict_time_uncertainty",
        "source_conflict_time_uncertainty",
        "source_conflict_time_uncertainty",
        "evidence_grounded_calculation_or_proof",
        "evidence_grounded_calculation_or_proof",
        "evidence_grounded_calculation_or_proof",
        "structured_field_semantics",
        "structured_field_semantics",
    ]


@pytest.mark.anyio
async def test_portfolio_and_candidate_work_run_inside_distinct_cost_free_parent_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: concurrent orchestration must remain attributable without duplicating stage cost."""
    starts: list[dict[str, object]] = []
    updates: list[tuple[object, dict[str, object]]] = []

    class ObservationScope:
        def __init__(self, observation: object) -> None:
            self.observation = observation

        def __enter__(self) -> object:
            return self.observation

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    def start(name: str, as_type: str, **kwargs: object) -> ObservationScope:
        observation = object()
        starts.append({"name": name, "as_type": as_type, **kwargs, "observation": observation})
        return ObservationScope(observation)

    def update(observation: object, **kwargs: object) -> None:
        updates.append((observation, kwargs))

    monkeypatch.setattr(refill_pipeline_module, "start_metadata_only_observation", start)
    monkeypatch.setattr(refill_pipeline_module, "update_observation_best_effort", update)

    await ShortfallRefillPipeline(
        runner=_PortfolioRunner(),  # type: ignore[arg-type]
        candidate_pipeline=_CandidatePipeline((_success(0),)),  # type: ignore[arg-type]
        attempt_id_factory=lambda: UUID(int=99),
    ).generate_batch(
        target_count=1,
    )

    assert [item["name"] for item in starts] == [
        "miner_task_generation.portfolio_call",
        "miner_task_generation.candidate_attempt",
    ]
    assert cast(dict[str, object], starts[0]["metadata"])["slot_count"] == 1
    assert cast(dict[str, object], starts[1]["metadata"])["attempt_id"] == str(UUID(int=99))
    assert [cast(dict[str, object], kwargs["metadata"])["outcome"] for _, kwargs in updates] == [
        "succeeded",
        "finalized",
    ]
    assert all("cost_details" not in kwargs and "usage_details" not in kwargs for _, kwargs in updates)


@pytest.mark.anyio
async def test_unavailable_candidate_cost_makes_successful_batch_cost_unavailable() -> None:
    """Future failure: exact completion must not turn one missing candidate bill into a known batch prefix."""
    unknown_cost_success = _success(0).model_copy(update={"tool_usage": ToolUsageSummary.zero()})

    result = await ShortfallRefillPipeline(
        runner=_PortfolioRunner(),  # type: ignore[arg-type]
        candidate_pipeline=_CandidatePipeline((unknown_cost_success,)),  # type: ignore[arg-type]
    ).generate_batch(
        target_count=1,
    )

    assert result.tool_usage.actual_total_cost_usd is None
    assert result.tool_usage.llm.actual_cost is None


@pytest.mark.anyio
async def test_candidate_parent_uses_final_outcome_after_duplicate_reclassification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: a duplicate candidate parent must not remain marked finalized."""
    starts: dict[object, str] = {}
    parent_outcomes: dict[str, str] = {}
    events: list[SlotAttemptEvent] = []
    attempt_ids = iter((UUID(int=101), UUID(int=102), UUID(int=103)))

    class ObservationScope:
        def __init__(self, observation: object) -> None:
            self.observation = observation

        def __enter__(self) -> object:
            return self.observation

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    def start(name: str, as_type: str, **kwargs: object) -> ObservationScope:
        observation = object()
        if name == "miner_task_generation.candidate_attempt":
            metadata = cast(dict[str, object], kwargs["metadata"])
            starts[observation] = cast(str, metadata["attempt_id"])
        return ObservationScope(observation)

    def update(observation: object, **kwargs: object) -> None:
        if observation not in starts:
            return
        metadata = cast(dict[str, object], kwargs["metadata"])
        if "outcome" in metadata:
            parent_outcomes[starts[observation]] = cast(str, metadata["outcome"])

    async def record(event: SlotAttemptEvent) -> None:
        events.append(event)

    monkeypatch.setattr(refill_pipeline_module, "start_metadata_only_observation", start)
    monkeypatch.setattr(refill_pipeline_module, "update_observation_best_effort", update)

    await ShortfallRefillPipeline(
        runner=_PortfolioRunner(),  # type: ignore[arg-type]
        candidate_pipeline=_CandidatePipeline((_success(0), _success(0), _success(1))),  # type: ignore[arg-type]
        attempt_id_factory=lambda: next(attempt_ids),
    ).generate_batch(
        target_count=2,
        on_attempt_completed=record,
    )

    assert len(events) == 3
    assert {event.attempt_id: event.outcome for event in events} == parent_outcomes
    assert sorted(parent_outcomes.values()) == ["contract_invalid", "finalized", "finalized"]


@pytest.mark.anyio
async def test_candidate_parent_keeps_final_outcome_when_finalized_callback_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: a persistence callback error must not erase the candidate's final verdict."""
    candidate_observations: set[object] = set()
    parent_outcomes: list[str] = []

    class ObservationScope:
        def __init__(self, observation: object) -> None:
            self.observation = observation

        def __enter__(self) -> object:
            return self.observation

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    def start(name: str, _as_type: str, **_kwargs: object) -> ObservationScope:
        observation = object()
        if name == "miner_task_generation.candidate_attempt":
            candidate_observations.add(observation)
        return ObservationScope(observation)

    def update(observation: object, **kwargs: object) -> None:
        if observation not in candidate_observations:
            return
        metadata = cast(dict[str, object], kwargs["metadata"])
        if "outcome" in metadata:
            parent_outcomes.append(cast(str, metadata["outcome"]))

    async def reject_persistence(_slot: int, _task: DomainTweakFinalizedTask) -> None:
        raise RuntimeError("persistence rejected")

    monkeypatch.setattr(refill_pipeline_module, "start_metadata_only_observation", start)
    monkeypatch.setattr(refill_pipeline_module, "update_observation_best_effort", update)

    with pytest.raises(RuntimeError, match="persistence rejected"):
        await ShortfallRefillPipeline(
            runner=_PortfolioRunner(),  # type: ignore[arg-type]
            candidate_pipeline=_CandidatePipeline((_success(0),)),  # type: ignore[arg-type]
        ).generate_batch(
            target_count=1,
            on_finalized_task=reject_persistence,
        )

    assert parent_outcomes == ["finalized"]


@pytest.mark.anyio
async def test_more_than_ten_slots_groups_portfolios_without_surplus_candidates() -> None:
    runner = _PortfolioRunner()
    candidates = _CandidatePipeline(tuple(_success(index) for index in range(13)))
    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    ).generate_batch(
        target_count=13,
    )

    assert sorted(runner.group_sizes) == [3, 10]
    assert candidates.calls == 13
    assert result.slot_attempt_count == 13


@pytest.mark.anyio
async def test_failed_attempt_preserves_private_blocker_without_serializing_it() -> None:
    """Future failure: orchestration needs the blocker, while telemetry must not expose model-authored text."""
    runner = _PortfolioRunner()
    candidates = _CandidatePipeline((_failure(), _success(0)))
    events: list[SlotAttemptEvent] = []

    async def record(event: SlotAttemptEvent) -> None:
        events.append(event)

    await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    ).generate_batch(
        target_count=1,
        on_attempt_completed=record,
    )

    assert events[0].outcome == "reasoning_no_generate"
    assert events[0].failure_class is None
    assert events[0].failure_reason == "complete route unavailable"
    assert events[0].source_failure_id == "source_failure:1"
    serialized = events[0].model_dump(mode="json")
    assert "failure_reason" not in serialized
    assert "source_failure_id" not in serialized
    assert events[1].outcome == "finalized"
    assert not events[1].repaired


@pytest.mark.anyio
async def test_one_failed_portfolio_group_preserves_other_successes_and_counts_shared_cost_once() -> None:
    """Future failure: one transient group must neither stop siblings nor duplicate its shared provider cost."""
    runner = _OneFailedPortfolioRunner()
    candidates = _CandidatePipeline(tuple(_success(index) for index in range(13)))
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
        sleep=sleep,
    ).generate_batch(
        target_count=13,
    )

    assert sorted(runner.group_sizes) == [3, 3, 10]
    assert candidates.calls == 13
    assert result.slot_attempt_count == 16
    assert result.failure_counts == {"transient_provider": 3}
    assert result.tool_usage.actual_total_cost_usd == 3.0
    assert delays == [0.25]


@pytest.mark.anyio
async def test_host_rejected_portfolio_preserves_billable_usage_exactly_once() -> None:
    """Future failure: host slot validation must not erase a schema-valid portfolio call's billed cost."""
    runner = _OneInvalidPortfolioRunner()
    candidates = _CandidatePipeline((_success(0),))
    events: list[PortfolioCallEvent] = []

    async def record(event: PortfolioCallEvent) -> None:
        events.append(event)

    result = await ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    ).generate_batch(
        target_count=1,
        on_portfolio_completed=record,
    )

    assert [event.outcome for event in events] == ["contract_invalid", "succeeded"]
    assert events[0].tool_usage.actual_total_cost_usd == 2.75
    assert result.tool_usage.actual_total_cost_usd == 5.5
    assert result.failure_counts == {"contract_invalid": 1}


@pytest.mark.anyio
async def test_batch_terminal_candidate_failure_cancels_and_drains_siblings() -> None:
    """Future failure: shared provider auth faults must stop the batch without orphaning candidate tasks."""
    runner = _PortfolioRunner()
    candidates = _BatchTerminalCandidatePipeline()
    pipeline = ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    )

    events: list[SlotAttemptEvent] = []

    async def record(event: SlotAttemptEvent) -> None:
        events.append(event)

    with pytest.raises(BatchTerminalGenerationError, match="credentials rejected") as captured:
        await pipeline.generate_batch(
            target_count=2,
            on_attempt_completed=record,
        )

    assert candidates.sibling_cancelled.is_set()
    assert len(events) == 1
    assert events[0].outcome == "batch_terminal"
    assert events[0].terminal_stage == "reference"
    assert events[0].tool_usage.actual_total_cost_usd == 2.5
    assert captured.value.tool_usage.actual_total_cost_usd is None


@pytest.mark.anyio
async def test_completed_success_is_retained_before_terminal_propagates() -> None:
    """Future failure: a sibling that finishes during terminal reporting must still be persisted once."""
    candidates = _CompletedSuccessRaceCandidatePipeline()
    events: list[SlotAttemptEvent] = []
    finalized_slots: list[int] = []

    async def record_attempt(event: SlotAttemptEvent) -> None:
        events.append(event)
        if event.outcome == "batch_terminal":
            candidates.release_success.set()
            await candidates.success_returned.wait()

    async def accept(slot: int, _task: DomainTweakFinalizedTask) -> None:
        finalized_slots.append(slot)

    with pytest.raises(BatchTerminalGenerationError, match="credentials rejected"):
        await ShortfallRefillPipeline(
            runner=_PortfolioRunner(),  # type: ignore[arg-type]
            candidate_pipeline=candidates,  # type: ignore[arg-type]
        ).generate_batch(
            target_count=2,
            on_attempt_completed=record_attempt,
            on_finalized_task=accept,
        )

    assert candidates.success_returned.is_set()
    assert finalized_slots == [1]
    assert [event.outcome for event in events] == ["batch_terminal", "finalized"]


@pytest.mark.anyio
async def test_batch_terminal_portfolio_failure_cancels_and_drains_siblings() -> None:
    """Future failure: a shared portfolio runtime fault must stop the batch and preserve its billable event."""
    runner = _BatchTerminalPortfolioRunner()
    candidates = _CandidatePipeline(())
    pipeline = ShortfallRefillPipeline(
        runner=runner,  # type: ignore[arg-type]
        candidate_pipeline=candidates,  # type: ignore[arg-type]
    )
    events: list[PortfolioCallEvent] = []

    async def record(event: PortfolioCallEvent) -> None:
        events.append(event)

    with pytest.raises(BatchTerminalGenerationError, match="credentials rejected") as captured:
        await pipeline.generate_batch(
            target_count=13,
            on_portfolio_completed=record,
        )

    assert runner.sibling_cancelled.is_set()
    assert candidates.calls == 0
    assert len(events) == 1
    assert events[0].outcome == "batch_terminal"
    assert events[0].failure_class == "provider_auth"
    assert events[0].tool_usage.actual_total_cost_usd == 1.5
    assert captured.value.tool_usage.actual_total_cost_usd is None


@pytest.mark.anyio
async def test_completed_portfolio_call_is_recorded_before_terminal_propagates() -> None:
    """Future failure: a paid sibling portfolio result must not disappear during terminal reporting."""
    runner = _CompletedPortfolioRaceRunner()
    events: list[PortfolioCallEvent] = []

    async def record(event: PortfolioCallEvent) -> None:
        events.append(event)
        if event.outcome == "batch_terminal":
            runner.release_success.set()
            await runner.success_returned.wait()

    with pytest.raises(BatchTerminalGenerationError, match="credentials rejected") as captured:
        await ShortfallRefillPipeline(
            runner=runner,  # type: ignore[arg-type]
            candidate_pipeline=_CandidatePipeline(()),  # type: ignore[arg-type]
        ).generate_batch(
            target_count=13,
            on_portfolio_completed=record,
        )

    assert runner.success_returned.is_set()
    assert [event.outcome for event in events] == ["batch_terminal", "succeeded"]
    assert [event.tool_usage.actual_total_cost_usd for event in events] == [1.5, 2.0]
    assert captured.value.tool_usage.actual_total_cost_usd == 3.5


@pytest.mark.anyio
async def test_unexpected_portfolio_exception_emits_a_typed_terminal_event() -> None:
    """Future failure: a shared host-code defect must not bypass the bounded portfolio event."""
    events: list[PortfolioCallEvent] = []

    async def record(event: PortfolioCallEvent) -> None:
        events.append(event)

    with pytest.raises(BatchTerminalGenerationError) as captured:
        await ShortfallRefillPipeline(
            runner=_UnexpectedPortfolioRunner(),  # type: ignore[arg-type]
            candidate_pipeline=_CandidatePipeline(()),  # type: ignore[arg-type]
        ).generate_batch(
            target_count=1,
            on_portfolio_completed=record,
        )

    assert captured.value.failure_class == "unexpected_pipeline_failure"
    assert len(events) == 1
    assert events[0].outcome == "batch_terminal"
    assert events[0].failure_class == "unexpected_pipeline_failure"


@pytest.mark.anyio
async def test_wrong_portfolio_output_type_is_batch_terminal() -> None:
    """Future failure: an internal executor type defect must not enter paid portfolio refill."""
    events: list[PortfolioCallEvent] = []

    async def record(event: PortfolioCallEvent) -> None:
        events.append(event)

    with pytest.raises(BatchTerminalGenerationError) as captured:
        await ShortfallRefillPipeline(
            runner=_WrongTypePortfolioRunner(),  # type: ignore[arg-type]
            candidate_pipeline=_CandidatePipeline(()),  # type: ignore[arg-type]
        ).generate_batch(
            target_count=1,
            on_portfolio_completed=record,
        )

    assert captured.value.failure_class == "unexpected_pipeline_failure"
    assert [event.outcome for event in events] == ["batch_terminal"]


def test_retry_delay_honors_provider_value_and_caps_fallback_at_sixty_seconds() -> None:
    """Future failure: refill pacing must follow the approved Retry-After and full-jitter bounds."""
    observed: list[tuple[float, float]] = []

    def choose(low: float, high: float) -> float:
        observed.append((low, high))
        return high

    pipeline = ShortfallRefillPipeline(
        runner=_PortfolioRunner(),  # type: ignore[arg-type]
        candidate_pipeline=_CandidatePipeline(()),  # type: ignore[arg-type]
        random_uniform=choose,
    )

    assert pipeline._retry_delay(20, (3.0, 7.5)) == (7.5, "provider_retry_after")
    assert pipeline._retry_delay(20, ()) == (60.0, "full_jitter_exponential")
    assert observed == [(0.0, 60.0)]
