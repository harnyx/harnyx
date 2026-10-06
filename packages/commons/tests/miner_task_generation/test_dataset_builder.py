"""Concurrent slot outcomes protect partial progress and fixed candidate counts."""

import asyncio
import json
import random
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from time import monotonic
from uuid import uuid4

import pytest
from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import FakeRunner, seed

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_commons.miner_task_generation import MinerTaskDatasetBuilder, MinerTaskDatasetRequest
from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError, CandidateResult, FinalizedTask
from harnyx_commons.miner_task_generation.dataset_builder import UnderfilledDatasetError

pytestmark = pytest.mark.anyio("asyncio")


def request(count=2):
    return MinerTaskDatasetRequest(
        batch_id=uuid4(),
        created_at=datetime.now(UTC),
        minimum_task_total=count,
    )


class SeedRunner(FakeRunner):
    async def generate_seeds(self, count, session, deadline, **kwargs):
        return [seed().model_copy(update={"seed_id": str(index)}) for index in range(count)]


@pytest.mark.parametrize("field", ["seed_id", "seed_question", "reference_answer", "scope"])
async def test_invalid_seed_stops_before_candidate_calls_and_preserves_seed_receipt(monkeypatch, field):
    """Invalid seed payloads must not start research or lose the already-paid seed call."""
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation.agent_runner import CallCapture, GenerationAgentRunner
    from harnyx_commons.miner_task_generation.contracts import AgentResult

    calls = []

    async def invoke(_runner, role, message, session, deadline):
        calls.append(role)
        assert role == "seed"
        capture = CallCapture(session, role, "gpt-6-luna", "openai")
        capture.attempted = 1
        capture.account(LlmUsage(prompt_tokens=10, completion_tokens=20, web_search_calls=0))
        capture.finish()
        return AgentResult(answer=json.dumps({"seeds": [seed().model_dump() | {field: " \t"}]}))

    monkeypatch.setattr(GenerationAgentRunner, "invoke", invoke)
    builder = MinerTaskDatasetBuilder(runner=GenerationAgentRunner(project_id=None, judge=None))
    result = await builder.build_with_result(request(1))
    assert calls == ["seed"] and not result.finalized_tasks
    assert result.error_type == "ValidationError"
    assert result.tool_usage.llm.call_count == 1 and result.tool_usage.llm.prompt_tokens == 10
    assert result.seed_attempted_calls == 1 and result.seed_missing_usage_calls == 0
    assert result.candidates[0].seed is None and result.candidates[0].status == "operational_failure"


async def test_shared_seed_fault_propagates_instead_of_becoming_candidate_underfill():
    error = BatchTerminalGenerationError("provider_auth", "Shared provider fault", stage="question_generation")

    class FailedSeeds(SeedRunner):
        async def generate_seeds(self, *args, **kwargs):
            raise error

    builder = MinerTaskDatasetBuilder(runner=FailedSeeds())
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await builder.build_with_result(request())
    assert caught.value is error


class Slots:
    def __init__(self, fail_slot=False, duplicate=False):
        self.started = []
        self.release = asyncio.Event()
        self.fail_slot = fail_slot
        self.duplicate = duplicate

    async def run(self, seed, *, mode, fast, session, deadline, result=None):
        slot = int(seed.seed_id)
        self.started.append(slot)
        if len(self.started) == 2:
            self.release.set()
        await asyncio.wait_for(self.release.wait(), 1)
        if slot == 1 and self.fail_slot:
            return CandidateResult(seed=seed, mode=mode, status="exhausted")
        task = MinerTask(
            task_id=uuid4(),
            query=Query(text=("Same  Question" if slot == 0 else " same\nquestion ") if self.duplicate else str(slot)),
            reference_answer=ReferenceAnswer(text="answer"),
        )
        return CandidateResult(seed=seed, mode=mode, status="finalized", finalized=FinalizedTask(task=task))


async def test_generation_after_one_hour_retains_time_until_two_hour_deadline():
    class DeadlineRunner(SeedRunner):
        async def generate_seeds(self, count, session, deadline, **kwargs):
            assert 1790 < deadline - monotonic() <= 1800
            return await super().generate_seeds(count, session, deadline, **kwargs)

    builder = MinerTaskDatasetBuilder(runner=DeadlineRunner())
    builder._pipeline = Slots()
    pending = request().model_copy(update={"created_at": datetime.now(UTC) - timedelta(minutes=90)})
    result = await builder.build_with_result(pending)
    assert len(result.finalized_tasks) == 2


async def test_slots_start_concurrently_and_preserve_partial_success_without_replacements():
    builder = MinerTaskDatasetBuilder(runner=SeedRunner(), rng=random.Random(1))  # noqa: S311 - reproducible sampling
    slots = Slots(fail_slot=True)
    builder._pipeline = slots
    accepted = []

    async def accept(slot, result):
        accepted.append(slot)

    result = await builder.build_with_result(request(), on_finalized_task=accept)
    assert set(slots.started) == {0, 1}
    assert accepted == [0]
    assert len(result.candidates) == 2 and len(result.finalized_tasks) == 1
    with pytest.raises(UnderfilledDatasetError) as error:
        await builder.build(request())
    assert len(error.value.result.candidates) == 2


async def test_identical_finalized_questions_are_distinct_eligible_tasks():
    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    slots = Slots(duplicate=True)
    builder._pipeline = slots
    result = await builder.build_with_result(request())
    assert len(result.finalized_tasks) == 2
    assert len({item.task.task_id for item in result.finalized_tasks}) == 2
    assert all(candidate.status == "finalized" and candidate.error_type is None for candidate in result.candidates)
    assert len(slots.started) == 2


async def test_terminal_configuration_fault_cancels_and_drains_sibling_generation():
    failure = BatchTerminalGenerationError("provider_authentication", "denied", stage="reference")
    sibling_started = asyncio.Event()
    sibling_drained = asyncio.Event()

    class TerminalSlots:
        async def run(self, seed, **kwargs):
            if seed.seed_id == "0":
                await sibling_started.wait()
                raise failure
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_drained.set()

    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    builder._pipeline = TerminalSlots()
    with pytest.raises(BatchTerminalGenerationError) as caught:
        await asyncio.wait_for(builder.build_with_result(request()), 1)
    assert caught.value is failure
    assert sibling_drained.is_set()


async def test_confirmed_slot_rejection_does_not_hide_completed_siblings():
    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    builder._pipeline = Slots()
    accepted = []
    from harnyx_commons.miner_task_generation.contracts import FinalizedTaskRejectedError

    failure = FinalizedTaskRejectedError("slot expired")

    async def accept(slot, result):
        if slot == 0:
            raise failure
        accepted.append(slot)

    with pytest.raises(FinalizedTaskRejectedError) as error:
        await builder.build_with_result(request(), on_finalized_task=accept)
    assert error.value is failure
    assert accepted == [1]


@pytest.mark.parametrize("terminal", [False, True])
async def test_shared_acceptance_fault_stops_active_sibling_and_preserves_committed_tasks_and_usage(terminal):
    """Shared storage failures must stop spending without losing committed progress or receipts."""
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation import BatchGenerationResult
    from harnyx_commons.miner_task_generation.agent_runner import CallCapture

    accepted = []
    prior_accepted = asyncio.Event()
    sibling_active = asyncio.Event()
    sibling_drained = asyncio.Event()
    failure = (
        BatchTerminalGenerationError("storage_configuration", "shared acceptance failure", stage="reference")
        if terminal
        else RuntimeError("shared storage failure")
    )

    class ActiveSlots(Slots):
        async def run(self, seed, *, mode, fast, session, deadline, result=None):
            slot = int(seed.seed_id)
            capture = CallCapture(session, "solver", "gemini-3.8-flash", "vertex")
            capture.attempted = 1
            capture.account(LlmUsage(prompt_tokens=10, completion_tokens=5, web_search_calls=0))
            try:
                if slot == 1:
                    await prior_accepted.wait()
                    await sibling_active.wait()
                if slot == 2:
                    sibling_active.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        sibling_drained.set()
                return await super().run(seed, mode=mode, fast=fast, session=session, deadline=deadline, result=result)
            finally:
                capture.finish()

    # These candidates do not need the two-slot rendezvous from the fixture.
    pipeline = ActiveSlots()
    pipeline.release.set()
    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    builder._pipeline = pipeline
    outcome = BatchGenerationResult(target_count=3)

    async def accept(slot, finalized):
        if slot == 0:
            accepted.append(slot)
            prior_accepted.set()
        else:
            raise failure

    with pytest.raises(type(failure)) as caught:
        await asyncio.wait_for(builder.build_with_result(request(3), on_finalized_task=accept, outcome=outcome), 2)
    assert caught.value is failure and sibling_drained.is_set()
    assert accepted == [0] and len(outcome.finalized_tasks) == 1
    assert outcome.candidates[1].error_type == type(failure).__name__
    assert outcome.tool_usage.llm.call_count == 3 and outcome.tool_usage.llm.prompt_tokens == 30
    assert outcome.available_cost_usd > 0


async def test_seed_failure_accounts_for_every_requested_slot():
    class FailedSeedRunner(SeedRunner):
        async def generate_seeds(self, count, session, deadline, **kwargs):
            session.attempted_calls = 2
            session.missing_usage_calls = 2
            raise RuntimeError("seed stage failed")

    result = await MinerTaskDatasetBuilder(runner=FailedSeedRunner()).build_with_result(request())
    assert len(result.candidates) == 2
    assert all(c.status == "operational_failure" and c.seed is None for c in result.candidates)
    assert result.seed_attempted_calls == result.seed_missing_usage_calls == 2


async def test_blocked_publication_does_not_block_another_candidate():
    """A slow publication must not delay an eligible peer with equivalent question text."""
    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    builder._pipeline = Slots(duplicate=True)
    first_started = asyncio.Event()
    peer_committed = asyncio.Event()
    release = asyncio.Event()
    accepted = []

    async def accept(slot, finalized):
        if not first_started.is_set():
            first_started.set()
            await release.wait()
        accepted.append(slot)
        peer_committed.set()

    invocation = asyncio.create_task(builder.build_with_result(request(), on_finalized_task=accept))
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        await asyncio.wait_for(peer_committed.wait(), 1)
        assert len(accepted) == 1 and not invocation.done()
        release.set()
        result = await asyncio.wait_for(invocation, 1)
        assert set(accepted) == {0, 1}
        assert len(result.finalized_tasks) == 2
    finally:
        release.set()
        invocation.cancel()
        with suppress(asyncio.CancelledError):
            await invocation


@pytest.mark.parametrize("outcome_kind", ["confirmed_rejection", "uncertain_commit", "cancellation"])
async def test_concurrent_publication_preserves_committed_peer_and_usage(outcome_kind):
    """Failures and shutdown must join callbacks without erasing a peer's committed output or paid calls."""
    from harnyx_commons.llm.schema import LlmUsage
    from harnyx_commons.miner_task_generation import BatchGenerationResult
    from harnyx_commons.miner_task_generation.agent_runner import CallCapture
    from harnyx_commons.miner_task_generation.contracts import FinalizedTaskRejectedError

    class AccountedSlots(Slots):
        async def run(self, seed, *, session, **kwargs):
            capture = CallCapture(session, "solver", "gemini-3.8-flash", "vertex")
            capture.attempted = 1
            capture.account(LlmUsage(prompt_tokens=10, completion_tokens=5, web_search_calls=0))
            try:
                return await super().run(seed, session=session, **kwargs)
            finally:
                capture.finish()

    builder = MinerTaskDatasetBuilder(runner=SeedRunner())
    builder._pipeline = AccountedSlots(duplicate=True)
    calls = []
    committed = []
    first_started = asyncio.Event()
    peer_committed = asyncio.Event()
    release = asyncio.Event()
    callbacks_drained = []
    outcome = BatchGenerationResult(target_count=2)
    failure = (
        FinalizedTaskRejectedError("slot expired")
        if outcome_kind == "confirmed_rejection"
        else RuntimeError("commit uncertain")
    )

    async def accept(slot, finalized):
        calls.append(slot)
        try:
            if len(calls) == 1:
                first_started.set()
                await release.wait()
                if outcome_kind != "cancellation":
                    raise failure
            committed.append(slot)
            peer_committed.set()
        finally:
            callbacks_drained.append(slot)

    invocation = asyncio.create_task(builder.build_with_result(request(), on_finalized_task=accept, outcome=outcome))
    try:
        await asyncio.wait_for(first_started.wait(), 1)
        await asyncio.wait_for(peer_committed.wait(), 1)
        peer_slot = committed[0]
        if outcome_kind == "cancellation":
            invocation.cancel()
        release.set()
        expected_error = asyncio.CancelledError if outcome_kind == "cancellation" else type(failure)
        with pytest.raises(expected_error) as caught:
            await asyncio.wait_for(invocation, 1)
        if outcome_kind != "cancellation":
            assert caught.value is failure
            assert outcome.candidates[calls[0]].error_type == type(failure).__name__
        assert outcome.candidates[peer_slot].finalized is not None
        assert len(outcome.finalized_tasks) == (2 if outcome_kind == "cancellation" else 1)
        assert set(callbacks_drained) == {0, 1}
        assert outcome.tool_usage.llm.call_count == 2 and outcome.tool_usage.llm.prompt_tokens == 20
        assert outcome.available_cost_usd > 0
        assert all(candidate.attempted_calls == 1 for candidate in outcome.candidates)
    finally:
        release.set()
        invocation.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await invocation
