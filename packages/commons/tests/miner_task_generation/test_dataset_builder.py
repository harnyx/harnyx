"""Protect fixed slots, reservation deadlines and partial acceptance."""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import (
    FakeAuthor,
    FakeRunner,
    reference,
)

from harnyx_commons.miner_task_generation import MinerTaskDatasetBuilder, MinerTaskDatasetRequest
from harnyx_commons.miner_task_generation.contracts import BatchTerminalGenerationError
from harnyx_commons.miner_task_generation.dataset_builder import UnderfilledDatasetError

pytestmark = pytest.mark.anyio("asyncio")


def request(count=2):
    return MinerTaskDatasetRequest(batch_id=uuid4(), created_at=datetime.now(UTC), minimum_task_total=count)


def builder(author=None, runner=None, **kwargs):
    return MinerTaskDatasetBuilder(
        author=author or FakeAuthor(), runner=runner or FakeRunner(), render_reference=reference, **kwargs
    )


async def test_one_fresh_context_per_slot_and_fixed_mode_draws():
    author = FakeAuthor()
    result = await builder(author, rng=random.Random(1)).build_with_result(request(8))  # noqa: S311
    assert len(result.candidates) == 8 and len(result.finalized_tasks) == 8
    assert len({key for key, _, _ in author.sessions}) == 8
    assert {mode for _, mode, _ in author.sessions} == {"structured", "plain_text"}
    assert all(len(item.cycles) == 1 for item in result.candidates)


async def test_expired_reservation_never_starts_author():
    author = FakeAuthor()
    result = await builder(author).build_with_result(
        request(1).model_copy(update={"created_at": datetime.now(UTC) - timedelta(hours=3)})
    )
    assert not author.sessions and not result.finalized_tasks
    assert result.candidates[0].error_type == "TimeoutError"


async def test_nonqualifying_slot_is_not_replaced_and_build_reports_underfill():
    author = FakeAuthor(status="exhausted")
    with pytest.raises(UnderfilledDatasetError) as caught:
        await builder(author).build(request(2))
    assert len(author.sessions) == 2 and len(caught.value.result.candidates) == 2


async def test_completed_slot_is_accepted_before_shared_peer_fault_and_peers_drain():
    accepted = asyncio.Event()
    drained = []

    class Author(FakeAuthor):
        async def run(self, **kwargs):
            slot = int(kwargs["session"].task_id.rsplit(":", 1)[1])
            if slot == 0:
                return await super().run(**kwargs)
            await accepted.wait()
            if slot == 1:
                raise BatchTerminalGenerationError("provider_auth", "bad credentials", stage="question_generation")
            try:
                await asyncio.Event().wait()
            finally:
                drained.append(slot)

    async def accept(slot, finalized):
        assert slot == 0
        accepted.set()

    from harnyx_commons.miner_task_generation import BatchGenerationResult

    outcome = BatchGenerationResult(target_count=3)
    with pytest.raises(BatchTerminalGenerationError):
        await builder(Author()).build_with_result(request(3), outcome=outcome, on_finalized_task=accept)
    assert len(outcome.finalized_tasks) == 1 and drained == [2]
