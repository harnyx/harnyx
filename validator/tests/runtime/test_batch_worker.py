from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest

from caster_commons.domain.claim import MinerTaskClaim, ReferenceAnswer, Rubric
from caster_commons.domain.verdict import BINARY_VERDICT_OPTIONS
from caster_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec
from caster_validator.application.status import StatusProvider
from caster_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from caster_validator.runtime.evaluation_worker import EvaluationWorker


def _sample_batch() -> MinerTaskBatchSpec:
    claim = MinerTaskClaim(
        claim_id=uuid4(),
        text="example",
        rubric=Rubric(title="title", description="desc", verdict_options=BINARY_VERDICT_OPTIONS),
        reference_answer=ReferenceAnswer(verdict=1, justification="justified", citations=()),
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=uuid4(), content_hash="abc", size_bytes=1)
    return MinerTaskBatchSpec(
        batch_id=uuid4(),
        entrypoint="evaluate_criterion",
        cutoff_at_iso="2025-01-01T00:00:00Z",
        created_at_iso="2025-01-01T00:00:00Z",
        claims=(claim,),
        candidates=(artifact,),
    )


class FakeBatchService:
    """Fake batch service for testing."""

    def __init__(self) -> None:
        self.processed: list[MinerTaskBatchSpec] = []
        self.processed_event = threading.Event()

    def process(self, batch: MinerTaskBatchSpec) -> None:
        self.processed.append(batch)
        self.processed_event.set()

    async def process_async(self, batch: MinerTaskBatchSpec) -> None:
        self.process(batch)


@pytest.mark.anyio
async def test_evaluation_worker_drains_inbox():
    inbox = InMemoryBatchInbox()
    status = StatusProvider()
    fake_service = FakeBatchService()

    worker = EvaluationWorker(
        batch_service=fake_service,
        batch_inbox=inbox,
        status_provider=status,
    )
    inbox.put(_sample_batch())
    status.state.queued_batches = len(inbox)

    worker.start()
    assert await asyncio.to_thread(fake_service.processed_event.wait, timeout=1.0)
    await worker.stop(timeout=1.0)

    assert fake_service.processed
    assert status.state.queued_batches == 0
