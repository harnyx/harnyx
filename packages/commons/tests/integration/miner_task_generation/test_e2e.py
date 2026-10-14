"""The actual policy and renderer must produce a canonical task without extra construction models."""

import pytest
from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import FakeAuthor, FakeRunner
from public.packages.commons.tests.miner_task_generation.test_dataset_builder import request

from harnyx_commons.miner_task_generation import MinerTaskDatasetBuilder
from harnyx_commons.miner_task_generation.reference_generation import render_author_reference

pytestmark = [pytest.mark.integration, pytest.mark.anyio("asyncio")]


@pytest.mark.parametrize("structured", [False, True])
async def test_feedback_loop_materializes_public_reference_and_schema(structured, monkeypatch):
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_generation.public_addresses",
        lambda host, port: ("93.184.216.34",),
    )

    async def render(draft, reference):
        return await render_author_reference(draft, reference)

    author = FakeAuthor()
    builder = MinerTaskDatasetBuilder(author=author, runner=FakeRunner(), render_reference=render)
    result = await builder.build_with_result(
        request(1).model_copy(update={"plain_text_probability": 0.0 if structured else 1.0})
    )
    task = result.finalized_tasks[0].task
    assert bool(task.query.output_schema) == structured
    assert task.reference_answer.citations[0].excerpts[0].text == "item"
    assert len(author.feedback) == 1 and author.feedback[0].paired_miss
