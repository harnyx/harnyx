from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.schema import (
    AbstractLlmRequest,
    GroundedLlmRequest,
    LlmChoice,
    LlmChoiceMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_commons.miner_task_generation import (
    MinerTaskDatasetBuilder,
    MinerTaskDatasetRequest,
    MinerTaskModelSpec,
)

pytestmark = pytest.mark.anyio("asyncio")


class _DuplicateOnlyGenerationProvider(LlmProviderPort):
    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        typed_request = cast(GroundedLlmRequest | LlmRequest, request)
        response = LlmResponse(
            id="duplicate-only-generation",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(
                            LlmMessageContentPart(
                                type="text",
                                text=json.dumps(
                                    {
                                        "tasks": [
                                            {"text": "What changed?"},
                                            {"text": "What changed?"},
                                            {"text": "What changed?"},
                                        ]
                                    }
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        if typed_request.postprocessor is None:
            return response
        result = typed_request.postprocessor(response)
        if not result.ok:
            raise RuntimeError(result.reason or "generation response rejected")
        return response.model_copy(update={"postprocessed": result.processed})


async def test_miner_task_dataset_builder_rejects_below_minimum_task_total() -> None:
    """Future failure: duplicate output must not silently shrink a requested dataset."""
    provider = _DuplicateOnlyGenerationProvider()
    builder = MinerTaskDatasetBuilder(
        generation_llm=provider,
        reference_llm=provider,
        clock=lambda: datetime(2026, 3, 6, tzinfo=UTC),
    )

    with pytest.raises(RuntimeError, match="generated unique task count below minimum_task_total"):
        await builder.build(
            MinerTaskDatasetRequest(
                batch_id=uuid4(),
                minimum_task_total=2,
                generation_task_buffer=1,
                generation_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="generation-test",
                    temperature=0.2,
                    max_output_tokens=256,
                ),
                reference_spec=MinerTaskModelSpec(
                    provider="vertex",
                    model="reference-test",
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
        )
