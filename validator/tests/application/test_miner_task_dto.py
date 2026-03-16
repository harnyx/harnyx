from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from harnyx_commons.domain.miner_task import MinerTask, Query, ReferenceAnswer
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec

_NOW = datetime.now(UTC)


def _task() -> MinerTask:
    return MinerTask(
        task_id=uuid4(),
        query=Query(text="Summarize the paper."),
        reference_answer=ReferenceAnswer(text="A short summary."),
        budget_usd=0.05,
    )


def test_batch_rejects_duplicate_artifact_ids() -> None:
    artifact_id = uuid4()

    with pytest.raises(ValidationError, match="artifact_id"):
        MinerTaskBatchSpec(
            batch_id=uuid4(),
            cutoff_at=_NOW.isoformat(),
            created_at=_NOW.isoformat(),
            tasks=(_task(),),
            artifacts=(
                ScriptArtifactSpec(uid=1, artifact_id=artifact_id, content_hash="a", size_bytes=10),
                ScriptArtifactSpec(uid=2, artifact_id=artifact_id, content_hash="b", size_bytes=20),
            ),
        )


def test_batch_rejects_extra_fields() -> None:
    payload = {
        "batch_id": uuid4(),
        "cutoff_at": _NOW.isoformat(),
        "created_at": _NOW.isoformat(),
        "tasks": [_task().model_dump(mode="python")],
        "artifacts": [
            {
                "uid": 1,
                "artifact_id": uuid4(),
                "content_hash": "hash",
                "size_bytes": 10,
                "extra": "boom",
            }
        ],
    }

    with pytest.raises(ValidationError, match="extra"):
        MinerTaskBatchSpec.model_validate(payload)
